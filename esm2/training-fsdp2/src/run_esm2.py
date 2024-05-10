# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import numpy as np
import os
import sys
import functools

import datasets
import torch
import transformers
from timeit import default_timer as timer

import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
import torch_xla.debug.metrics as met

from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union, List, Dict, Tuple

from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel as FSDPv2,
)
from torch_xla.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch_xla.distributed.fsdp import checkpoint_module
from torch_xla.amp.syncfree import AdamW as AdamWXLA
from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    EsmForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
    get_scheduler,
    SchedulerType,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    MaskedLMOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)

check_min_version("4.39.3")
logger = logging.getLogger(__name__)

xr.use_spmd()
assert xr.is_spmd() == True


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_id: Optional[str] = field(
        default="facebook/esm2_t33_650M_UR50D",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the tokenizer if different from model_id"},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )


@dataclass
class MoreTrainingArguments(TrainingArguments):
    profile_step: Optional[int] = field(
        default=-1, metadata={"help": "Step to profile"}
    )
    profile_logdir: Optional[str] = field(
        default=".", metadata={"help": "Directory to store the profile"}
    )
    profile_duration: Optional[int] = field(
        default="20000", metadata={"help": "Duration (ms) to capture profile"}
    )
    replicas: Optional[int] = field(
        default=1, metadata={"help": "Number of replicas for multislice."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: str = field(
        metadata={"help": "The input training data folder (a dir)."}
    )

    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )


class PoorsManTrainer:
    """Poor's man trainer."""

    def __init__(
        self,
        model: nn.Module,
        args: MoreTrainingArguments,
        data_collator: Optional[DataCollatorForLanguageModeling],
        train_dataset: Optional[Union[Dataset, IterableDataset]],
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]],
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
    ):
        self.args = args
        self.optimizer, self.lr_scheduler = optimizers
        self.model = model
        self.device = xm.xla_device()
        self.train_batch_size = args.per_device_train_batch_size
        self.eval_batch_size = args.per_device_eval_batch_size
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.replicas = self.args.replicas
        self.use_fsdp = True if args.fsdp else False

        # Set up SPMD mesh
        num_devices = xr.global_runtime_device_count()
        if self.replicas == 1:
            xs.set_global_mesh(
                xs.Mesh(
                    np.array(range(num_devices)),
                    (num_devices, 1),
                    axis_names=("fsdp", "tensor"),
                )
            )
            self.input_sharding_spec = xs.ShardingSpec(
                xs.get_global_mesh(), ("fsdp", None)
            )
        else:
            logger.info(f"Creating Mesh for {self.replicas} replicas")
            dcn_axis = self.replicas
            model_axis = 1  # For FSDP this is 1.
            fsdp_axis = num_devices // model_axis // dcn_axis
            ici_mesh_shape = (1, fsdp_axis, model_axis)
            dcn_mesh_shape = (dcn_axis, 1, 1)
            spmd_mesh = xs.HybridMesh(
                ici_mesh_shape=ici_mesh_shape,
                dcn_mesh_shape=dcn_mesh_shape,
                axis_names=("dcn", "fsdp", "model"),
            )
            xs.set_global_mesh(spmd_mesh)
            self.input_sharding_spec = xs.ShardingSpec(
                mesh=xs.get_global_mesh(), partition_spec=(("dcn", "fsdp"), None)
            )

        logger.info(f"Logical mesh shape: {xs.get_global_mesh().shape()}")
        logger.info(f"Input sharding: {self.input_sharding_spec}")

    def _check_model_optimizer_placement(self, model, optimizer):
        for param in model.parameters():
            model_device = param.device
            break
        for param_group in optimizer.param_groups:
            if len(param_group["params"]) > 0:
                optimizer_device = param_group["params"][0].device
                break
        if model_device != optimizer_device:
            raise ValueError(
                "The model and the optimizer parameters are not on the same device."
            )

    def _get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader = DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )
        loader = pl.MpDeviceLoader(
            dataloader,
            self.device,
            input_sharding=self.input_sharding_spec,
            loader_prefetch_size=self.train_batch_size,
            device_prefetch_size=4,
        )
        return loader

    def _get_eval_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader = DataLoader(
            self.eval_dataset,
            collate_fn=self.data_collator,
            batch_size=self.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )
        loader = pl.MpDeviceLoader(
            dataloader,
            self.device,
            input_sharding=self.input_sharding_spec,
            loader_prefetch_size=self.eval_batch_size,
            device_prefetch_size=4,
        )
        return loader

    def _wrap_model(self, model):

        if self.use_fsdp:
            auto_wrap_policy = None
            auto_wrapper_callable = None
            default_transformer_cls_names_to_wrap = getattr(
                model, "_no_split_modules", None
            )
            default_transformer_cls_names_to_wrap = None
            fsdp_transformer_layer_cls_to_wrap = self.args.fsdp_config.get(
                "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
            )
            if self.args.fsdp_config["min_num_params"] > 0:
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy,
                    min_num_params=self.args.fsdp_config["min_num_params"],
                )
            elif fsdp_transformer_layer_cls_to_wrap is not None:
                transformer_cls_to_wrap = set()
                for layer_class in fsdp_transformer_layer_cls_to_wrap:
                    transformer_cls = get_module_class_from_name(model, layer_class)
                    if transformer_cls is None:
                        raise Exception(
                            "Could not find the transformer layer class to wrap in the model."
                        )
                    else:
                        transformer_cls_to_wrap.add(transformer_cls)
                logger.info(f"ESM2 classes to wrap: {transformer_cls_to_wrap}")
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls=transformer_cls_to_wrap,
                )

            if self.args.fsdp_config["xla_fsdp_grad_ckpt"]:
                # Apply gradient checkpointing to auto-wrapped sub-modules if specified
                logger.info("Enabling gradient checkpointing")

                def auto_wrapper_callable(m, *args, **kwargs):
                    target_cls = FSDPv2
                    return target_cls(checkpoint_module(m), *args, **kwargs)

            def shard_output(output, mesh):
                real_output = None
                if isinstance(output, torch.Tensor):
                    real_output = output
                elif isinstance(output, tuple):
                    real_output = output[0]
                elif isinstance(output, CausalLMOutputWithPast):
                    real_output = output.logits
                elif isinstance(output, MaskedLMOutput):
                    real_output = output.logits
                if real_output is None:
                    raise ValueError(
                        "Something went wrong, the output of the model shouldn't be `None`"
                    )
                if self.replicas > 1:
                    xs.mark_sharding(real_output, mesh, (("dcn", "fsdp"), None, None))
                else:
                    xs.mark_sharding(real_output, mesh, ("fsdp", None, None))

            model = FSDPv2(
                model,
                shard_output=shard_output,
                auto_wrap_policy=auto_wrap_policy,
                auto_wrapper_callable=auto_wrapper_callable,
            )

            # def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
            #     loss = optimizer.step(**optimizer_args)
            #     if barrier:
            #         xm.mark_step()
            #     return loss

            # xm.optimizer_step = patched_optimizer_step
        else:
            logger.info("Using DDP. Model not wrapped")

        return model

    def _log_metrics(self, step, start_time, loss, sample_count):
        xm.mark_step()
        loss = loss.item()
        now = timer()
        elapsed_time = now - start_time
        samples_per_sec = sample_count / elapsed_time
        logger.info(
            f"Step: {step}, loss: {loss:0.4f}, Step time: {elapsed_time:0.2f} Samples: {sample_count} Samples/sec: {samples_per_sec:0.4f}"
        )
        self.run_history["step_history"].append(
            {
                "step": step,
                "loss": loss,
                "elapsed_time": elapsed_time,
                "sample_count": sample_count,
            }
        )

    def _save_checkpoint(self):
        pass

    def train_loop(self):
        self.model.train()
        self.model.zero_grad()
        # TBD restart from a given step. May skip the x number of batches
        # For now we assume that we wil never train for mor than one epoch
        start_step = 1
        max_step = self.args.max_steps
        train_loader = self._get_train_dataloader()
        train_iterator = iter(train_loader)
        model = self._wrap_model(self.model)

        self.optimizer = AdamW(
            params=model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        steps = self.args.max_steps
        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=steps,
        )
        self._check_model_optimizer_placement(self.model, self.optimizer)

        logger.info("Starting training")
        logger.info(f"    Using {'FSDP' if self.use_fsdp else 'DDP'}")
        logger.info(f"    Start step: {start_step}")
        logger.info(f"    Max step: {max_step}")
        logger.info(f"    Global batch size: {self.train_batch_size}")

        self.run_history = {"step_history": [], "elapsed_time": 0.0}
        sample_count = self.train_batch_size * self.args.logging_steps
        total_steps = 0
        start_time = timer()
        adjusted_total_steps = -10
        for step in range(start_step, max_step + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                break

            if adjusted_total_steps == 0:
                adjusted_start_time = timer()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            model.zero_grad()

            if step % self.args.logging_steps == 0:
                # xm.add_step_closure(
                #    self._log_metrics,
                #    args=(step, start_time, loss, sample_count),
                #    run_async=False,
                # )
                self._log_metrics(step, start_time, loss, sample_count)
                start_time = timer()
            total_steps += 1
            adjusted_total_steps += 1

            # Capture profile at the prefer step
            if step == self.args.profile_step:
                # Wait until device execution catches up to tracing before triggering the profile. This will
                # interrupt training slightly on the hosts which are capturing, but by waiting after tracing
                # for the step, the interruption will be minimal.
                xm.wait_device_ops()
                xp.trace_detached(
                    "127.0.0.1:9012",
                    self.args.profile_logdir,
                    self.args.profile_duration,
                )

        adjusted_elapsed_time = timer() - adjusted_start_time

        logger.info("Finished training run")
        logger.info(self.run_history)

        logger.info("Performance summary")
        logger.info(f"  Number of steps: {adjusted_total_steps}")
        logger.info(f"  Elapsed time: {adjusted_elapsed_time:0.2f}")
        logger.info(
            f"  Steps per second: {adjusted_total_steps/adjusted_elapsed_time:0.2f}"
        )


def main():
    # Parse CLI arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MoreTrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)
    server = xp.start_server(9012)
    logger.info(f"Profiling server started: {str(server)}")

    tokenizer_name = (
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_id
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    config = AutoConfig.from_pretrained(
        model_args.model_id,
        vocab_size=len(tokenizer),
        torch_dtype=model_args.torch_dtype,
    )
    model = EsmForMaskedLM(config)
    logger.info(f"Loaded model: {model_args.model_id}")
    logger.info(f"Model parameters: {model.num_parameters()}")

    model = apply_xla_patch_to_nn_linear(model, xs.xla_patched_nn_linear_forward)

    model = model.to(xm.xla_device(), dtype=getattr(torch, model_args.torch_dtype))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
    )

    # Load datasets
    raw_datasets = datasets.load_from_disk(data_args.dataset_dir)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    trainer = PoorsManTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(None, None),
    )

    results = trainer.train_loop()
    logger.info("Training results:")
    logger.info(results)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
