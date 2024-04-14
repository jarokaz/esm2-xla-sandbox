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

import torch_xla

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
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
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
from transformers.modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput, BaseModelOutputWithPastAndCrossAttentions

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
        default="facebook/esm2_t33_650M_UR50D", metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    model_dimension: Optional[int] = field(
        default=1, metadata={"help": "The dimension of the model axis"}
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
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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
        metadata={
            "help": "Ratio of tokens to mask for masked language modeling loss"},
    )


class PoorsManTrainer:
    """Poor's man trainer."""

    def __init__(
        self,
        model: nn.Module,
        model_dimension: int,
        args: TrainingArguments,
        data_collator: Optional[DataCollatorForLanguageModeling],
        train_dataset: Optional[Union[Dataset, IterableDataset]],
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]],
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR],
    ):
        self.args = args
        self.optimizer, self.lr_scheduler = optimizers
        self.model = model
        self._check_model_optimizer_placement(self.model, self.optimizer)
        self.device = xm.xla_device()
        self.train_batch_size = args.per_device_train_batch_size
        self.eval_batch_size = args.per_device_eval_batch_size
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.model_dimension = model_dimension

        # Set up SPMD mesh

        self.input_sharding_spec = xs.ShardingSpec(
            xs.get_global_mesh(), ("data", None))
        logger.info(f"Logical mesh shape: {xs.get_global_mesh().shape()}")
        logger.info(f"Input sharding: {self.input_sharding_spec}")

        self.wrapped_model = self._wrap_model(self.model)

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
            drop_last=self.args.dataloader_drop_last
        )
        loader = pl.MpDeviceLoader(dataloader,
                                   self.device,
                                   input_sharding=self.input_sharding_spec,
                                   loader_prefetch_size=self.train_batch_size,
                                   device_prefetch_size=4)
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
            drop_last=self.args.dataloader_drop_last
        )
        loader = pl.MpDeviceLoader(dataloader,
                                   self.device,
                                   input_sharding=self.input_sharding_spec,
                                   loader_prefetch_size=self.eval_batch_size,
                                   device_prefetch_size=4)
        return loader

    def _wrap_model(self, model):
        mesh = xs.get_global_mesh()

        for name, param in model.named_parameters():
            if len(param.shape) == 1:
                continue

            if 'word_embeddings' in name or 'position_embeddings' in name:
                xs.mark_sharding(param, mesh,
                                 ('model', 'data')
                                 )

            if 'query' in name or 'key' in name or 'value' in name:
                xs.mark_sharding(param, mesh,
                                 ('model', 'data')
                                 )

            if 'attention.output.dense' in name:
                xs.mark_sharding(param, mesh,
                                 ('data', 'model')
                                 )

            if 'intermediate.dense' in name:
                xs.mark_sharding(param, mesh,
                                 ('data', 'model')
                                 )

            if 'output.dense' in name and not 'attention.output.dense' in name:
                xs.mark_sharding(param, mesh,
                                 ('model', 'data')
                                 )
            if 'lm_head.dense' in name or 'lm_head.decoder' in name:
                xs.mark_sharding(param, mesh,
                                 ('data', 'model'))

            print(f'{name} {torch_xla._XLAC._get_xla_sharding_spec(param)}')

        print("Sharding model complete", flush=True)

        return model

    def _log_metrics(self, step, start_time, loss,  sample_count):
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
        model = self.wrapped_model

        logger.info("Starting training")
        logger.info(f"    Start step: {start_step}")
        logger.info(f"    Max step: {max_step}")

        self.run_history = {
            "step_history": [],
            "elapsed_time": 0.0
        }
        sample_count = self.train_batch_size * self.args.logging_steps
        start_time = timer()
        total_steps = 0
        adjusted_total_steps = -5
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

        adjusted_elapsed_time = timer() - adjusted_start_time

        logger.info("Finished training run")
        logger.info(self.run_history)

        logger.info("Performance summary")
        logger.info(f"  Number of steps: {adjusted_total_steps}")
        logger.info(f"  Elapsed time: {adjusted_elapsed_time:0.2f}")
        logger.info(
            f"  Steps per second: {adjusted_total_steps/adjusted_elapsed_time:0.2f}")


def main():
    # Parse CLI arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    num_devices = xr.global_runtime_device_count()
    xs.set_global_mesh(xs.Mesh(np.array(range(num_devices)),
                               (num_devices//model_args.model_dimension, model_args.model_dimension), axis_names=("data", "model")))

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
    # server = xp.start_server(9012)
    # logger.info(f'Profiling server started: {str(server)}')

    # Create the model
    # TBD
    # Currently we intialize all model weights at once on a host. Since v5e has a limited CPU memory
    # we may need to modify this process for larger models
    # We also need to add loading the existing checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id)
    config = AutoConfig.from_pretrained(
        model_args.model_id,
        vocab_size=len(tokenizer),
        torch_dtype=model_args.torch_dtype,
    )
    model = EsmForMaskedLM(config)

    model = apply_xla_patch_to_nn_linear(
        model, xs.xla_patched_nn_linear_forward)

    logger.info(f"Number of model parameters: {model.num_parameters()}")
    model = model.to(xm.xla_device(), dtype=getattr(
        torch, model_args.torch_dtype))

    optimizer = AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon)

    steps = training_args.max_steps
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=steps
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability
    )

    # Load datasets
    raw_datasets = datasets.load_from_disk(data_args.dataset_dir)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    trainer = PoorsManTrainer(
        model=model,
        model_dimension=model_args.model_dimension,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, lr_scheduler)
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
