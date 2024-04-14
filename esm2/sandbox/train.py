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

import argparse
import contextlib
import logging
import os
import numpy as np
import sys

import datasets
import torch
import transformers

from torch import nn
from enum import Enum
from typing import Optional, Union, Tuple, List
from dataclasses import dataclass, field

from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler


from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    HfArgumentParser,
    is_torch_tpu_available,
    get_scheduler,
    set_seed,
)

import torch_xla
import torch_xla.debug.profiler as xp
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
from torch_xla.distributed.spmd.debugging import visualize_sharding
from torch_xla.distributed.spmd import XLAShardedTensor


class IntervalStrategy(Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max length of sequence for collator.",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="/outputs",
        help="Base directory for training artifacts.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run1",
        help="Run name.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps."
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Number of eval steps."
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/datasets/uniref",
        help="Path to a training dataset.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/hf_cache",
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--logging_interval",
        type=int,
        default=10,
        help="Number of steps between logging updates.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Number of steps between logging updates.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps between gradient optimization.",
    )

    parser.add_argument(
        "--steps_this_run",
        type=int,
        default=None,
        help="Max number of steps.",
    )

    parser.add_argument(
        "--num_slices",
        type=int,
        default=1,
        help="Num of slices to train on.",
    )

    parser.add_argument(
        "--model_dimension",
        type=int,
        default=1,
        help="Model axis dimension for XLA SPMD 2D sharding - weigts + activations.",
    )

    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss.",
    )

    args, _ = parser.parse_known_args()
    return args


class PoorsManTrainer:
    """Poor's man trainer."""

    def __init__(
        self,
        base_output_dir: str = "/outputs",
        train_steps: int = 100,
        eval_steps: int = 10,
        model_dimension: int = 1,
        num_slices: int = 1,


    ):

        self.spmd_mesh, self.input_sharding_spec = self._setup_spmd_logical_mesh_and_input_sharding(
            model_dimension=model_dimension,
            num_slices=num_slices
        )

        self.device = xm.xla_device()
        self.mp_train_loader = pl.MpDeviceLoader(self.train_loader, args.device, input_sharding=sharding_spec,
                                                 loader_prefetch_size=args.per_device_train_batch_size, device_prefetch_size=4)

        self.mp_eval_loader = pl.MpDeviceLoader(self.eval_loader, args.device, input_sharding=sharding_spec,
                                                loader_prefetch_size=args.per_device_eval_batch_size, device_prefetch_size=4)

    def _setup_spmd_logical_mesh_and_input_sharding(
            self,
            num_devices: int,
            model_dimension: int = 1,
            num_slices: int = 1,
    ):
        num_devices = xr.global_runtime_device_count()
        data_axis = num_devices // model_dimension // num_slices
        ici_mesh_shape = (1, data_axis, model_dimension)
        dcn_mesh_shape = (num_slices, 1, 1)
        spmd_mesh = xs.HybridMesh(ici_mesh_shape=ici_mesh_shape,
                                  dcn_mesh_shape=dcn_mesh_shape,
                                  axis_names=('dcn', 'data', 'model'))
        sharding_spec = xs.ShardingSpec(spmd_mesh, (('dcn', 'data'), None))
        return spmd_mesh, sharding_spec

    def _setup_xla_sharded_loader(
            self,
            dataloader: DataLoader):

        device = xm.xla_device()
        loader = pl.MpDeviceLoader(dataloader,
                                   device,
                                   input_sharding=self.input_sharding_spec,
                                   loader_prefetch_size=self.per_device_train_batch_size,
                                   device_prefetch_size=4)

        return loader

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        pass

    def _log_metrics(step, metrics):
        pass

    def _maybe_log_save_evaluate(self, step: int, loss: float):
        if step % 10 == 0:
            logger.info(f"Step: {step}, loss: {loss}")

    def _write_metrics(

    ):
        pass

    def _save_checkpoint(self):
        pass

    def train_step(self, batch):
        pass

    def eval_step(self, batch):
        pass

    def train_loop(
        self,
        model: nn.Module,
        optimizers: Tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR],
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model.train()
        # tr_loss = torch.tensor(0.0).to(device)
        self.model.zero_grad()
        for step, batch in enumerate(self.mp_train_loader):
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.model.zero_grad()
            self._maybe_log_save_evaluate(step, loss)
            if step >= training_steps:
                break


def get_train_dataloader(self) -> pl.MpDeviceLoader:
    train_loader = DataLoader(
        self.train_dataset,
        collate_fn=self.data_collator,
        batch_size=self.per_device_train_batch_size,
        num_workers=self.args.dataloader_num_workers,
        pin_memory=self.args.dataloader_pin_memory,
    )


def get_eval_dataloader(self) -> pl.MpDeviceLoader:
    pass


def main(args):

    print(xr.global_runtime_device_count())
    return

    # server = xp.start_server(9012)
    # logger.info(f'Profiling server started: {str(server)}')

    tokenizer_kwargs = {

    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_id, **tokenizer_kwargs)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability
    )

    # if training_args.xla_cache_path:
    #    readonly = training_args.xla_cache_single_writer and xr.process_index() != 0
    #    xr.initialize_cache(training_args.xla_cache_path, readonly)

    # TBD - Configure checkpoint manager
    tokenized_datasets = datasets.load_from_disk(data_args.dataset_dir)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    config_kwargs = {
        "torch_dtype": torch_dtype
    }
    config = AutoConfig.from_pretrained(
        model_args.model_id, **config_kwargs
    )
    model = AutoModelForMaskedLM.from_config(config)

    device = xm.xla_device()
    sharding_spec = xs.ShardingSpec(spmd_mesh, (('dcn', 'data'), None))

    model.to(device, dtype=torch_dtype)

    # TBD Shard the model

    # TBD Fully configure optimizer and scheduler

    # training_steps = training_args.max_steps
    # warmup_steps = training_args.warmup_steps
    training_steps = 1000
    warmup_steps = 0
    optimizer = AdamW(model.parameters(), training_args.learning_rate)
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps)

    trainer = PoorsManTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        optimizers=(optimizer, lr_scheduler)
    )

    trainer.train(training_steps=training_steps)


def test_loading():
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices // 2, 2)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("data", "model"))
    print(mesh)
    print(mesh.get_logical_mesh)
    print(mesh.shape())
    # st1 = xs.mark_sharding(t1, mesh, ('data', None))
    # assert isinstance(st1, XLAShardedTensor)
    # print('***')
    # for shard in st1.local_shards:
    #    print(shard)
    # print('***')

    t1 = torch.tensor(
        np.array([[0, 1],
                  [2, 3],
                  [4, 5],
                  [6, 7],
                  [8, 9],
                  [10, 11],
                  [12, 13],
                  [14, 15],
                  [16, 17],
                  [18, 19],
                  [20, 21],
                  [22, 23],
                  [24, 25],
                  [26, 27],
                  [28, 29],
                  [30, 31],
                  ]))

    dataset = TensorDataset(t1)

    random_sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size=4, sampler=random_sampler)
    sharding_spec = xs.ShardingSpec(mesh, ('data', None))
    xla_loader = pl.MpDeviceLoader(
        loader,
        device,
        input_sharding=sharding_spec
    )

    loader_iter = iter(xla_loader)

    batch = next(loader_iter)
    print(type(batch[0]))
    print(batch[0])


if __name__ == "__main__":
    assert xr.device_type() == 'TPU'
    xr.use_spmd()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    args = parse_args()

    test_loading()
    # main(args)
