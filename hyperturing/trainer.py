# pyright: reportAttributeAccessIssue=none
import logging
import os
import socket
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.types import Tensor

from cs336_scaling.model import BasicsTransformerLM

from .compat import DEVICE_TYPE, compat
from .dataload import DataPrefetcher

logger = logging.getLogger(__name__)


def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def setup_distributed(rank: int, world_size: int, port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    backend = compat.get_dist_backend()
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    compat.set_device(rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_worker(
    rank: int,
    world_size: int,
    port: str,
    config: Dict[str, Any],
    train_data_path: str,
    vocab_size: int,
    context_length: int,
    return_dict: Dict[int, float],
    progress_queue: Optional[mp.Queue] = None,
):
    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    try:
        if world_size > 1:
            setup_distributed(rank, world_size, port)
        else:
            # Still set the device for single-process isolated run
            compat.set_device(rank)

        device = compat.get_device(rank)

        d_model = config["d_model"]
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        train_flops = config["train_flops"]

        local_batch_size = batch_size // world_size if world_size > 0 else batch_size
        if local_batch_size == 0:
            local_batch_size = 1

        d_ff = 4 * d_model
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            attn_pdrop=0.1,
            residual_pdrop=0.1,
        ).to(device)

        if world_size > 1:
            # FSDP for maximizing model size
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
                if (compat.is_npu or compat.is_cuda)
                else None,
                device_id=device,
            )

        compat.empty_cache()

        # C = 6 * N * D
        n_params = 12 * num_layers * (d_model**2)
        num_tokens = int(train_flops / (6 * n_params))
        num_steps = num_tokens // (batch_size * context_length)
        if num_steps <= 0:
            num_steps = 1

        data = np.load(train_data_path, mmap_mode="r")

        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(42 + rank)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=learning_rate / 10.0
        )

        model.train()
        last_loss = 0.0

        prefetcher = DataPrefetcher(
            data, local_batch_size, context_length, cpu_generator, device, num_steps
        )

        # Rationalize logging frequency: target ~10 logs per run
        log_interval = max(1, num_steps // 10)

        for step in range(num_steps):
            x, y = prefetcher.next()
            if x is None or y is None:
                break

            optimizer.zero_grad()
            logits: Tensor = model(x)  # type: ignore
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            last_loss = loss.item()

            if rank == 0:
                if (step + 1) % log_interval == 0 or step == num_steps - 1:
                    logger.info(
                        f"[Rank 0] Step {step + 1}/{num_steps}, Loss: {last_loss:.4f}"
                    )
                    if progress_queue:
                        progress_queue.put(
                            {
                                "step": step + 1,
                                "total_steps": num_steps,
                                "loss": last_loss,
                            }
                        )

        if rank == 0:
            return_dict[0] = last_loss
    except Exception as e:
        logger.error(f"Error in train_worker rank {rank}: {e}")
        raise e
    finally:
        if world_size > 1:
            cleanup_distributed()
        # Ensure model and optimizer are deleted
        if "model" in locals():
            del model
        if "optimizer" in locals():
            del optimizer
        compat.empty_cache()


class Trainer:
    def __init__(
        self, train_data_path: str, vocab_size: int = 32000, context_length: int = 512
    ):
        self.train_data_path = train_data_path
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.world_size = compat.device_count()
        if self.world_size == 0:
            self.world_size = 1
        logger.info(
            f"Trainer initialized for {DEVICE_TYPE} with world_size: {self.world_size}"
        )

    def train(
        self, config: Dict[str, Any], progress_queue: Optional[mp.Queue] = None
    ) -> float:
        vocab_size = config.get("vocab_size", self.vocab_size)
        context_length = config.get("context_length", self.context_length)

        # Always use mp.spawn to ensure process isolation for torch context
        # This prevents NPU/GPU memory from being held by the parent process after a crash
        ctx = mp.get_context("spawn")
        port = find_free_port()
        manager = ctx.Manager()
        return_dict = manager.dict()

        try:
            mp.spawn(  # type: ignore
                train_worker,
                args=(
                    self.world_size,
                    port,
                    config,
                    self.train_data_path,
                    vocab_size,
                    context_length,
                    return_dict,
                    progress_queue,
                ),
                nprocs=self.world_size,
                join=True,
            )
        except Exception as e:
            logger.error(f"mp.spawn failed: {e}")
            raise e
        finally:
            compat.empty_cache()

        return return_dict.get(0, 0.0)
