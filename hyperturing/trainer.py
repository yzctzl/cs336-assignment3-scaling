import logging
import os
import socket
from multiprocessing import shared_memory
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.types import Tensor

from cs336_scaling.model import BasicsTransformerLM

from .compat import DEVICE_TYPE, compat
from .dataload import DataPrefetcher

logger = logging.getLogger(__name__)


def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def is_port_conflict_error(err: Exception) -> bool:
    msg = str(err).lower()
    port_markers = [
        "address already in use",
        "eaddrinuse",
        "bind the ip port",
        "port have been bound already",
        "failed to bind the ip port",
        "ej0003",
    ]
    return any(m in msg for m in port_markers)


def setup_distributed(rank: int, world_size: int, port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    backend = compat.get_dist_backend()
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_worker(
    rank: int,
    world_size: int,
    port: str,
    config: Dict[str, Any],
    train_data_path: str,
    shm_info: Optional[Dict[str, Any]],
    vocab_size: int,
    context_length: int,
    device_ids: Optional[list[int]],
    return_dict: Dict[int, float],
):
    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    shm = None
    try:
        device_id: Optional[int] = None
        if device_ids:
            device_id = device_ids[rank]
            compat.set_device(device_id)
        else:
            compat.set_device(rank)

        if world_size > 1:
            setup_distributed(rank, world_size, port)

        if compat.device_count() > 0:
            device = torch.device(f"{compat.device_type}:{device_id if device_id is not None else rank}")
        else:
            device = torch.device("cpu")

        if compat.is_cuda:
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

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
        use_checkpoint = os.getenv("TRAINING_GRAD_CHECKPOINT", "0") == "1"
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            attn_pdrop=0.1,
            residual_pdrop=0.1,
            use_checkpoint=use_checkpoint,
        ).to(device)

        if world_size > 1:
            # Use DDP (no parameter sharding) for ZeRO-2-like behavior.
            ddp_kwargs = {
                "device_ids": [device_id] if device_id is not None else None,
                "output_device": device_id if device_id is not None else None,
                "broadcast_buffers": False,
                "gradient_as_bucket_view": True,
            }
            if os.getenv("DDP_STATIC_GRAPH", "1") == "1":
                ddp_kwargs["static_graph"] = True
            model = DDP(model, **ddp_kwargs)

        compat.empty_cache()

        # n_params = 12 * num_layers * (d_model**2)
        # num_tokens = int(train_flops / (6 * n_params))
        # num_steps = num_tokens // (batch_size * context_length)
        # if num_steps <= 0:
        #     num_steps = 1

        n_logic = 12 * num_layers * (d_model**2)
        n_emb = vocab_size * d_model 
        n_total = n_logic + n_emb 
        num_tokens = int(train_flops / (6 * n_total))
        num_steps = num_tokens // (batch_size * context_length)

        # Shared Memory / Direct Memory Loading
        data = None
        shm = None
        if shm_info:
            try:
                shm = shared_memory.SharedMemory(name=shm_info["name"])
                data = np.ndarray(
                    shm_info["shape"], dtype=shm_info["dtype"], buffer=shm.buf
                )
                logger.info(f"Attached to SharedMemory: {shm_info['name']}")
            except Exception as e:
                logger.error(f"Failed to attach to SharedMemory: {e}")
                data = np.load(train_data_path)
        else:
            data = np.load(train_data_path)

        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(42 + rank)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=learning_rate / 10.0
        )

        amp_enabled = (
            os.getenv("TRAINING_AMP", "1") == "1"
            and (compat.is_cuda or compat.is_npu)
        )
        amp_dtype = os.getenv("TRAINING_AMP_DTYPE", "fp16").lower()
        if amp_dtype == "bf16":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
        scaler = None
        if amp_enabled and compat.is_cuda:
            scaler = torch.cuda.amp.GradScaler()
        elif amp_enabled and compat.is_npu:
            try:
                from torch_npu.amp import GradScaler as NPUGradScaler  # type: ignore

                scaler = NPUGradScaler()
            except Exception:
                scaler = None

        model.train()
        prefetcher = DataPrefetcher(
            data, local_batch_size, context_length, cpu_generator, device, num_steps
        )

        log_interval = max(1, num_steps // 16)

        for step in range(num_steps):
            x, y = prefetcher.next()
            if x is None or y is None:
                break

            optimizer.zero_grad()
            if amp_enabled:
                with torch.autocast(
                    device_type=compat.device_type,
                    dtype=autocast_dtype,
                    enabled=True,
                ):
                    logits: Tensor = model(x)  # type: ignore
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            else:
                logits = model(x)  # type: ignore
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if rank == 0 and ((step + 1) % log_interval == 0 or step == num_steps - 1):
                last_loss = loss.item()
                return_dict[0] = last_loss
                logger.info(
                    f"[Rank 0] Step {step + 1}/{num_steps}, Loss: {last_loss:.4f}"
                )

    except Exception as e:
        logger.error(f"Error in train_worker rank {rank}: {e}")
        raise e
    finally:
        if world_size > 1:
            cleanup_distributed()
        if "model" in locals():
            del model
        if "optimizer" in locals():
            del optimizer
        if "shm" in locals() and shm is not None:
            shm.close()
        compat.empty_cache()


class Trainer:
    def __init__(
        self,
        train_data_path: str,
        vocab_size: int = 32000,
        context_length: int = 512,
        shm_info: Optional[Dict[str, Any]] = None,
    ):
        self.train_data_path = train_data_path
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.shm_info = shm_info
        self.world_size = compat.device_count()
        if self.world_size == 0:
            self.world_size = 1
        logger.info(
            f"Trainer initialized for {DEVICE_TYPE} with world_size: {self.world_size}"
        )

    def train(self, config: Dict[str, Any], device_ids: Optional[list[int]] = None) -> float:
        vocab_size = config.get("vocab_size", self.vocab_size)
        context_length = config.get("context_length", self.context_length)

        ctx = mp.get_context("spawn")
        world_size = len(device_ids) if device_ids else self.world_size
        max_port_retries = 3
        last_error: Optional[Exception] = None

        for attempt in range(1, max_port_retries + 1):
            port = find_free_port()
            with ctx.Manager() as manager:
                return_dict = manager.dict()
                try:
                    mp.spawn(  # type: ignore
                        train_worker,
                        args=(
                            world_size,
                            port,
                            config,
                            self.train_data_path,
                            self.shm_info,
                            vocab_size,
                            context_length,
                            device_ids,
                            return_dict,
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                    return return_dict.get(0, 0.0)
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    # Expanded OOM detection keywords based on NPU error logs
                    oom_keywords = [
                        "out of memory",
                        "acl api failed",
                        "failed to allocate",
                        "tried to allocate",
                        "memory_allocation_failure",
                        "npu out of memory",
                    ]
                    if any(k in msg for k in oom_keywords):
                        logger.warning(
                            "Trainer worker failed with OOM error. Propagating up."
                        )
                        break
                    if is_port_conflict_error(e) and attempt < max_port_retries:
                        logger.warning(
                            f"Distributed port conflict detected. Retrying with a new port (attempt {attempt + 1}/{max_port_retries})."
                        )
                        continue
                    logger.error(f"mp.spawn failed: {e}")
                    break
                finally:
                    compat.empty_cache()

        if last_error is not None:
            raise last_error
        return 0.0
