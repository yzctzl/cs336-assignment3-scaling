# ruff: noqa: F401
# pyright: reportPossiblyUnboundVariable=none
import logging

import torch

logger = logging.getLogger(__name__)

# Basic hardware status
HAS_NPU = False
HAS_CUDA = torch.cuda.is_available()

try:
    # import torch_npu
    from torch_npu import npu
    # from torch_npu.contrib import transfer_to_npu

    HAS_NPU = npu.is_available()
except (ImportError, AttributeError):
    pass


class HardwareCompat:
    @property
    def is_npu(self) -> bool:
        return HAS_NPU

    @property
    def is_cuda(self) -> bool:
        return HAS_CUDA

    @property
    def device_type(self) -> str:
        if self.is_npu:
            return "npu"
        if self.is_cuda:
            return "cuda"
        return "cpu"

    def device_count(self) -> int:
        if self.is_npu:
            return npu.device_count()
        if self.is_cuda:
            return torch.cuda.device_count()
        return 0

    def set_device(self, rank: int) -> None:
        if self.is_npu:
            npu.set_device(rank)
        elif self.is_cuda:
            torch.cuda.set_device(rank)

    def get_device(self, rank: int) -> torch.device:
        return (
            torch.device(f"{self.device_type}:{rank}")
            if self.device_count() > 0
            else torch.device("cpu")
        )

    def get_dist_backend(self) -> str:
        if self.is_npu:
            return "hccl"
        if self.is_cuda:
            return "nccl"
        return "gloo"

    def empty_cache(self) -> None:
        if self.is_npu:
            npu.empty_cache()
        elif self.is_cuda:
            torch.cuda.empty_cache()


compat = HardwareCompat()
DEVICE_TYPE = compat.device_type

__all__ = ["npu", "compat", "DEVICE_TYPE"]
