# pyright: reportAttributeAccessIssue=none
import os
from collections import deque

import numpy as np
import torch

from .compat import compat, npu


class DataPrefetcher:
    """Async prefetcher to overlap CPU data prep and H2D transfer with Device compute."""

    def __init__(self, data, batch_size, context_length, generator, device, num_steps):
        # Ensure int64 once to avoid per-step cast cost
        if data.dtype != np.int64:
            data = data.astype(np.int64, copy=False)
        self.data = data
        # Keep a torch CPU tensor to avoid numpy->torch conversion each step
        self.data_t = torch.from_numpy(self.data)
        self.batch_size = batch_size
        self.context_length = context_length
        self.generator = generator
        self.device = device
        self.num_steps = num_steps
        self._offsets_t = torch.arange(self.context_length, dtype=torch.int64)
        self.prefetch_batches = int(os.getenv("DATA_PREFETCH_BATCHES", "8"))
        self.prefetch_batches = max(1, self.prefetch_batches)
        self._cpu_queue = deque()
        self.stream = None
        if compat.is_cuda:
            self.stream = torch.cuda.Stream()
        elif compat.is_npu:
            self.stream = npu.Stream()

        self.next_x = None
        self.next_y = None
        self.step = 0
        self._preload()

    def _refill_cpu_queue(self):
        if len(self._cpu_queue) >= self.prefetch_batches:
            return

        remaining = self.prefetch_batches - len(self._cpu_queue)
        max_ix = len(self.data) - self.context_length - 1
        if remaining <= 0 or max_ix <= 0:
            return

        # Batch-generate random start indices for multiple batches at once
        ix = torch.randint(
            max_ix,
            (remaining, self.batch_size),
            generator=self.generator if self.generator.device.type == "cpu" else None,
        )
        indices = ix[:, :, None] + self._offsets_t[None, None, :]
        flat = indices.reshape(-1)
        x_all = torch.take(self.data_t, flat).view(
            remaining, self.batch_size, self.context_length
        )
        y_all = torch.take(self.data_t, flat + 1).view(
            remaining, self.batch_size, self.context_length
        )

        for i in range(remaining):
            self._cpu_queue.append((x_all[i], y_all[i]))

    def _preload(self):
        if self.step >= self.num_steps:
            self.next_x = None
            self.next_y = None
            return

        self._refill_cpu_queue()
        if not self._cpu_queue:
            self.next_x = None
            self.next_y = None
            return

        tx, ty = self._cpu_queue.popleft()

        if self.stream:
            # Use non_blocking transfer within the prefetch stream
            try:
                if compat.is_cuda:
                    with torch.cuda.stream(self.stream):
                        self.next_x = tx.to(self.device, non_blocking=True)
                        self.next_y = ty.to(self.device, non_blocking=True)
                elif compat.is_npu:
                    with npu.stream(self.stream):
                        self.next_x = tx.to(self.device, non_blocking=True)
                        self.next_y = ty.to(self.device, non_blocking=True)
            except Exception:
                # Fallback if stream context fails
                self.next_x = tx.to(self.device)
                self.next_y = ty.to(self.device)
        else:
            self.next_x = tx.to(self.device)
            self.next_y = ty.to(self.device)

        self.step += 1

    def next(self):
        if self.stream:
            if compat.is_cuda:
                torch.cuda.current_stream().wait_stream(self.stream)
            elif compat.is_npu:
                npu.current_stream().wait_stream(self.stream)

        x = self.next_x
        y = self.next_y
        self._preload()
        return x, y
