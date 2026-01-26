# pyright: reportAttributeAccessIssue=none
import numpy as np
import torch

from .compat import compat, npu


class DataPrefetcher:
    """Async prefetcher to overlap CPU data prep and H2D transfer with Device compute."""

    def __init__(self, data, batch_size, context_length, generator, device, num_steps):
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.generator = generator
        self.device = device
        self.num_steps = num_steps
        self.stream = None
        if compat.is_cuda:
            self.stream = torch.cuda.Stream()
        elif compat.is_npu:
            self.stream = npu.Stream()

        self.next_x = None
        self.next_y = None
        self.step = 0
        self._preload()

    def _preload(self):
        if self.step >= self.num_steps:
            self.next_x = None
            self.next_y = None
            return

        # Sample indices on CPU to avoid device-to-host sync for numpy indexing
        ix = torch.randint(
            len(self.data) - self.context_length - 1,
            (self.batch_size,),
            generator=self.generator if self.generator.device.type == "cpu" else None,
        )

        ix_np = ix.numpy()
        x_np = np.stack(
            [self.data[i : i + self.context_length].astype(np.int64) for i in ix_np]
        )
        y_np = np.stack(
            [
                self.data[i + 1 : i + self.context_length + 1].astype(np.int64)
                for i in ix_np
            ]
        )

        tx = torch.from_numpy(x_np)
        ty = torch.from_numpy(y_np)

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
