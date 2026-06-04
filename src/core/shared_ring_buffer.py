"""
SharedRingBuffer — zero-copy ring buffer backed by multiprocessing.shared_memory.

Layout (per channel, one SharedMemory block):
  bytes [0:8]    uint64  write_ptr   (next write position, wraps at capacity)
  bytes [8:16]   uint64  total_count (total samples written, never wraps)
  bytes [16:16+cap*8]   float64[cap]  data
  bytes [16+cap*8:]     float64[cap]  timestamps

The writer atomically bumps write_ptr AFTER writing data, so readers never
see partially-written samples (on x86 a 64-bit aligned store is atomic).

Usage (single process / multiple threads):
  buf = SharedRingBuffer(capacity=200_000)
  buf.write(data_array, ts_array)
  ts, data = buf.read(n_samples=50_000)

Usage (multi-process):
  # Creator (main process):
  buf = SharedRingBuffer(capacity=200_000, create=True)
  name = buf.shm_name          # pass this to the worker process

  # Worker process:
  buf = SharedRingBuffer(capacity=200_000, name=name, create=False)
  buf.write(data_array, ts_array)
"""
from __future__ import annotations
from multiprocessing.shared_memory import SharedMemory
import numpy as np

_HDR = 16   # 2 × uint64 = 16 bytes


class SharedRingBuffer:
    def __init__(
        self,
        capacity: int,
        name: str | None = None,
        create: bool = True,
    ) -> None:
        self.capacity = capacity
        self._create = create
        total_bytes = _HDR + capacity * 8 * 2   # data + timestamps, float64

        if create:
            self._shm = SharedMemory(create=True, size=total_bytes)
        else:
            assert name is not None, "name required when create=False"
            self._shm = SharedMemory(name=name, create=False)

        buf = self._shm.buf
        self._ctrl = np.ndarray(2, dtype=np.uint64, buffer=buf[:_HDR])
        self._data = np.ndarray(capacity, dtype=np.float64,
                                buffer=buf[_HDR: _HDR + capacity * 8])
        self._ts = np.ndarray(capacity, dtype=np.float64,
                              buffer=buf[_HDR + capacity * 8:])

        if create:
            self._ctrl[:] = 0

    @property
    def shm_name(self) -> str:
        return self._shm.name

    @property
    def total_written(self) -> int:
        return int(self._ctrl[1])

    @property
    def write_ptr(self) -> int:
        return int(self._ctrl[0])

    # ------------------------------------------------------------------
    # Writer API (called from acquisition thread/process)
    # ------------------------------------------------------------------
    def write(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        n = len(data)
        if n == 0:
            return
        cap = self.capacity
        wp = int(self._ctrl[0])

        # Vectorised wrap-around write
        idx = np.arange(wp, wp + n, dtype=np.intp) % cap
        self._data[idx] = data
        self._ts[idx] = timestamps

        # Bump counters AFTER data — memory ordering matters
        new_wp = (wp + n) % cap
        new_total = int(self._ctrl[1]) + n
        self._ctrl[1] = new_total   # total first (reader only uses write_ptr)
        self._ctrl[0] = new_wp      # write_ptr last

    # ------------------------------------------------------------------
    # Reader API (called from Qt main thread)
    # ------------------------------------------------------------------
    def read(self, n_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        wp = int(self._ctrl[0])
        total = int(self._ctrl[1])
        available = min(total, self.capacity)
        count = min(n_samples or available, available)

        if count == 0:
            return np.empty(0), np.empty(0)

        start = (wp - count) % self.capacity
        idx = np.arange(start, start + count, dtype=np.intp) % self.capacity

        # .copy() — snapshot to avoid race with concurrent writer
        return self._ts[idx].copy(), self._data[idx].copy()

    def clear(self) -> None:
        self._ctrl[:] = 0

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        """Call only from the creating process when completely done."""
        if self._create:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
