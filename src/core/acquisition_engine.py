"""
Acquisition engine — bridges hardware drivers to Channel ring buffers.

Two modes:
  ThreadedEngine   — driver runs in a thread (GIL released for numpy/I/O ops).
                     Good for <50 kHz per channel.

  MultiprocessEngine — driver runs in a separate process + SharedRingBuffer.
                       True GIL bypass. Required for >50 kHz aggregate.

Both expose the same interface:
  .start()
  .stop()
  .reset()
  .elapsed_time  (property)
  data_ready  (Qt signal, emitted after each batch)
  status_changed (Qt signal, str)
"""
from __future__ import annotations
import multiprocessing
import time
from typing import List

import numpy as np
from PyQt6.QtCore import QThread, QTimer, pyqtSignal

from .channel import Channel
from .shared_ring_buffer import SharedRingBuffer
from ..hardware.base import BaseDriver, SampleCallback


# ---------------------------------------------------------------------------
# Threaded engine (default, < 50 kHz)
# ---------------------------------------------------------------------------

class ThreadedAcquisitionEngine(QThread):
    data_ready = pyqtSignal()
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        driver: BaseDriver,
        channels: List[Channel],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._driver = driver
        self.channels = channels
        self._running = False
        self._elapsed = 0.0

    def start_acquisition(self) -> None:
        self._running = True
        # The driver fires its own thread; we just need to forward callbacks to channels
        self._driver.start(self._on_data)
        self.status_changed.emit("running")

    def stop_acquisition(self) -> None:
        self._running = False
        self._driver.stop()
        self.status_changed.emit("stopped")

    def reset(self) -> None:
        self._elapsed = 0.0
        for ch in self.channels:
            ch.clear()

    def _on_data(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Called from driver thread. Push samples into Channel buffers."""
        for i, ch in enumerate(self.channels):
            if i >= data.shape[0]:
                break
            if ch.enabled:
                ch.push(data[i], timestamps)

        if len(timestamps) > 0:
            self._elapsed = float(timestamps[-1])

        self.data_ready.emit()

    @property
    def elapsed_time(self) -> float:
        return self._elapsed

    # QThread.run is unused — driver owns its own thread
    def run(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Multiprocess engine (> 50 kHz / GIL-sensitive workloads)
# ---------------------------------------------------------------------------

def _worker_process(
    driver_cls_path: str,
    driver_kwargs: dict,
    shm_names: list[str],
    capacity: int,
    stop_event,         # multiprocessing.Event
) -> None:
    """Entry point for worker subprocess. Imports driver, writes to shared memory."""
    import importlib

    # Dynamic import: "src.hardware.sounddevice_driver.SounddeviceDriver"
    module_path, cls_name = driver_cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    DriverClass = getattr(module, cls_name)

    driver: BaseDriver = DriverClass(**driver_kwargs)

    buffers = [
        SharedRingBuffer(capacity, name=name, create=False)
        for name in shm_names
    ]

    def _callback(data: np.ndarray, timestamps: np.ndarray) -> None:
        for i, buf in enumerate(buffers):
            if i < data.shape[0]:
                buf.write(data[i], timestamps)

    driver.start(_callback)
    stop_event.wait()
    driver.stop()

    for buf in buffers:
        buf.close()


class MultiprocessAcquisitionEngine(QThread):
    """
    Acquisition runs in a subprocess via shared memory.
    A QTimer in the main process polls the shared buffers and updates Channels.
    """
    data_ready = pyqtSignal()
    status_changed = pyqtSignal(str)

    BUFFER_SECONDS = 30.0   # ring buffer covers this many seconds

    def __init__(
        self,
        driver_cls_path: str,       # e.g. "src.hardware.sounddevice_driver.SounddeviceDriver"
        driver_kwargs: dict,
        channels: List[Channel],
        sample_rate: float,
        poll_interval_ms: int = 50,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._driver_cls_path = driver_cls_path
        self._driver_kwargs = driver_kwargs
        self.channels = channels
        self.sample_rate = sample_rate
        self._poll_ms = poll_interval_ms

        capacity = int(self.BUFFER_SECONDS * sample_rate)
        self._buffers = [
            SharedRingBuffer(capacity, create=True) for _ in channels
        ]
        self._last_read: list[int] = [0] * len(channels)  # total_written at last read

        self._stop_event: multiprocessing.Event | None = None
        self._proc: multiprocessing.Process | None = None
        self._timer: QTimer | None = None
        self._elapsed = 0.0

    def start_acquisition(self) -> None:
        self._stop_event = multiprocessing.Event()
        shm_names = [b.shm_name for b in self._buffers]

        self._proc = multiprocessing.Process(
            target=_worker_process,
            args=(
                self._driver_cls_path,
                self._driver_kwargs,
                shm_names,
                self._buffers[0].capacity,
                self._stop_event,
            ),
            daemon=True,
        )
        self._proc.start()

        # Poll timer in main process
        self._timer = QTimer()
        self._timer.setInterval(self._poll_ms)
        self._timer.timeout.connect(self._poll_buffers)
        self._timer.start()

        self.status_changed.emit("running")

    def stop_acquisition(self) -> None:
        if self._timer:
            self._timer.stop()
        if self._stop_event:
            self._stop_event.set()
        if self._proc:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.kill()
        self.status_changed.emit("stopped")

    def reset(self) -> None:
        for buf in self._buffers:
            buf.clear()
        self._last_read = [0] * len(self.channels)
        self._elapsed = 0.0
        for ch in self.channels:
            ch.clear()

    def _poll_buffers(self) -> None:
        """Read new samples from shared memory and push to Channels."""
        updated = False
        for i, (ch, buf) in enumerate(zip(self.channels, self._buffers)):
            total = buf.total_written
            new_samples = total - self._last_read[i]
            if new_samples <= 0:
                continue
            self._last_read[i] = total

            ts, data = buf.read(n_samples=new_samples)
            if len(data) > 0 and ch.enabled:
                ch.push(data, ts)
                self._elapsed = float(ts[-1])
                updated = True

        if updated:
            self.data_ready.emit()

    def close(self) -> None:
        """Release shared memory. Call on application exit."""
        for buf in self._buffers:
            buf.close()
            buf.unlink()

    @property
    def elapsed_time(self) -> float:
        return self._elapsed

    def run(self) -> None:
        pass
