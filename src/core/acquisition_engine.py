"""
Acquisition engine — drives data collection in a background thread and
emits batches of samples via Qt signals at a configurable update rate.
"""
from __future__ import annotations
from typing import List
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from .channel import Channel
from .signal_generator import SignalGenerator


class AcquisitionEngine(QThread):
    """Produces simulated samples and pushes them into Channel ring-buffers."""

    data_ready = pyqtSignal()          # emitted after each batch push
    status_changed = pyqtSignal(str)   # "running" | "stopped" | "error: …"

    def __init__(
        self,
        channels: List[Channel],
        generators: List[SignalGenerator],
        sample_rate: float = 1000.0,
        update_interval_ms: int = 50,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.channels = channels
        self.generators = generators
        self.sample_rate = sample_rate
        self.update_interval_ms = update_interval_ms
        self._running = False
        self._mutex = QMutex()
        self._elapsed = 0.0           # seconds of acquired data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start_acquisition(self) -> None:
        self._running = True
        self.start()

    def stop_acquisition(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(2000)
        self.status_changed.emit("stopped")

    def reset(self) -> None:
        self._elapsed = 0.0
        for ch in self.channels:
            ch.clear()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.status_changed.emit("running")
        dt = 1.0 / self.sample_rate
        batch_size = max(1, int(self.sample_rate * self.update_interval_ms / 1000))

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break

            t_start = self._elapsed
            timestamps = t_start + np.arange(batch_size) * dt
            self._elapsed += batch_size * dt

            for ch, gen in zip(self.channels, self.generators):
                if ch.enabled:
                    samples = gen.generate(timestamps)
                    ch.push(samples, timestamps)

            self.data_ready.emit()

            # Pace the loop to the desired update interval
            time.sleep(self.update_interval_ms / 1000.0)

    @property
    def elapsed_time(self) -> float:
        return self._elapsed
