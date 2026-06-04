"""
Playback engine — replays saved HDF5/CSV recordings into Channel ring buffers.
Implements the same interface as AcquisitionEngine so UI works unchanged.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from .channel import Channel
from ..io.data_writer import load_hdf5, load_csv


class PlaybackEngine(QThread):
    data_ready = pyqtSignal()
    status_changed = pyqtSignal(str)
    progress = pyqtSignal(float)      # 0.0 – 1.0

    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._records: list[dict] = []      # loaded channel records
        self._speed = 1.0
        self._running = False
        self._paused = False
        self._pos = 0                        # current sample index
        self._mutex = QMutex()
        self._elapsed = 0.0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, path: str | Path) -> int:
        """Load recording. Returns number of channels found."""
        path = Path(path)
        if path.suffix in (".h5", ".hdf5"):
            self._records = load_hdf5(path)
        elif path.suffix == ".csv":
            self._records = load_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        self._pos = 0
        self._elapsed = 0.0
        return len(self._records)

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------
    def start_playback(self) -> None:
        if not self._records:
            return
        with QMutexLocker(self._mutex):
            self._running = True
            self._paused = False
        self.start()
        self.status_changed.emit("playing")

    def stop_playback(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(2000)
        self.status_changed.emit("stopped")

    def pause(self) -> None:
        with QMutexLocker(self._mutex):
            self._paused = not self._paused
        state = "paused" if self._paused else "playing"
        self.status_changed.emit(state)

    def seek(self, fraction: float) -> None:
        if not self._records:
            return
        max_len = max(len(r["data"]) for r in self._records)
        with QMutexLocker(self._mutex):
            self._pos = int(np.clip(fraction, 0.0, 1.0) * max_len)

    def set_speed(self, speed: float) -> None:
        with QMutexLocker(self._mutex):
            self._speed = max(0.1, min(speed, 32.0))

    @property
    def elapsed_time(self) -> float:
        return self._elapsed

    @property
    def duration(self) -> float:
        if not self._records:
            return 0.0
        return float(max(
            r["timestamps"][-1] for r in self._records if len(r["timestamps"]) > 0
        ))

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------
    def run(self) -> None:
        if not self._records:
            return

        sr = self._records[0].get("sample_rate", 1000.0)
        batch = max(1, int(sr * 0.05))   # 50 ms batches at real time
        max_len = max(len(r["data"]) for r in self._records)

        for ch in self.channels:
            ch.clear()

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
                if self._paused:
                    pass
                speed = self._speed
                pos = self._pos

            if self._paused:
                time.sleep(0.05)
                continue

            if pos >= max_len:
                self.status_changed.emit("finished")
                break

            end = min(pos + batch, max_len)

            for i, rec in enumerate(self._records):
                if i >= len(self.channels):
                    break
                ch = self.channels[i]
                data_slice = rec["data"][pos:end]
                ts_slice = rec["timestamps"][pos:min(end, len(rec["timestamps"]))]
                if len(data_slice) > 0 and len(ts_slice) > 0:
                    n = min(len(data_slice), len(ts_slice))
                    ch.push(data_slice[:n], ts_slice[:n])
                    self._elapsed = float(ts_slice[-1])

            self._pos = end
            self.progress.emit(end / max_len)
            self.data_ready.emit()

            sleep_s = (batch / sr) / speed
            time.sleep(sleep_s)
