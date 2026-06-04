"""
Software signal generator driver — no hardware required.
Runs in its own thread; callback rate matches update_interval_ms.
"""
from __future__ import annotations
import threading
import time
import numpy as np
from .base import BaseDriver, DriverInfo, SampleCallback
from ..core.signal_generator import SignalGenerator, WaveformType


class SimulatedDriver(BaseDriver):
    INFO = DriverInfo(
        name="Simulated",
        description="Software signal generator — no hardware needed",
        n_channels=8,
        max_sample_rate=100_000.0,
        requires=[],
    )

    def __init__(
        self,
        n_channels: int = 4,
        sample_rate: float = 1000.0,
        generators: list[SignalGenerator] | None = None,
        update_interval_ms: int = 50,
    ) -> None:
        super().__init__(n_channels, sample_rate)
        self._update_ms = update_interval_ms
        self._generators = generators or [
            SignalGenerator(
                waveform=[WaveformType.SINE, WaveformType.SQUARE,
                          WaveformType.CHIRP, WaveformType.TRIANGLE][i % 4],
                frequency=10.0 * (i + 1),
                amplitude=1.0,
            )
            for i in range(n_channels)
        ]
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._elapsed = 0.0

    def start(self, callback: SampleCallback) -> None:
        self._callback = callback
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        dt = 1.0 / self.sample_rate
        batch = max(1, int(self.sample_rate * self._update_ms / 1000))

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            timestamps = self._elapsed + np.arange(batch) * dt
            self._elapsed += batch * dt

            data = np.empty((self.n_channels, batch), dtype=np.float64)
            for i, gen in enumerate(self._generators[:self.n_channels]):
                data[i] = gen.generate(timestamps)

            self._callback(data, timestamps)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, self._update_ms / 1000.0 - elapsed))
