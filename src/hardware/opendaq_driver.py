"""
openDAQ driver — official open-source SDK from DEWE Instruments (DeweSoft makers).

openDAQ provides a hardware abstraction layer for:
  - openDAQ hardware modules (USB/LAN)
  - Simulated device (built-in, no hardware needed for testing)
  - Reference device

Install: pip install opendaq

openDAQ docs: https://docs.opendaq.com
"""
from __future__ import annotations
from typing import Optional
import threading
import time
import numpy as np

from .base import BaseDriver, DriverInfo, SampleCallback


class OpenDAQDriver(BaseDriver):
    INFO = DriverInfo(
        name="openDAQ",
        description="Official openDAQ SDK — supports openDAQ hardware modules + simulated device",
        n_channels=4,
        max_sample_rate=100_000.0,
        requires=["opendaq"],
    )

    def __init__(
        self,
        n_channels: int = 4,
        sample_rate: float = 1000.0,
        connection_string: str = "",   # "" → auto-discover; "daq.opcua://ip" → network
        use_simulated: bool = False,
    ) -> None:
        super().__init__(n_channels, sample_rate)
        self._conn_str = connection_string
        self._use_simulated = use_simulated
        self._instance = None
        self._device = None
        self._readers: list = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @classmethod
    def probe(cls) -> bool:
        try:
            import opendaq  # noqa: F401
            return True
        except ImportError:
            return False

    def start(self, callback: SampleCallback) -> None:
        import opendaq as daq

        self._callback = callback
        self._stop_event.clear()

        # Create instance and connect to device
        self._instance = daq.Instance()

        if self._use_simulated or not self._conn_str:
            # Use built-in simulated device — always available, great for testing
            self._device = self._instance.add_device("daqref://device0")
        else:
            self._device = self._instance.add_device(self._conn_str)

        # Get analog input channels
        channels = [
            sig for sig in self._device.signals_recursive
            if "ai" in sig.name.lower() or "analog" in sig.name.lower()
        ][:self.n_channels]

        # Create packet readers
        self._readers = [daq.StreamReader(ch) for ch in channels]

        # Set sample rate on device if supported
        try:
            self._device.set_property_value("SampleRate", self.sample_rate)
        except Exception:
            pass

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._device:
            self._instance.remove_device(self._device)

    def _read_loop(self) -> None:
        """Poll readers and fire callback with batched samples."""
        dt = 1.0 / self.sample_rate
        batch = max(1, int(self.sample_rate * 0.05))   # 50 ms batches
        t_elapsed = 0.0

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            rows = []
            for reader in self._readers:
                try:
                    samples, _ = reader.read(batch)
                    rows.append(np.asarray(samples, dtype=np.float64))
                except Exception:
                    rows.append(np.zeros(batch))

            if rows:
                # Align to shortest read
                min_len = min(len(r) for r in rows)
                if min_len > 0:
                    data = np.vstack([r[:min_len] for r in rows])  # (ch, samples)
                    timestamps = t_elapsed + np.arange(min_len) * dt
                    t_elapsed += min_len * dt
                    self._callback(data, timestamps)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, 0.05 - elapsed))
