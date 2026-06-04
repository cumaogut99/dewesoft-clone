"""
Serial / Arduino driver.

Arduino side example (outputs CSV at fixed intervals):
  void loop() {
    Serial.print(analogRead(A0) / 1023.0 * 5.0);  // CH1 in volts
    Serial.print(",");
    Serial.println(analogRead(A1) / 1023.0 * 5.0); // CH2
    delay(1);  // 1ms → ~1kHz
  }

Protocol: each line is "val0,val1,...,valN\n"  (plain CSV, no header)

Install: pip install pyserial
"""
from __future__ import annotations
import threading
import time
import numpy as np

from .base import BaseDriver, DriverInfo, SampleCallback


class SerialDriver(BaseDriver):
    INFO = DriverInfo(
        name="Serial / Arduino",
        description="Any device sending CSV-per-line over serial (Arduino, STM32, etc.)",
        n_channels=8,
        max_sample_rate=10_000.0,
        requires=["pyserial"],
    )

    def __init__(
        self,
        n_channels: int = 2,
        sample_rate: float = 1000.0,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        timeout: float = 1.0,
    ) -> None:
        super().__init__(n_channels, sample_rate)
        self._port = port
        self._baud = baud_rate
        self._timeout = timeout
        self._ser = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @classmethod
    def probe(cls) -> bool:
        try:
            import serial  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def list_ports(cls) -> list[str]:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]

    def start(self, callback: SampleCallback) -> None:
        import serial

        self._callback = callback
        self._stop_event.clear()
        self._ser = serial.Serial(self._port, self._baud, timeout=self._timeout)
        self._ser.reset_input_buffer()

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _read_loop(self) -> None:
        dt = 1.0 / self.sample_rate
        t_elapsed = 0.0
        batch: list[list[float]] = []
        batch_target = max(1, int(self.sample_rate * 0.05))  # 50 ms batches

        while not self._stop_event.is_set():
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                break

            if not line:
                continue

            try:
                vals = [float(v) for v in line.split(",")]
            except ValueError:
                continue

            # Pad or truncate to n_channels
            while len(vals) < self.n_channels:
                vals.append(0.0)
            batch.append(vals[:self.n_channels])

            if len(batch) >= batch_target:
                arr = np.array(batch, dtype=np.float64).T  # (n_ch, n_samples)
                n = arr.shape[1]
                timestamps = t_elapsed + np.arange(n) * dt
                t_elapsed += n * dt
                self._callback(arr, timestamps)
                batch.clear()
