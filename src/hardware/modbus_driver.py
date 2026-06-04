"""
Modbus RTU / TCP driver.

Reads Holding Registers at a fixed poll rate and converts raw register values
to engineering units using a scale factor.

Install: pip install pymodbus
"""
from __future__ import annotations
import threading
import time
import numpy as np

from .base import BaseDriver, DriverInfo, SampleCallback


class ModbusDriver(BaseDriver):
    INFO = DriverInfo(
        name="Modbus RTU / TCP",
        description="Poll any Modbus-compatible PLC, sensor, or transmitter",
        n_channels=8,
        max_sample_rate=100.0,   # Modbus polling is slow by nature
        requires=["pymodbus"],
    )

    def __init__(
        self,
        n_channels: int = 4,
        sample_rate: float = 10.0,
        mode: str = "tcp",          # "tcp" or "rtu"
        host: str = "192.168.1.100",
        port: int = 502,
        serial_port: str = "/dev/ttyUSB0",
        baud_rate: int = 9600,
        slave_id: int = 1,
        start_register: int = 0,   # first holding register address
        scale: float = 0.1,        # raw int → engineering unit
        word_order: str = ">",     # big-endian (Modbus default)
    ) -> None:
        super().__init__(n_channels, sample_rate)
        self.mode = mode
        self.host = host
        self.port = port
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.slave_id = slave_id
        self.start_register = start_register
        self.scale = scale
        self._client = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @classmethod
    def probe(cls) -> bool:
        try:
            import pymodbus  # noqa: F401
            return True
        except ImportError:
            return False

    def start(self, callback: SampleCallback) -> None:
        from pymodbus.client import ModbusTcpClient, ModbusSerialClient

        self._callback = callback
        self._stop_event.clear()

        if self.mode == "tcp":
            self._client = ModbusTcpClient(self.host, port=self.port)
        else:
            self._client = ModbusSerialClient(
                "rtu", port=self.serial_port, baudrate=self.baud_rate
            )

        if not self._client.connect():
            raise ConnectionError(f"Modbus: cannot connect to {self.host}:{self.port}")

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._client:
            self._client.close()

    def _poll_loop(self) -> None:
        dt = 1.0 / self.sample_rate
        t_elapsed = 0.0

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            try:
                result = self._client.read_holding_registers(
                    self.start_register, self.n_channels, slave=self.slave_id
                )
                if not result.isError():
                    values = np.array(result.registers[:self.n_channels], dtype=np.float64)
                    values *= self.scale
                    data = values.reshape(self.n_channels, 1)
                    timestamps = np.array([t_elapsed])
                    t_elapsed += dt
                    self._callback(data, timestamps)
            except Exception:
                pass

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))
