"""
Data channel — represents a single measured or calculated signal.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class ChannelType(Enum):
    ANALOG = "analog"
    DIGITAL = "digital"
    MATH = "math"
    COUNTER = "counter"


class ChannelUnit(Enum):
    VOLT = "V"
    AMPERE = "A"
    CELSIUS = "°C"
    PASCAL = "Pa"
    METER_PER_SEC2 = "m/s²"
    HZ = "Hz"
    NONE = ""


@dataclass
class Channel:
    id: int
    name: str
    channel_type: ChannelType = ChannelType.ANALOG
    unit: str = "V"
    sample_rate: float = 1000.0   # Hz
    color: str = "#00ff88"
    enabled: bool = True
    scale: float = 1.0
    offset: float = 0.0
    # Ring buffer for live display
    _buffer_size: int = field(default=10_000, repr=False)
    _data: np.ndarray = field(init=False, repr=False)
    _timestamps: np.ndarray = field(init=False, repr=False)
    _write_idx: int = field(default=0, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._data = np.zeros(self._buffer_size, dtype=np.float64)
        self._timestamps = np.zeros(self._buffer_size, dtype=np.float64)

    def push(self, values: np.ndarray, timestamps: np.ndarray) -> None:
        n = len(values)
        idx = np.arange(self._write_idx, self._write_idx + n) % self._buffer_size
        self._data[idx] = values * self.scale + self.offset
        self._timestamps[idx] = timestamps
        self._write_idx = (self._write_idx + n) % self._buffer_size
        self._count = min(self._count + n, self._buffer_size)

    def get_data(self, n_samples: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        if self._count == 0:
            return np.array([]), np.array([])
        count = min(n_samples or self._count, self._count)
        start = (self._write_idx - count) % self._buffer_size
        idx = np.arange(start, start + count) % self._buffer_size
        return self._timestamps[idx], self._data[idx]

    def clear(self) -> None:
        self._data[:] = 0
        self._timestamps[:] = 0
        self._write_idx = 0
        self._count = 0

    @property
    def latest_value(self) -> float:
        if self._count == 0:
            return 0.0
        return float(self._data[(self._write_idx - 1) % self._buffer_size])
