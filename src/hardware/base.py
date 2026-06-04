"""
Abstract hardware driver interface.

Every driver exposes:
  - start(callback)  → acquisition begins; callback(data, timestamps) fired per batch
  - stop()           → acquisition ends, resources released
  - probe()          → class method, returns True if hardware is reachable

callback signature:
  data       : np.ndarray  shape (n_channels, n_samples)  float64
  timestamps : np.ndarray  shape (n_samples,)              float64  seconds
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, ClassVar
import numpy as np

SampleCallback = Callable[[np.ndarray, np.ndarray], None]


@dataclass
class DriverInfo:
    name: str
    description: str
    n_channels: int
    max_sample_rate: float
    requires: list[str] = field(default_factory=list)   # pip packages needed


class BaseDriver(ABC):
    INFO: ClassVar[DriverInfo]          # every subclass must set this

    def __init__(self, n_channels: int, sample_rate: float, **kwargs) -> None:
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self._callback: SampleCallback | None = None

    @abstractmethod
    def start(self, callback: SampleCallback) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @classmethod
    def probe(cls) -> bool:
        """Return True if driver can be instantiated (hardware present / libs installed)."""
        return True

    @classmethod
    def available_channels(cls) -> list[str]:
        """List channel names detectable from hardware (if supported)."""
        return [f"CH{i+1}" for i in range(cls.INFO.n_channels)]
