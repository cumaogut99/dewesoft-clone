"""
Math engine — computes derived channels from source channels after each batch.
Math channels live alongside hardware channels; the UI reads them identically.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

from .channel import Channel
from .math_channel import (
    running_rms, cumulative_integral, derivative,
    lowpass_filter, highpass_filter, bandpass_filter,
)


@dataclass
class MathChannelDef:
    source_id: int
    operation: str          # key into OPERATIONS
    output_channel: Channel
    params: dict = field(default_factory=dict)


OPERATIONS = {
    "RMS":        lambda d, ts, sr, p: running_rms(d, p.get("window", 100)),
    "Integral":   lambda d, ts, sr, p: cumulative_integral(d, 1.0 / sr),
    "Derivative": lambda d, ts, sr, p: derivative(d, 1.0 / sr),
    "Low-pass":   lambda d, ts, sr, p: lowpass_filter(d, p.get("cutoff", 100.0), sr),
    "High-pass":  lambda d, ts, sr, p: highpass_filter(d, p.get("cutoff", 10.0), sr),
    "Band-pass":  lambda d, ts, sr, p: bandpass_filter(
        d, p.get("low", 10.0), p.get("high", 500.0), sr
    ),
    "Abs":        lambda d, ts, sr, p: np.abs(d),
    "Square":     lambda d, ts, sr, p: d ** 2,
    "Invert":     lambda d, ts, sr, p: -d,
}


class MathEngine:
    def __init__(self) -> None:
        self._defs: List[MathChannelDef] = []

    @property
    def definitions(self) -> List[MathChannelDef]:
        return list(self._defs)

    def add(self, defn: MathChannelDef) -> None:
        self._defs.append(defn)

    def remove(self, output_channel_id: int) -> None:
        self._defs = [d for d in self._defs if d.output_channel.id != output_channel_id]

    def update(self, source_channels: List[Channel]) -> None:
        """Call after each acquisition batch to refresh all math channel buffers."""
        src_map = {ch.id: ch for ch in source_channels}
        for defn in self._defs:
            src = src_map.get(defn.source_id)
            if src is None:
                continue
            ts, data = src.get_data()
            if len(data) < 4:
                continue
            fn = OPERATIONS.get(defn.operation)
            if fn is None:
                continue
            try:
                result = fn(data, ts, src.sample_rate, defn.params)
                # Trim to same length (some ops change length)
                n = min(len(result), len(data))
                defn.output_channel.push(result[-n:], ts[-n:])
            except Exception:
                pass

    @property
    def output_channels(self) -> List[Channel]:
        return [d.output_channel for d in self._defs]
