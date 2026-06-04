"""
Trigger detection — edge/level finder with hysteresis for oscilloscope sync.

Design: stateful "arm then fire" logic.
  Rising edge:  signal must fall below (level − hysteresis) to arm,
                then rise to ≥ level to fire.
  Falling edge: signal must rise above (level + hysteresis) to arm,
                then fall to ≤ level to fire.

Numpy is used for the initial crossing search; the hysteresis arm check
scans from the data start to each candidate crossing — fast for typical
trigger window sizes (a few thousand samples).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal

Edge = Literal["rising", "falling", "either"]


@dataclass
class TriggerConfig:
    enabled: bool = False
    edge: Edge = "rising"
    level: float = 0.0
    hysteresis: float = 0.02   # signal must pass level ± hysteresis before re-arming
    pre_samples: int = 50       # samples kept before the trigger point
    channel_id: int = 1


def find_trigger(
    data: np.ndarray,
    level: float,
    edge: Edge = "rising",
    hysteresis: float = 0.02,
) -> int:
    """
    Return index of the first valid trigger crossing, or -1 if none found.

    A crossing is valid only if the signal was previously armed
    (passed the hysteresis threshold on the other side of level).
    """
    if len(data) < 2:
        return -1

    if edge == "either":
        r = find_trigger(data, level, "rising",  hysteresis)
        f = find_trigger(data, level, "falling", hysteresis)
        if r == -1:
            return f
        if f == -1:
            return r
        return min(r, f)

    if edge == "rising":
        arm_level = level - hysteresis
        # Find all samples where signal crosses from below-level to at/above-level
        rel = data - level
        candidates = np.where((rel[:-1] < 0) & (rel[1:] >= 0))[0]
        for ci in candidates:
            # Armed if signal was ever at/below arm_level before this crossing
            if np.any(data[: ci + 1] <= arm_level):
                return int(ci) + 1
    else:  # falling
        arm_level = level + hysteresis
        rel = data - level
        candidates = np.where((rel[:-1] > 0) & (rel[1:] <= 0))[0]
        for ci in candidates:
            if np.any(data[: ci + 1] >= arm_level):
                return int(ci) + 1

    return -1


def align_to_trigger(
    timestamps: np.ndarray,
    data: np.ndarray,
    cfg: TriggerConfig,
    n_display: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice (timestamps, data) to n_display samples starting pre_samples before
    the first trigger crossing. Returns the last n_display samples if not found.
    """
    if not cfg.enabled or len(data) < 2:
        return timestamps[-n_display:], data[-n_display:]

    idx = find_trigger(data, cfg.level, cfg.edge, cfg.hysteresis)
    if idx < 0:
        return timestamps[-n_display:], data[-n_display:]

    start = max(0, idx - cfg.pre_samples)
    end = min(len(data), start + n_display)
    return timestamps[start:end], data[start:end]
