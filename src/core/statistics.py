"""
Statistics computation — channel-level descriptive statistics, optionally
restricted to a time-range selection [t_start, t_end].
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .channel import Channel


@dataclass
class ChannelStats:
    name: str
    unit: str
    n_samples: int
    mean: float
    std: float
    rms: float
    minimum: float
    maximum: float
    peak_to_peak: float
    crest_factor: float   # peak / RMS
    duration: float       # seconds covered


_EMPTY = ChannelStats(
    name="", unit="", n_samples=0,
    mean=0.0, std=0.0, rms=0.0,
    minimum=0.0, maximum=0.0,
    peak_to_peak=0.0, crest_factor=0.0, duration=0.0,
)


def compute_stats(
    channel: Channel,
    t_start: float | None = None,
    t_end: float | None = None,
) -> ChannelStats:
    """
    Compute descriptive statistics for *channel*, optionally restricted to
    the time window [t_start, t_end].  Returns a zeroed ChannelStats if the
    channel has no data in the requested range.
    """
    ts, data = channel.get_data()
    if len(data) == 0:
        return ChannelStats(name=channel.name, unit=channel.unit,
                            **{f: 0.0 for f in _EMPTY.__dataclass_fields__
                               if f not in ("name", "unit", "n_samples")},
                            n_samples=0)

    if t_start is not None or t_end is not None:
        lo = t_start if t_start is not None else ts[0]
        hi = t_end   if t_end   is not None else ts[-1]
        mask = (ts >= lo) & (ts <= hi)
        data = data[mask]
        ts   = ts[mask]

    if len(data) == 0:
        return ChannelStats(name=channel.name, unit=channel.unit,
                            n_samples=0, mean=0.0, std=0.0, rms=0.0,
                            minimum=0.0, maximum=0.0, peak_to_peak=0.0,
                            crest_factor=0.0, duration=0.0)

    rms_val = float(np.sqrt(np.mean(data ** 2)))
    peak    = float(np.max(np.abs(data)))

    return ChannelStats(
        name=channel.name,
        unit=channel.unit,
        n_samples=len(data),
        mean=float(np.mean(data)),
        std=float(np.std(data)),
        rms=rms_val,
        minimum=float(data.min()),
        maximum=float(data.max()),
        peak_to_peak=float(data.max() - data.min()),
        crest_factor=peak / rms_val if rms_val > 0 else 0.0,
        duration=float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
    )


def compute_stats_all(
    channels: list[Channel],
    t_start: float | None = None,
    t_end: float | None = None,
) -> list[ChannelStats]:
    return [compute_stats(ch, t_start, t_end) for ch in channels]
