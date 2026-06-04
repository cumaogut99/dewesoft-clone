"""
Display decimators — reduce N samples to M points without losing visual information.

MinMaxDecimator (default, O(N)):
  For each display pixel column, keep the min and max of the bucket.
  Interleaves [min0, max0, min1, max1, ...] — preserves peaks and troughs
  exactly as an oscilloscope would.

LTTBDecimator (Largest Triangle Three Buckets, O(N)):
  Keeps the point in each bucket that forms the largest triangle with its
  neighbours. Better for smooth signals; slightly more expensive.

Rule of thumb:
  <  2× n_out  → return original (no decimation needed)
  oscilloscope → MinMax (preserves spikes)
  recorder     → LTTB   (smoother appearance)
"""
from __future__ import annotations
import numpy as np


def minmax_decimate(
    timestamps: np.ndarray,
    data: np.ndarray,
    n_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (ts_out, data_out) with at most 2*n_out points.
    Works entirely with numpy — no Python loops.
    """
    n = len(data)
    if n <= 2 * n_out:
        return timestamps, data

    # Reshape into buckets; trim tail to fit evenly
    bucket_size = n // n_out
    usable = bucket_size * n_out
    d = data[:usable].reshape(n_out, bucket_size)
    t = timestamps[:usable].reshape(n_out, bucket_size)

    min_idx_local = d.argmin(axis=1)           # shape (n_out,)
    max_idx_local = d.argmax(axis=1)           # shape (n_out,)
    offsets = np.arange(n_out) * bucket_size

    min_idx = offsets + min_idx_local
    max_idx = offsets + max_idx_local

    # Interleave min/max in chronological order per bucket
    pick_first = np.where(min_idx <= max_idx, min_idx, max_idx)
    pick_second = np.where(min_idx <= max_idx, max_idx, min_idx)

    idx_out = np.empty(2 * n_out, dtype=np.intp)
    idx_out[0::2] = pick_first
    idx_out[1::2] = pick_second

    return timestamps[idx_out], data[idx_out]


def lttb_decimate(
    timestamps: np.ndarray,
    data: np.ndarray,
    n_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Largest Triangle Three Buckets — O(N) vectorised implementation.
    Returns exactly n_out points.
    """
    n = len(data)
    if n <= n_out:
        return timestamps, data

    out_idx = np.empty(n_out, dtype=np.intp)
    out_idx[0] = 0
    out_idx[-1] = n - 1

    bucket_size = (n - 2) / (n_out - 2)

    a = 0
    for i in range(1, n_out - 1):
        b_start = int((i - 1) * bucket_size) + 1
        b_end = int(i * bucket_size) + 1
        c_start = int(i * bucket_size) + 1
        c_end = min(int((i + 1) * bucket_size) + 1, n)

        # Average of next bucket (point C)
        avg_x = np.mean(np.arange(c_start, c_end, dtype=np.float64))
        avg_y = np.mean(data[c_start:c_end])

        # Candidates in current bucket
        xs = np.arange(b_start, min(b_end, n), dtype=np.float64)
        ys = data[b_start:min(b_end, n)]

        # Triangle area with points A and C
        area = np.abs(
            (timestamps[a] - avg_x) * (ys - data[a]) -
            (timestamps[a] - xs) * (avg_y - data[a])
        ) * 0.5

        best = b_start + int(np.argmax(area))
        out_idx[i] = best
        a = best

    return timestamps[out_idx], data[out_idx]


def auto_decimate(
    timestamps: np.ndarray,
    data: np.ndarray,
    max_points: int = 4000,
    mode: str = "minmax",
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper — chooses decimation only when necessary."""
    if len(data) <= max_points:
        return timestamps, data
    if mode == "lttb":
        return lttb_decimate(timestamps, data, max_points)
    return minmax_decimate(timestamps, data, max_points // 2)
