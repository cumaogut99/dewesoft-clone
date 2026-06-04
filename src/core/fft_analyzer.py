"""
FFT analysis helpers — windowed FFT, power spectrum, peak detection.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import windows, find_peaks
from typing import Literal

WindowType = Literal["hann", "hamming", "blackman", "flattop", "rectangular"]


def compute_fft(
    data: np.ndarray,
    sample_rate: float,
    window: WindowType = "hann",
    db_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, magnitudes) for the positive spectrum."""
    n = len(data)
    if n < 2:
        return np.array([]), np.array([])

    w = _make_window(window, n)
    windowed = data * w
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Correct for window amplitude loss
    mag = np.abs(spectrum) * 2.0 / w.sum()
    mag[0] /= 2.0   # DC bin — no mirror

    if db_scale:
        mag = 20.0 * np.log10(np.maximum(mag, 1e-12))

    return freqs, mag


def compute_psd(
    data: np.ndarray,
    sample_rate: float,
    window: WindowType = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, power_spectral_density) using Welch-style single-block PSD."""
    n = len(data)
    if n < 2:
        return np.array([]), np.array([])

    w = _make_window(window, n)
    windowed = data * w
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    psd = (np.abs(spectrum) ** 2) / (sample_rate * (w ** 2).sum())
    psd[1:-1] *= 2.0  # one-sided

    return freqs, psd


def find_spectral_peaks(
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    n_peaks: int = 5,
    min_height_db: float = -60.0,
) -> list[tuple[float, float]]:
    """Return list of (frequency, magnitude_db) for the top N peaks."""
    peaks, _ = find_peaks(magnitudes, height=min_height_db, distance=3)
    if len(peaks) == 0:
        return []
    order = np.argsort(magnitudes[peaks])[::-1]
    top = peaks[order[:n_peaks]]
    return [(float(freqs[i]), float(magnitudes[i])) for i in top]


def _make_window(window: WindowType, n: int) -> np.ndarray:
    match window:
        case "hann":
            return windows.hann(n)
        case "hamming":
            return windows.hamming(n)
        case "blackman":
            return windows.blackman(n)
        case "flattop":
            return windows.flattop(n)
        case _:
            return np.ones(n)
