"""
FFT analysis helpers — windowed FFT, power spectrum, peak detection,
and distortion / noise metrics (THD, THD+N, SNR, SINAD, SFDR).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.signal import windows, find_peaks
from typing import Literal

WindowType = Literal["hann", "hamming", "blackman", "flattop", "rectangular"]


# -----------------------------------------------------------------------
# Spectrum computation
# -----------------------------------------------------------------------

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
    """Return (frequencies, power_spectral_density) — single-block Welch."""
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


# -----------------------------------------------------------------------
# THD / distortion metrics
# -----------------------------------------------------------------------

@dataclass
class THDResult:
    """Results of a full harmonic distortion analysis."""
    fundamental_freq: float = 0.0
    fundamental_amplitude: float = 0.0   # linear (V peak)

    harmonic_freqs: list[float] = field(default_factory=list)
    harmonic_amplitudes: list[float] = field(default_factory=list)  # linear

    # Distortion
    thd_percent: float = 0.0      # THD  = √(ΣVn²) / V1 × 100
    thd_db: float = 0.0           # THD  in dB  (20 log10(THD/100))

    # Noise + distortion combined
    thd_n_percent: float = 0.0    # THD+N = √(total² − V1²) / V1 × 100
    thd_n_db: float = 0.0

    # Noise metrics
    snr_db: float = 0.0           # SNR  = 10 log10(P_signal / P_noise)
    sinad_db: float = 0.0         # SINAD = 10 log10(V1² / (total² − V1²))

    # Dynamic range
    sfdr_dbc: float = 0.0         # SFDR = fundamental − max spurious  (dBc)

    n_harmonics: int = 0          # number of harmonics actually found


def compute_thd(
    data: np.ndarray,
    sample_rate: float,
    window: WindowType = "flattop",
    fundamental_freq: float | None = None,
    n_harmonics: int = 9,
    search_window_hz: float | None = None,
) -> THDResult:
    """
    Full harmonic analysis: THD, THD+N, SNR, SINAD, SFDR.

    Parameters
    ----------
    data              : time-domain signal (1-D float array)
    sample_rate       : samples per second
    window            : FFT window — 'flattop' gives best amplitude accuracy for THD
    fundamental_freq  : override auto-detection
    n_harmonics       : how many harmonics to search for (2nd … (n+1)th)
    search_window_hz  : search radius around each harmonic (default: 2 × bin spacing)
    """
    result = THDResult()
    n = len(data)
    if n < 64:
        return result

    w = _make_window(window, n)
    windowed = data * w
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Linear amplitude spectrum (peak volts), window-corrected
    mag_lin = np.abs(spectrum) * 2.0 / w.sum()
    mag_lin[0] /= 2.0   # DC bin

    df = freqs[1] - freqs[0]                           # bin spacing
    sw = search_window_hz if search_window_hz else 2.5 * df

    # ---- Fundamental ----
    if fundamental_freq is not None:
        f0 = fundamental_freq
        f0_amp = _peak_near(freqs, mag_lin, f0, sw)
    else:
        # Skip DC and very low freq (<= 1 Hz) when auto-detecting
        min_bin = max(1, int(1.0 / df))
        f0_idx = min_bin + int(np.argmax(mag_lin[min_bin:]))
        f0 = float(freqs[f0_idx])
        f0_amp = float(mag_lin[f0_idx])

    if f0_amp < 1e-12:
        return result

    result.fundamental_freq = f0
    result.fundamental_amplitude = f0_amp

    # ---- Harmonics ----
    h_amps: list[float] = []
    h_freqs: list[float] = []
    nyquist = sample_rate / 2.0

    for k in range(2, n_harmonics + 2):
        hf = k * f0
        if hf > nyquist:
            break
        ha = _peak_near(freqs, mag_lin, hf, sw * k)   # wider window for higher harmonics
        h_freqs.append(hf)
        h_amps.append(ha)

    result.harmonic_freqs = h_freqs
    result.harmonic_amplitudes = h_amps
    # Relative threshold: harmonic must be > 0.001% of fundamental to count
    result.n_harmonics = sum(1 for a in h_amps if a > f0_amp * 1e-5)

    # ---- THD ----
    harmonic_rms = float(np.sqrt(sum(a ** 2 for a in h_amps)))
    thd = harmonic_rms / f0_amp
    result.thd_percent = thd * 100.0
    result.thd_db = 20.0 * np.log10(max(thd, 1e-12))

    # ---- Total power via Parseval (time-domain; immune to window leakage) ----
    total_power = float(np.mean(data ** 2))
    f0_power = (f0_amp ** 2) / 2.0
    harm_power = sum(a ** 2 / 2.0 for a in h_amps)
    sig_power = f0_power + harm_power

    # ---- THD+N ----
    noise_dist_power = max(total_power - f0_power, 1e-24)
    thd_n = math.sqrt(noise_dist_power) / (f0_amp / math.sqrt(2.0))
    result.thd_n_percent = thd_n * 100.0
    result.thd_n_db = 20.0 * math.log10(max(thd_n, 1e-12))

    # ---- SNR ----
    noise_power = max(total_power - sig_power, 1e-24)
    result.snr_db = float(10.0 * math.log10(sig_power / noise_power))

    # ---- SINAD ----
    result.sinad_db = float(10.0 * math.log10(f0_power / noise_dist_power))

    # ---- SFDR ----
    # Use Blackman-Harris window (92 dB sidelobes) to prevent window artefacts
    # being mistaken for spurs.  Exclude a wider band around the fundamental.
    bh_win = windows.blackmanharris(n)
    bh_mag = np.abs(np.fft.rfft(data * bh_win)) * 2.0 / bh_win.sum()
    bh_mag[0] = 0.0  # kill DC
    bh_f0_amp = float(bh_mag[np.argmin(np.abs(freqs - f0))])
    excl_hz = max(sw * 4.0, 3.0 * df)
    fund_mask = np.abs(freqs - f0) > excl_hz
    bh_spur = bh_mag.copy()
    bh_spur[~fund_mask] = 0.0
    if bh_spur.max() > 1e-12 and bh_f0_amp > 1e-12:
        result.sfdr_dbc = float(20.0 * np.log10(bh_f0_amp / bh_spur.max()))
    else:
        result.sfdr_dbc = 120.0

    return result


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _peak_near(
    freqs: np.ndarray,
    mag_lin: np.ndarray,
    target_hz: float,
    window_hz: float,
) -> float:
    """Return the maximum linear amplitude within ±window_hz of target_hz."""
    mask = np.abs(freqs - target_hz) <= window_hz
    if not np.any(mask):
        return 0.0
    return float(mag_lin[mask].max())


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
