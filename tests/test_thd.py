"""
Tests for THD, SNR, SINAD, SFDR calculations.
All tests use analytically known signals so expected values are exact.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from src.core.fft_analyzer import compute_thd, compute_fft, THDResult


def _make_signal(
    sr: float,
    n: int,
    components: list[tuple[float, float]],   # (frequency_hz, amplitude)
) -> np.ndarray:
    """Sum of sinusoids."""
    t = np.arange(n) / sr
    out = np.zeros(n)
    for freq, amp in components:
        out += amp * np.sin(2 * np.pi * freq * t)
    return out


# ---- Fundamental detection ------------------------------------------

def test_fundamental_auto_detect():
    sr, n = 10_000.0, 8192
    sig = _make_signal(sr, n, [(1_000.0, 1.0)])
    r = compute_thd(sig, sr)
    assert abs(r.fundamental_freq - 1_000.0) < 5.0   # within 5 Hz


def test_fundamental_override():
    sr, n = 10_000.0, 8192
    # Two sinusoids; lower amplitude is the "fundamental" by definition
    sig = _make_signal(sr, n, [(500.0, 0.5), (1_000.0, 1.0)])
    r = compute_thd(sig, sr, fundamental_freq=500.0)
    assert abs(r.fundamental_freq - 500.0) < 5.0


# ---- THD accuracy ---------------------------------------------------

def test_thd_pure_sine_is_near_zero():
    sr, n = 48_000.0, 16_384
    sig = _make_signal(sr, n, [(1_000.0, 1.0)])
    r = compute_thd(sig, sr, window="flattop")
    # Pure sine: THD should be very small (< 0.5 % due to spectral leakage)
    assert r.thd_percent < 0.5


def test_thd_known_harmonic_content():
    """
    Signal: 1 kHz (A=1.0) + 2 kHz (A=0.1) + 3 kHz (A=0.05).
    THD = sqrt(0.1² + 0.05²) / 1.0 ≈ 11.18 %.
    """
    sr, n = 48_000.0, 32_768
    sig = _make_signal(sr, n, [(1_000.0, 1.0), (2_000.0, 0.1), (3_000.0, 0.05)])
    expected_thd = math.sqrt(0.1**2 + 0.05**2) / 1.0 * 100.0  # ≈ 11.18 %
    r = compute_thd(sig, sr, window="flattop", n_harmonics=5)
    # Allow ±1 % absolute tolerance (window leakage + bin interpolation)
    assert abs(r.thd_percent - expected_thd) < 1.0, (
        f"THD={r.thd_percent:.2f}% vs expected={expected_thd:.2f}%"
    )


def test_thd_harmonics_found():
    sr, n = 48_000.0, 16_384
    sig = _make_signal(sr, n, [
        (1_000.0, 1.0),
        (2_000.0, 0.1),
        (3_000.0, 0.05),
        (4_000.0, 0.02),
    ])
    r = compute_thd(sig, sr, window="flattop", n_harmonics=5)
    assert r.n_harmonics >= 3


def test_thd_increases_with_distortion():
    sr, n = 48_000.0, 16_384
    clean = _make_signal(sr, n, [(1_000.0, 1.0)])
    dirty = _make_signal(sr, n, [(1_000.0, 1.0), (2_000.0, 0.5)])
    r_clean = compute_thd(clean, sr)
    r_dirty = compute_thd(dirty, sr)
    assert r_dirty.thd_percent > r_clean.thd_percent


# ---- THD dB conversion ----------------------------------------------

def test_thd_db_consistent_with_percent():
    sr, n = 48_000.0, 16_384
    sig = _make_signal(sr, n, [(1_000.0, 1.0), (2_000.0, 0.1)])
    r = compute_thd(sig, sr)
    expected_db = 20.0 * math.log10(r.thd_percent / 100.0)
    assert abs(r.thd_db - expected_db) < 0.01


# ---- SNR / SINAD ----------------------------------------------------

def test_snr_pure_sine_is_high():
    sr, n = 48_000.0, 16_384
    # Very clean sine — SNR should be > 60 dB
    sig = _make_signal(sr, n, [(1_000.0, 1.0)])
    r = compute_thd(sig, sr, window="flattop")
    assert r.snr_db > 60.0


def test_sinad_pure_sine_is_high():
    sr, n = 48_000.0, 16_384
    sig = _make_signal(sr, n, [(1_000.0, 1.0)])
    r = compute_thd(sig, sr, window="flattop")
    assert r.sinad_db > 60.0


def test_sinad_less_than_snr_with_harmonics():
    """SINAD ≤ SNR (distortion degrades SINAD but not necessarily SNR the same way)."""
    sr, n = 48_000.0, 16_384
    sig = _make_signal(sr, n, [(1_000.0, 1.0), (2_000.0, 0.2), (3_000.0, 0.1)])
    r = compute_thd(sig, sr)
    # SINAD includes distortion; SNR only noise — SINAD should be lower
    assert r.sinad_db <= r.snr_db + 1.0   # SINAD ≤ SNR (+1 dB tolerance)


# ---- SFDR -----------------------------------------------------------

def test_sfdr_pure_sine_is_high():
    sr, n = 48_000.0, 16_384
    sig = _make_signal(sr, n, [(1_000.0, 1.0)])
    r = compute_thd(sig, sr, window="flattop")
    assert r.sfdr_dbc > 60.0


def test_sfdr_decreases_with_spur():
    sr, n = 48_000.0, 16_384
    clean = _make_signal(sr, n, [(1_000.0, 1.0)])
    with_spur = _make_signal(sr, n, [(1_000.0, 1.0), (3_700.0, 0.1)])
    r_clean = compute_thd(clean, sr)
    r_spur  = compute_thd(with_spur, sr)
    assert r_spur.sfdr_dbc < r_clean.sfdr_dbc


# ---- Edge cases -----------------------------------------------------

def test_thd_empty_signal():
    r = compute_thd(np.zeros(8192), 48_000.0)
    assert r.fundamental_amplitude == 0.0
    assert r.thd_percent == 0.0


def test_thd_too_short():
    r = compute_thd(np.ones(10), 1_000.0)
    assert r.thd_percent == 0.0


def test_thd_harmonics_beyond_nyquist_skipped():
    """High fundamental — only 1–2 harmonics fit under Nyquist."""
    sr, n = 10_000.0, 4_096
    # Fundamental at 4 kHz → 2nd harmonic at 8 kHz (close to Nyquist=5 kHz)
    sig = _make_signal(sr, n, [(4_000.0, 1.0)])
    r = compute_thd(sig, sr, window="flattop", n_harmonics=9)
    assert r.n_harmonics == 0   # no harmonics fit within Nyquist
