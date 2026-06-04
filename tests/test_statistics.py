"""
Tests for statistics computation and project-level coverage of new modules.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from src.core.channel import Channel
from src.core.statistics import compute_stats, compute_stats_all, ChannelStats


def _sine_channel(ch_id=1, freq=10.0, amp=1.0, sr=1000.0, n=1000):
    ch = Channel(id=ch_id, name=f"CH{ch_id}", unit="V",
                 sample_rate=sr, _buffer_size=n + 10)
    t = np.linspace(0, n / sr, n)
    ch.push(amp * np.sin(2 * np.pi * freq * t), t)
    return ch


# ---- Basic stats correctness ----------------------------------------

def test_stats_dc_signal():
    ch = Channel(id=1, name="DC", _buffer_size=1010)
    t = np.linspace(0, 1, 1000)
    ch.push(np.full(1000, 5.0), t)
    s = compute_stats(ch)
    assert abs(s.mean - 5.0) < 1e-6
    assert s.std < 1e-6
    assert abs(s.rms - 5.0) < 1e-6
    assert abs(s.minimum - 5.0) < 1e-6
    assert abs(s.maximum - 5.0) < 1e-6
    assert s.peak_to_peak < 1e-6
    assert s.n_samples == 1000


def test_stats_sine_rms():
    ch = _sine_channel(amp=1.0)
    s = compute_stats(ch)
    # RMS of sin(t) = 1/√2 ≈ 0.707
    assert abs(s.rms - 1.0 / math.sqrt(2)) < 0.01


def test_stats_sine_peak_to_peak():
    ch = _sine_channel(amp=2.0)
    s = compute_stats(ch)
    # Peak-to-peak ≈ 4.0 (−2 to +2)
    assert abs(s.peak_to_peak - 4.0) < 0.05


def test_stats_crest_factor_sine():
    ch = _sine_channel(amp=1.0)
    s = compute_stats(ch)
    # Crest factor of sine = √2 ≈ 1.414
    assert abs(s.crest_factor - math.sqrt(2)) < 0.05


def test_stats_empty_channel():
    ch = Channel(id=1, name="Empty")
    s = compute_stats(ch)
    assert s.n_samples == 0
    assert s.mean == 0.0


# ---- Range selection ------------------------------------------------

def test_stats_range_selection():
    ch = Channel(id=1, name="CH1", _buffer_size=2010)
    t = np.linspace(0, 2, 2000)
    # First second: zeros; second second: ones
    data = np.where(t < 1.0, 0.0, 1.0)
    ch.push(data, t)

    s_all = compute_stats(ch)
    assert abs(s_all.mean - 0.5) < 0.01

    s_second = compute_stats(ch, t_start=1.0, t_end=2.0)
    assert abs(s_second.mean - 1.0) < 0.01

    s_first = compute_stats(ch, t_start=0.0, t_end=1.0)
    assert abs(s_first.mean - 0.0) < 0.01


def test_stats_range_outside_data_returns_zero():
    ch = _sine_channel()
    s = compute_stats(ch, t_start=100.0, t_end=200.0)
    assert s.n_samples == 0


def test_stats_all_channels():
    channels = [_sine_channel(i + 1) for i in range(4)]
    results = compute_stats_all(channels)
    assert len(results) == 4
    for s in results:
        assert s.n_samples == 1000
        assert abs(s.rms - 1.0 / math.sqrt(2)) < 0.02


# ---- Duration -------------------------------------------------------

def test_stats_duration():
    ch = _sine_channel(n=1000, sr=1000.0)
    s = compute_stats(ch)
    # 1000 samples at 1000 Hz → ≈ 1.0 s
    assert abs(s.duration - 1.0) < 0.01
