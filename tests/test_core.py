"""
Core module tests — channel buffer, signal generator, FFT analyzer, math operations.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.channel import Channel, ChannelType
from src.core.signal_generator import SignalGenerator, WaveformType
from src.core.fft_analyzer import compute_fft, find_spectral_peaks
from src.core.math_channel import lowpass_filter, derivative, cumulative_integral


# ---- Channel tests -------------------------------------------------------

def test_channel_push_and_read():
    ch = Channel(id=1, name="Test", _buffer_size=1000)
    t = np.linspace(0, 1, 500)
    data = np.sin(2 * np.pi * 10 * t)
    ch.push(data, t)
    ts_out, data_out = ch.get_data()
    assert len(data_out) == 500
    np.testing.assert_allclose(data_out, data, atol=1e-10)


def test_channel_ring_buffer_wrap():
    ch = Channel(id=1, name="Test", _buffer_size=100)
    for _ in range(5):
        t = np.arange(50, dtype=float)
        ch.push(np.ones(50), t)
    ts_out, data_out = ch.get_data()
    assert len(data_out) == 100


def test_channel_scale_offset():
    ch = Channel(id=1, name="Test", scale=2.0, offset=1.0)
    t = np.array([0.0])
    ch.push(np.array([3.0]), t)
    assert abs(ch.latest_value - 7.0) < 1e-10  # 3*2+1


def test_channel_clear():
    ch = Channel(id=1, name="Test")
    ch.push(np.ones(50), np.arange(50, dtype=float))
    ch.clear()
    ts, data = ch.get_data()
    assert len(data) == 0


# ---- SignalGenerator tests -----------------------------------------------

@pytest.mark.parametrize("wave", list(WaveformType))
def test_signal_generator_all_waveforms(wave):
    gen = SignalGenerator(waveform=wave, frequency=10.0, amplitude=1.0)
    t = np.linspace(0, 1, 1000)
    sig = gen.generate(t)
    assert sig.shape == (1000,)
    assert np.isfinite(sig).all()


def test_sine_amplitude():
    gen = SignalGenerator(waveform=WaveformType.SINE, frequency=10.0, amplitude=2.0)
    t = np.linspace(0, 1, 100_000)
    sig = gen.generate(t)
    assert abs(sig.max() - 2.0) < 0.01


def test_noise_with_level():
    gen = SignalGenerator(waveform=WaveformType.SINE, noise_level=0.5)
    t = np.linspace(0, 1, 1000)
    sig = gen.generate(t)
    assert sig.std() > 0.1


# ---- FFT tests -----------------------------------------------------------

def test_fft_detects_frequency():
    sr = 1000.0
    t = np.arange(0, 1, 1 / sr)
    data = np.sin(2 * np.pi * 50 * t)  # 50 Hz
    freqs, mag = compute_fft(data, sr, window="hann", db_scale=True)
    peak_idx = np.argmax(mag)
    assert abs(freqs[peak_idx] - 50.0) < 2.0


def test_fft_empty_input():
    freqs, mag = compute_fft(np.array([]), 1000.0)
    assert len(freqs) == 0


def test_find_spectral_peaks():
    sr = 1000.0
    t = np.arange(0, 1, 1 / sr)
    data = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    freqs, mag = compute_fft(data, sr, db_scale=True)
    peaks = find_spectral_peaks(freqs, mag, n_peaks=2)
    peak_freqs = sorted([p[0] for p in peaks])
    assert abs(peak_freqs[0] - 50.0) < 3.0
    assert abs(peak_freqs[1] - 120.0) < 3.0


# ---- Math channel tests --------------------------------------------------

def test_lowpass_removes_high_freq():
    sr = 1000.0
    t = np.linspace(0, 1, int(sr))
    low = np.sin(2 * np.pi * 5 * t)
    high = np.sin(2 * np.pi * 200 * t)
    mixed = low + high
    filtered = lowpass_filter(mixed, cutoff_hz=20.0, sample_rate=sr)
    # High-freq component should be attenuated
    residual = filtered - low
    assert residual.std() < 0.1


def test_derivative_of_linear():
    t = np.linspace(0, 1, 1000)
    linear = 3.0 * t + 2.0
    deriv = derivative(linear, dt=t[1] - t[0])
    np.testing.assert_allclose(deriv[10:-10], 3.0, atol=0.01)


def test_cumulative_integral():
    t = np.linspace(0, 1, 1000)
    dt = t[1] - t[0]
    ones = np.ones(1000)
    integral = cumulative_integral(ones, dt)
    assert abs(integral[-1] - 1.0) < 0.01
