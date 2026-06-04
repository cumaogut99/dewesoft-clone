"""
Tests for trigger, math engine, playback engine, and project I/O.
"""
import sys, os, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from src.core.trigger import find_trigger, align_to_trigger, TriggerConfig
from src.core.math_engine import MathEngine, MathChannelDef, OPERATIONS
from src.core.channel import Channel
from src.io.project import save_project, load_project
from src.core.signal_generator import SignalGenerator, WaveformType


# ---- Trigger ------------------------------------------------------------

def test_rising_trigger_found():
    # Sine wave starting from 0: should cross 0.5 on rising edge
    t = np.linspace(0, 1, 10_000)
    data = np.sin(2 * np.pi * 5 * t)
    idx = find_trigger(data, level=0.5, edge="rising")
    assert idx > 0
    assert data[idx] >= 0.5
    assert data[idx - 1] < 0.5


def test_falling_trigger_found():
    t = np.linspace(0, 1, 10_000)
    data = np.sin(2 * np.pi * 5 * t)
    idx = find_trigger(data, level=0.5, edge="falling")
    assert idx > 0
    assert data[idx] <= 0.5


def test_trigger_not_found_below_level():
    data = np.zeros(1000)  # flat zero — never crosses 1.0
    idx = find_trigger(data, level=1.0, edge="rising")
    assert idx == -1


def test_trigger_hysteresis_prevents_retrigger():
    # Noisy signal near level 0 — hysteresis should suppress false triggers
    np.random.seed(42)
    data = np.random.randn(1000) * 0.01  # tiny noise around 0
    # With large hysteresis=1.0, no trigger should be found
    idx = find_trigger(data, level=0.0, edge="rising", hysteresis=1.0)
    assert idx == -1


def test_either_edge():
    t = np.linspace(0, 1, 10_000)
    data = np.sin(2 * np.pi * 3 * t)
    idx = find_trigger(data, level=0.0, edge="either")
    assert idx > 0


def test_align_to_trigger_returns_slice():
    sr = 1000.0
    t = np.linspace(0, 2, int(2 * sr))
    data = np.sin(2 * np.pi * 5 * t)
    cfg = TriggerConfig(enabled=True, edge="rising", level=0.5, pre_samples=10)
    ts_out, data_out = align_to_trigger(t, data, cfg, n_display=200)
    assert len(ts_out) <= 200
    assert len(data_out) == len(ts_out)


def test_align_passthrough_when_disabled():
    t = np.linspace(0, 1, 1000)
    data = np.ones(1000)
    cfg = TriggerConfig(enabled=False)
    ts_out, data_out = align_to_trigger(t, data, cfg, n_display=100)
    assert len(data_out) == 100  # last 100 samples


# ---- MathEngine ---------------------------------------------------------

def _make_channel(ch_id=1, n=500, sr=1000.0, amp=1.0, freq=10.0):
    ch = Channel(id=ch_id, name=f"CH{ch_id}", sample_rate=sr, _buffer_size=2000)
    t = np.linspace(0, n / sr, n)
    data = amp * np.sin(2 * np.pi * freq * t)
    ch.push(data, t)
    return ch


def test_math_rms_output_is_positive():
    src = _make_channel()
    out = Channel(id=99, name="RMS", sample_rate=1000.0, _buffer_size=2000)
    defn = MathChannelDef(source_id=1, operation="RMS", output_channel=out,
                          params={"window": 50})
    engine = MathEngine()
    engine.add(defn)
    engine.update([src])
    _, data = out.get_data()
    assert len(data) > 0
    assert (data >= 0).all()


def test_math_integral_increases():
    src = _make_channel(amp=1.0, freq=1.0)
    # Set up a DC-ish signal: use absolute value to ensure positive area
    ch = Channel(id=1, name="CH1", sample_rate=1000.0, _buffer_size=2000)
    t = np.linspace(0, 0.5, 500)
    ch.push(np.ones(500), t)

    out = Channel(id=99, name="Int", sample_rate=1000.0, _buffer_size=2000)
    defn = MathChannelDef(source_id=1, operation="Integral", output_channel=out)
    engine = MathEngine()
    engine.add(defn)
    engine.update([ch])
    _, data = out.get_data()
    assert data[-1] > data[0]  # integral of 1 should be monotonically increasing


def test_math_lowpass_attenuates_high_freq():
    sr = 1000.0
    t = np.linspace(0, 1, int(sr))
    low_freq = np.sin(2 * np.pi * 5 * t)
    high_freq = np.sin(2 * np.pi * 400 * t)
    mixed = low_freq + high_freq   # std ≈ √(0.5+0.5) ≈ 1.0

    src = Channel(id=1, name="CH1", sample_rate=sr, _buffer_size=2000)
    src.push(mixed, t)

    out = Channel(id=99, name="Filt", sample_rate=sr, _buffer_size=2000)
    defn = MathChannelDef(source_id=1, operation="Low-pass", output_channel=out,
                          params={"cutoff": 50.0})
    engine = MathEngine()
    engine.add(defn)
    engine.update([src])
    _, data = out.get_data()
    # After 50 Hz LP, 400 Hz component is strongly attenuated.
    # Filtered std ≈ √0.5 ≈ 0.707 (only 5 Hz remains).
    # Verify it's significantly less than the mixed signal's std (≈ 1.0).
    assert data.std() < mixed.std() * 0.85


def test_math_engine_remove():
    src = _make_channel()
    out = Channel(id=99, name="M", sample_rate=1000.0, _buffer_size=2000)
    defn = MathChannelDef(source_id=1, operation="Abs", output_channel=out)
    engine = MathEngine()
    engine.add(defn)
    engine.remove(99)
    engine.update([src])
    _, data = out.get_data()
    assert len(data) == 0  # removed — no update


def test_math_all_operations_run():
    src = _make_channel()
    engine = MathEngine()
    for i, op in enumerate(OPERATIONS.keys()):
        out = Channel(id=100 + i, name=f"M{i}", sample_rate=1000.0, _buffer_size=2000)
        engine.add(MathChannelDef(source_id=1, operation=op, output_channel=out))
    engine.update([src])   # should not raise


# ---- Project I/O --------------------------------------------------------

def test_save_and_load_project_roundtrip():
    channels = [
        Channel(id=1, name="CH1", unit="V", sample_rate=5000.0, color="#00ff88",
                scale=2.0, offset=-1.0),
        Channel(id=2, name="CH2", unit="°C", sample_rate=1000.0, color="#ff6b6b"),
    ]
    generators = [
        SignalGenerator(waveform=WaveformType.SINE, frequency=50.0, amplitude=2.5,
                        dc_offset=0.5, phase_deg=45.0, noise_level=0.1),
        SignalGenerator(waveform=WaveformType.CHIRP, chirp_f0=1.0, chirp_f1=500.0),
    ]
    with tempfile.NamedTemporaryFile(suffix=".daqproj", delete=False) as f:
        path = f.name

    try:
        save_project(channels, generators, hardware_config={"driver": "simulated"}, path=path)
        doc = load_project(path)

        loaded_chs = doc["channels"]
        loaded_gens = doc["generators"]

        assert len(loaded_chs) == 2
        assert loaded_chs[0].name == "CH1"
        assert loaded_chs[0].sample_rate == 5000.0
        assert loaded_chs[0].scale == 2.0
        assert loaded_chs[0].offset == -1.0
        assert loaded_chs[1].unit == "°C"

        assert len(loaded_gens) == 2
        assert loaded_gens[0].waveform == WaveformType.SINE
        assert abs(loaded_gens[0].frequency - 50.0) < 1e-9
        assert abs(loaded_gens[0].phase_deg - 45.0) < 1e-9
        assert loaded_gens[1].waveform == WaveformType.CHIRP

        assert doc["hardware"]["driver"] == "simulated"
    finally:
        os.unlink(path)


def test_load_project_missing_file():
    with pytest.raises(Exception):
        load_project("/nonexistent/path/project.daqproj")
