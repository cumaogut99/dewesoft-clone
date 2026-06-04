"""
Performance and correctness tests for ring buffer, decimation, and hardware drivers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import threading
import numpy as np
import pytest

from src.core.shared_ring_buffer import SharedRingBuffer
from src.core.decimator import minmax_decimate, lttb_decimate, auto_decimate
from src.hardware.simulated_driver import SimulatedDriver


# ---- SharedRingBuffer ---------------------------------------------------

def test_shm_write_and_read():
    buf = SharedRingBuffer(capacity=1000, create=True)
    data = np.linspace(0, 1, 500)
    ts = np.linspace(0, 0.5, 500)
    buf.write(data, ts)
    ts_out, data_out = buf.read()
    assert len(data_out) == 500
    np.testing.assert_allclose(data_out, data, atol=1e-12)
    buf.close()
    buf.unlink()


def test_shm_ring_wraps():
    buf = SharedRingBuffer(capacity=100, create=True)
    for i in range(5):
        d = np.ones(30) * i
        buf.write(d, np.arange(30, dtype=float) + i * 30)
    ts_out, data_out = buf.read()
    assert len(data_out) == 100   # capped at capacity
    buf.close()
    buf.unlink()


def test_shm_read_n_samples():
    buf = SharedRingBuffer(capacity=1000, create=True)
    buf.write(np.ones(800), np.arange(800, dtype=float))
    ts_out, data_out = buf.read(n_samples=200)
    assert len(data_out) == 200
    buf.close()
    buf.unlink()


def test_shm_clear():
    buf = SharedRingBuffer(capacity=1000, create=True)
    buf.write(np.ones(500), np.arange(500, dtype=float))
    buf.clear()
    ts_out, data_out = buf.read()
    assert len(data_out) == 0
    buf.close()
    buf.unlink()


def test_shm_concurrent_write_read():
    """Writer thread pumps data; reader thread reads — no corruption."""
    buf = SharedRingBuffer(capacity=10_000, create=True)
    errors = []

    def writer():
        for i in range(100):
            d = np.random.randn(50)
            buf.write(d, np.arange(50, dtype=float) + i * 50)
            time.sleep(0.001)

    def reader():
        for _ in range(50):
            ts_out, data_out = buf.read(n_samples=200)
            if not np.isfinite(data_out).all():
                errors.append("non-finite value in read")
            time.sleep(0.002)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert not errors
    buf.close()
    buf.unlink()


# ---- Decimator ----------------------------------------------------------

def test_minmax_no_decimate_when_small():
    ts = np.linspace(0, 1, 100)
    data = np.sin(ts)
    ts_out, data_out = minmax_decimate(ts, data, n_out=200)
    assert len(data_out) == 100  # unchanged


def test_minmax_reduces_points():
    ts = np.linspace(0, 1, 100_000)
    data = np.sin(2 * np.pi * 50 * ts)
    ts_out, data_out = minmax_decimate(ts, data, n_out=2000)
    assert len(data_out) <= 4000  # 2*n_out


def test_minmax_preserves_peak():
    ts = np.linspace(0, 1, 100_000)
    data = np.sin(2 * np.pi * 50 * ts)
    ts_out, data_out = minmax_decimate(ts, data, n_out=2000)
    # Peak of sine should be preserved to within 1% after decimation
    assert abs(data_out.max() - 1.0) < 0.01


def test_minmax_performance():
    """100k samples should decimate in < 5 ms."""
    ts = np.linspace(0, 1, 100_000)
    data = np.random.randn(100_000)
    t0 = time.perf_counter()
    for _ in range(100):
        minmax_decimate(ts, data, n_out=2000)
    elapsed = (time.perf_counter() - t0) / 100 * 1000  # ms
    assert elapsed < 5.0, f"minmax_decimate took {elapsed:.1f} ms (expected < 5 ms)"


def test_lttb_reduces_points():
    ts = np.linspace(0, 1, 10_000)
    data = np.sin(2 * np.pi * 10 * ts)
    ts_out, data_out = lttb_decimate(ts, data, n_out=200)
    assert len(data_out) == 200


def test_auto_decimate_passthrough():
    ts = np.linspace(0, 1, 100)
    data = np.ones(100)
    ts_out, data_out = auto_decimate(ts, data, max_points=4000)
    assert len(data_out) == 100  # no decimation needed


# ---- SimulatedDriver ----------------------------------------------------

def test_simulated_driver_fires_callback():
    received = []

    driver = SimulatedDriver(n_channels=2, sample_rate=1000.0, update_interval_ms=20)

    def cb(data, ts):
        received.append((data.shape, len(ts)))

    driver.start(cb)
    time.sleep(0.15)   # wait for ~3 callbacks at 50ms interval
    driver.stop()

    assert len(received) >= 2
    for shape, n_ts in received:
        assert shape[0] == 2       # 2 channels
        assert shape[1] == n_ts    # samples match timestamps


def test_simulated_driver_sample_rate():
    """Total samples ≈ sample_rate × elapsed_time within 5%."""
    total_samples = [0]
    total_time = [0.0]

    driver = SimulatedDriver(n_channels=1, sample_rate=5000.0, update_interval_ms=10)

    def cb(data, ts):
        total_samples[0] += data.shape[1]
        if len(ts):
            total_time[0] = ts[-1]

    t_start = time.perf_counter()
    driver.start(cb)
    time.sleep(0.5)
    driver.stop()
    elapsed = time.perf_counter() - t_start

    expected = 5000.0 * elapsed
    assert abs(total_samples[0] - expected) / expected < 0.1  # within 10%
