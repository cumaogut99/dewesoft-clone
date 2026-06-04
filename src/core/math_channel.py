"""
Math channel — computes derived signals from existing channels at runtime.
Supported operations: +, -, *, /, RMS, integral, derivative, filter.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Callable


def rms(data: np.ndarray) -> np.ndarray:
    """Running RMS over full array (scalar output per call)."""
    if len(data) == 0:
        return np.array([0.0])
    return np.array([np.sqrt(np.mean(data ** 2))])


def running_rms(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Element-wise RMS with sliding window."""
    out = np.empty(len(data))
    for i in range(len(data)):
        start = max(0, i - window + 1)
        out[i] = np.sqrt(np.mean(data[start:i + 1] ** 2))
    return out


def cumulative_integral(data: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(data) * dt


def derivative(data: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(data, dt)


def lowpass_filter(data: np.ndarray, cutoff_hz: float, sample_rate: float, order: int = 4) -> np.ndarray:
    nyq = sample_rate / 2.0
    if cutoff_hz >= nyq:
        return data.copy()
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, data) if len(data) > 3 * order else data.copy()


def highpass_filter(data: np.ndarray, cutoff_hz: float, sample_rate: float, order: int = 4) -> np.ndarray:
    nyq = sample_rate / 2.0
    if cutoff_hz <= 0:
        return data.copy()
    b, a = butter(order, cutoff_hz / nyq, btype="high")
    return filtfilt(b, a, data) if len(data) > 3 * order else data.copy()


def bandpass_filter(data: np.ndarray, low_hz: float, high_hz: float, sample_rate: float, order: int = 4) -> np.ndarray:
    nyq = sample_rate / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, data) if len(data) > 3 * order else data.copy()


# Registry of named operations callable from the UI
MATH_OPERATIONS: dict[str, Callable] = {
    "RMS": running_rms,
    "Integral": cumulative_integral,
    "Derivative": derivative,
    "Low-pass": lowpass_filter,
    "High-pass": highpass_filter,
    "Band-pass": bandpass_filter,
}
