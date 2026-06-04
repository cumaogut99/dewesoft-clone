"""
Software signal generator — simulates hardware input for testing.
Generates sine, square, triangle, sawtooth, noise, and chirp waveforms.
"""
from __future__ import annotations
from enum import Enum
import numpy as np


class WaveformType(Enum):
    SINE = "Sine"
    SQUARE = "Square"
    TRIANGLE = "Triangle"
    SAWTOOTH = "Sawtooth"
    NOISE = "Noise"
    CHIRP = "Chirp"
    DC = "DC"


class SignalGenerator:
    def __init__(
        self,
        waveform: WaveformType = WaveformType.SINE,
        frequency: float = 10.0,
        amplitude: float = 1.0,
        dc_offset: float = 0.0,
        phase_deg: float = 0.0,
        noise_level: float = 0.0,
        chirp_f0: float = 1.0,
        chirp_f1: float = 100.0,
        chirp_period: float = 1.0,
    ) -> None:
        self.waveform = waveform
        self.frequency = frequency
        self.amplitude = amplitude
        self.dc_offset = dc_offset
        self.phase_deg = phase_deg
        self.noise_level = noise_level
        self.chirp_f0 = chirp_f0
        self.chirp_f1 = chirp_f1
        self.chirp_period = chirp_period
        self._phase_accum = 0.0

    def generate(self, timestamps: np.ndarray) -> np.ndarray:
        t = timestamps
        phase = 2 * np.pi * self.frequency * t + np.deg2rad(self.phase_deg)

        match self.waveform:
            case WaveformType.SINE:
                signal = np.sin(phase)
            case WaveformType.SQUARE:
                signal = np.sign(np.sin(phase))
            case WaveformType.TRIANGLE:
                signal = 2 * np.abs(2 * (t * self.frequency - np.floor(t * self.frequency + 0.5))) - 1
            case WaveformType.SAWTOOTH:
                signal = 2 * (t * self.frequency - np.floor(t * self.frequency + 0.5))
            case WaveformType.NOISE:
                signal = np.random.randn(len(t))
            case WaveformType.CHIRP:
                k = (self.chirp_f1 - self.chirp_f0) / self.chirp_period
                inst_phase = 2 * np.pi * (self.chirp_f0 * (t % self.chirp_period) + 0.5 * k * (t % self.chirp_period) ** 2)
                signal = np.sin(inst_phase)
            case WaveformType.DC:
                signal = np.ones(len(t))
            case _:
                signal = np.zeros(len(t))

        if self.noise_level > 0:
            signal = signal + self.noise_level * np.random.randn(len(t))

        return self.amplitude * signal + self.dc_offset
