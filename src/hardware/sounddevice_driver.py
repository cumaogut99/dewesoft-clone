"""
sounddevice driver — uses the computer's audio input (microphone / line-in) as a DAQ.
- Works with any ASIO/WASAPI/CoreAudio device
- Supports up to 192 kHz (device dependent)
- The sounddevice callback runs in a C thread → no Python GIL overhead

Install: pip install sounddevice
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import time

from .base import BaseDriver, DriverInfo, SampleCallback


class SounddeviceDriver(BaseDriver):
    INFO = DriverInfo(
        name="Sound Card (sounddevice)",
        description="Audio input card as DAQ — up to 192 kHz, no extra hardware needed",
        n_channels=2,
        max_sample_rate=192_000.0,
        requires=["sounddevice"],
    )

    def __init__(
        self,
        n_channels: int = 2,
        sample_rate: float = 44100.0,
        device: Optional[int | str] = None,
        latency: str = "low",
    ) -> None:
        super().__init__(n_channels, sample_rate)
        self._device = device
        self._latency = latency
        self._stream = None
        self._t0: float = 0.0

    @classmethod
    def probe(cls) -> bool:
        try:
            import sounddevice
            return True
        except ImportError:
            return False

    @classmethod
    def list_devices(cls) -> str:
        import sounddevice as sd
        return str(sd.query_devices())

    def start(self, callback: SampleCallback) -> None:
        import sounddevice as sd

        self._callback = callback
        self._t0 = time.perf_counter()

        def _sd_callback(indata, frames, time_info, status):
            # indata: (frames, channels) float32
            t_start = time_info.inputBufferAdcTime or (time.perf_counter() - self._t0)
            dt = 1.0 / self.sample_rate
            timestamps = t_start + np.arange(frames) * dt

            # shape: (n_channels, n_samples)
            data = indata[:, :self.n_channels].T.astype(np.float64)
            callback(data, timestamps)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.n_channels,
            device=self._device,
            latency=self._latency,
            dtype="float32",
            callback=_sd_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
