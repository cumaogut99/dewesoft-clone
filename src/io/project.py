"""
Project file — save and restore the complete session configuration as JSON.
Saved: channel metadata, generator settings, hardware config, UI preferences.
NOT saved: acquired data (use save_hdf5 for that).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List

from ..core.channel import Channel, ChannelType
from ..core.signal_generator import SignalGenerator, WaveformType


def save_project(
    channels: List[Channel],
    generators: List[SignalGenerator],
    hardware_config: dict | None = None,
    path: str | Path = "project.daqproj",
) -> None:
    path = Path(path)
    doc = {
        "version": 1,
        "channels": [
            {
                "id": ch.id,
                "name": ch.name,
                "unit": ch.unit,
                "sample_rate": ch.sample_rate,
                "color": ch.color,
                "enabled": ch.enabled,
                "scale": ch.scale,
                "offset": ch.offset,
                "buffer_size": ch._buffer_size,
            }
            for ch in channels
        ],
        "generators": [
            {
                "waveform": gen.waveform.value,
                "frequency": gen.frequency,
                "amplitude": gen.amplitude,
                "dc_offset": gen.dc_offset,
                "phase_deg": gen.phase_deg,
                "noise_level": gen.noise_level,
                "chirp_f0": gen.chirp_f0,
                "chirp_f1": gen.chirp_f1,
                "chirp_period": gen.chirp_period,
            }
            for gen in generators
        ],
        "hardware": hardware_config or {},
    }
    path.write_text(json.dumps(doc, indent=2))


def load_project(path: str | Path) -> dict:
    """
    Returns a dict with keys 'channels', 'generators', 'hardware'.
    Caller reconstructs Channel / SignalGenerator objects from these.
    """
    path = Path(path)
    doc = json.loads(path.read_text())

    channels = [
        Channel(
            id=c["id"],
            name=c["name"],
            unit=c["unit"],
            sample_rate=c["sample_rate"],
            color=c["color"],
            enabled=c["enabled"],
            scale=c.get("scale", 1.0),
            offset=c.get("offset", 0.0),
            _buffer_size=c.get("buffer_size", 10_000),
        )
        for c in doc["channels"]
    ]

    generators = [
        SignalGenerator(
            waveform=WaveformType(g["waveform"]),
            frequency=g["frequency"],
            amplitude=g["amplitude"],
            dc_offset=g.get("dc_offset", 0.0),
            phase_deg=g.get("phase_deg", 0.0),
            noise_level=g.get("noise_level", 0.0),
            chirp_f0=g.get("chirp_f0", 1.0),
            chirp_f1=g.get("chirp_f1", 100.0),
            chirp_period=g.get("chirp_period", 1.0),
        )
        for g in doc["generators"]
    ]

    return {
        "channels": channels,
        "generators": generators,
        "hardware": doc.get("hardware", {}),
        "version": doc.get("version", 1),
    }
