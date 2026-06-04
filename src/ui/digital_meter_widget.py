"""
Digital meter panel — shows live numeric values, min/max, and RMS for each channel.
"""
from __future__ import annotations
from typing import List
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QFrame, QVBoxLayout, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from ..core.channel import Channel


class _ChannelMeter(QFrame):
    def __init__(self, channel: Channel, parent=None) -> None:
        super().__init__(parent)
        self.channel = channel
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                border: 1px solid {channel.color};
                border-radius: 6px;
                background: #181825;
            }}
        """)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        name_lbl = QLabel(self.channel.name)
        name_lbl.setStyleSheet(f"color: {self.channel.color}; font-weight: bold; font-size: 11px;")
        layout.addWidget(name_lbl)

        self._value_lbl = QLabel("—")
        self._value_lbl.setFont(QFont("Courier New", 22, QFont.Weight.Bold))
        self._value_lbl.setStyleSheet(f"color: {self.channel.color};")
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._value_lbl)

        stats = QGridLayout()
        stats.setContentsMargins(0, 0, 0, 0)

        for col, label in enumerate(["Min", "Max", "RMS"]):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #666; font-size: 9px;")
            stats.addWidget(lbl, 0, col, Qt.AlignmentFlag.AlignCenter)

        self._min_lbl = QLabel("—")
        self._max_lbl = QLabel("—")
        self._rms_lbl = QLabel("—")

        for col, lbl in enumerate([self._min_lbl, self._max_lbl, self._rms_lbl]):
            lbl.setStyleSheet("color: #aaa; font-size: 10px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stats.addWidget(lbl, 1, col)

        layout.addLayout(stats)

        unit_lbl = QLabel(self.channel.unit)
        unit_lbl.setStyleSheet("color: #555; font-size: 9px;")
        unit_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(unit_lbl)

    def refresh(self) -> None:
        ts, data = self.channel.get_data(n_samples=500)
        if len(data) == 0:
            return
        val = self.channel.latest_value
        self._value_lbl.setText(f"{val:+.4f}")
        self._min_lbl.setText(f"{data.min():.3f}")
        self._max_lbl.setText(f"{data.max():.3f}")
        rms = float(np.sqrt(np.mean(data ** 2)))
        self._rms_lbl.setText(f"{rms:.3f}")


class DigitalMeterWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._meters: list[_ChannelMeter] = []
        self._build_ui()

    def _build_ui(self) -> None:
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setSpacing(8)
        self._rebuild_meters()

    def _rebuild_meters(self) -> None:
        # Clear
        for m in self._meters:
            self._grid.removeWidget(m)
            m.deleteLater()
        self._meters.clear()

        cols = 4
        for i, ch in enumerate(self.channels):
            meter = _ChannelMeter(ch)
            meter.setMinimumSize(160, 120)
            meter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._meters.append(meter)
            self._grid.addWidget(meter, i // cols, i % cols)

    def update_plots(self) -> None:
        for m in self._meters:
            if m.channel.enabled:
                m.refresh()

    def set_channels(self, channels: List[Channel]) -> None:
        self.channels = channels
        self._rebuild_meters()
