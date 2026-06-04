"""
Oscilloscope widget — real-time multi-channel waveform display.
Uses pyqtgraph for GPU-accelerated rendering.
"""
from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox, QCheckBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ..core.channel import Channel


class OscilloscopeWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._time_window = 2.0  # seconds visible

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Time window:"))
        self._tw_spin = QDoubleSpinBox()
        self._tw_spin.setRange(0.1, 60.0)
        self._tw_spin.setValue(self._time_window)
        self._tw_spin.setSuffix(" s")
        self._tw_spin.valueChanged.connect(self._on_time_window_changed)
        toolbar.addWidget(self._tw_spin)

        toolbar.addWidget(QLabel("Trigger:"))
        self._trig_combo = QComboBox()
        self._trig_combo.addItems(["Free run", "Rising edge", "Falling edge"])
        toolbar.addWidget(self._trig_combo)
        toolbar.addStretch()

        for ch in self.channels:
            cb = QCheckBox(ch.name)
            cb.setChecked(ch.enabled)
            cb.setStyleSheet(f"color: {ch.color};")
            cb.toggled.connect(lambda checked, c=ch: self._toggle_channel(c, checked))
            toolbar.addWidget(cb)

        layout.addLayout(toolbar)

        # Plot area
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1e1e2e")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Amplitude")
        self._plot.addLegend(offset=(10, 10))

        for ch in self.channels:
            pen = pg.mkPen(color=QColor(ch.color), width=1.5)
            curve = self._plot.plot(pen=pen, name=ch.name)
            self._curves[ch.id] = curve

        layout.addWidget(self._plot)

    # ------------------------------------------------------------------
    def update_plots(self) -> None:
        for ch in self.channels:
            if not ch.enabled:
                continue
            curve = self._curves.get(ch.id)
            if curve is None:
                continue
            ts, data = ch.get_data()
            if len(ts) < 2:
                continue

            # Show only the last time_window seconds
            t_end = ts[-1]
            t_start = t_end - self._time_window
            mask = ts >= t_start
            curve.setData(ts[mask] - t_start, data[mask])

    def _on_time_window_changed(self, val: float) -> None:
        self._time_window = val

    def _toggle_channel(self, ch: Channel, enabled: bool) -> None:
        ch.enabled = enabled
        curve = self._curves.get(ch.id)
        if curve:
            curve.setVisible(enabled)

    def set_channels(self, channels: List[Channel]) -> None:
        self.channels = channels
        self._plot.clear()
        self._curves.clear()
        for ch in channels:
            pen = pg.mkPen(color=QColor(ch.color), width=1.5)
            curve = self._plot.plot(pen=pen, name=ch.name)
            self._curves[ch.id] = curve
