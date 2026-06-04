"""
Oscilloscope widget — real-time multi-channel waveform display.

Performance features:
  - MinMax decimation: 100k+ samples → ≤ 4000 plot points per update
  - pyqtgraph with OpenGL hints (set useOpenGL=True in pg.setConfigOptions)
  - Per-channel Y-axis autoscale with configurable padding
"""
from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QCheckBox, QPushButton
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ..core.channel import Channel
from ..core.decimator import auto_decimate

MAX_DISPLAY_POINTS = 4000   # min/max pairs after decimation


class OscilloscopeWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._time_window = 2.0
        self._decimate = True

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Time window:"))

        self._tw_spin = QDoubleSpinBox()
        self._tw_spin.setRange(0.01, 120.0)
        self._tw_spin.setValue(self._time_window)
        self._tw_spin.setSuffix(" s")
        self._tw_spin.valueChanged.connect(lambda v: setattr(self, "_time_window", v))
        toolbar.addWidget(self._tw_spin)

        toolbar.addWidget(QLabel("  Trigger:"))
        self._trig_combo = QComboBox()
        self._trig_combo.addItems(["Free run", "Rising edge", "Falling edge"])
        toolbar.addWidget(self._trig_combo)

        self._dec_cb = QCheckBox("Decimate")
        self._dec_cb.setChecked(True)
        self._dec_cb.setToolTip(
            "Min-max decimation: reduces 100k+ points to 4k without losing peaks"
        )
        self._dec_cb.toggled.connect(lambda v: setattr(self, "_decimate", v))
        toolbar.addWidget(self._dec_cb)

        toolbar.addStretch()

        for ch in self.channels:
            cb = QCheckBox(ch.name)
            cb.setChecked(ch.enabled)
            cb.setStyleSheet(f"color: {ch.color};")
            cb.toggled.connect(lambda checked, c=ch: self._toggle_channel(c, checked))
            toolbar.addWidget(cb)

        layout.addLayout(toolbar)

        # Plot
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

    def update_plots(self) -> None:
        for ch in self.channels:
            curve = self._curves.get(ch.id)
            if curve is None:
                continue
            if not ch.enabled:
                curve.setVisible(False)
                continue

            ts, data = ch.get_data()
            if len(ts) < 2:
                continue

            # Window
            t_end = ts[-1]
            mask = ts >= (t_end - self._time_window)
            ts_w, data_w = ts[mask], data[mask]

            # Decimate for display
            if self._decimate:
                ts_w, data_w = auto_decimate(ts_w, data_w, MAX_DISPLAY_POINTS)

            curve.setData(ts_w - (t_end - self._time_window), data_w)
            curve.setVisible(True)

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
