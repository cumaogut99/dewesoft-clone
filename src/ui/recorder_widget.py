"""
Data recorder widget — strip-chart style scrolling recorder similar to DeweSoft's recorder.
Shows all channels stacked vertically with individual Y-axis scaling.
"""
from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDoubleSpinBox, QSplitter
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ..core.channel import Channel

CHANNEL_COLORS = [
    "#00ff88", "#ff6b6b", "#4ecdc4", "#ffe66d",
    "#a8e6cf", "#ff8b94", "#b8b8ff", "#ffd3b6",
]


class RecorderWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._plots: list[pg.PlotWidget] = []
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._time_window = 10.0

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Record window:"))
        self._tw_spin = QDoubleSpinBox()
        self._tw_spin.setRange(1.0, 300.0)
        self._tw_spin.setValue(self._time_window)
        self._tw_spin.setSuffix(" s")
        self._tw_spin.valueChanged.connect(lambda v: setattr(self, "_time_window", v))
        toolbar.addWidget(self._tw_spin)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # One sub-plot per channel, stacked in a splitter
        self._splitter = QSplitter(Qt.Orientation.Vertical)

        for i, ch in enumerate(self.channels):
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            pw = pg.PlotWidget()
            pw.setBackground("#1e1e2e")
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.setLabel("left", ch.name, units=ch.unit)
            pw.setMaximumHeight(180)
            pw.setMinimumHeight(80)

            curve = pw.plot(pen=pg.mkPen(color=QColor(color), width=1.5))
            self._curves[ch.id] = curve
            self._plots.append(pw)
            self._splitter.addWidget(pw)

        layout.addWidget(self._splitter)

    def update_plots(self) -> None:
        for ch in self.channels:
            curve = self._curves.get(ch.id)
            if curve is None or not ch.enabled:
                continue
            ts, data = ch.get_data()
            if len(ts) < 2:
                continue
            t_end = ts[-1]
            mask = ts >= (t_end - self._time_window)
            curve.setData(ts[mask], data[mask])
