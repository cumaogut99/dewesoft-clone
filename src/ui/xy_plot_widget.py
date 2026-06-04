"""
XY Plot / Lissajous widget — channel X vs channel Y scatter plot.
Useful for: phase diagrams, Lissajous figures, hysteresis loops.
"""
from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ..core.channel import Channel
from ..core.decimator import minmax_decimate


class XYPlotWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("X axis:"))
        self._x_combo = QComboBox()
        toolbar.addWidget(self._x_combo)
        toolbar.addWidget(QLabel("  Y axis:"))
        self._y_combo = QComboBox()
        toolbar.addWidget(self._y_combo)

        toolbar.addWidget(QLabel("  Points:"))
        self._pts_spin = QSpinBox()
        self._pts_spin.setRange(100, 50_000)
        self._pts_spin.setValue(5000)
        self._pts_spin.setSingleStep(500)
        toolbar.addWidget(self._pts_spin)

        self._persist_cb = QCheckBox("Persistence")
        self._persist_cb.setToolTip("Keep previous traces (Lissajous mode)")
        toolbar.addWidget(self._persist_cb)

        self._line_cb = QCheckBox("Connect")
        toolbar.addWidget(self._line_cb)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1e1e2e")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setAspectLocked(False)

        self._scatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush("#00ff88aa"))
        self._line_curve = pg.PlotDataItem(pen=pg.mkPen("#00ff88", width=1))
        self._plot.addItem(self._scatter)
        self._plot.addItem(self._line_curve)
        layout.addWidget(self._plot)

        self._refresh_combos()

    def _refresh_combos(self) -> None:
        names = [ch.name for ch in self.channels]
        for cb in (self._x_combo, self._y_combo):
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(names)
            if cur in names:
                cb.setCurrentText(cur)
            cb.blockSignals(False)
        if len(names) >= 2:
            self._y_combo.setCurrentIndex(1)

    def update_plots(self) -> None:
        xi = self._x_combo.currentIndex()
        yi = self._y_combo.currentIndex()
        if xi < 0 or yi < 0 or xi >= len(self.channels) or yi >= len(self.channels):
            return

        n = self._pts_spin.value()
        _, x_data = self.channels[xi].get_data(n_samples=n)
        _, y_data = self.channels[yi].get_data(n_samples=n)

        if len(x_data) < 2 or len(y_data) < 2:
            return

        m = min(len(x_data), len(y_data))
        x, y = x_data[-m:], y_data[-m:]

        x_ch = self.channels[xi]
        y_ch = self.channels[yi]
        self._plot.setLabel("bottom", x_ch.name, units=x_ch.unit)
        self._plot.setLabel("left", y_ch.name, units=y_ch.unit)

        if not self._persist_cb.isChecked():
            if self._line_cb.isChecked():
                self._scatter.setData([], [])
                self._line_curve.setData(x, y)
            else:
                self._line_curve.setData([], [])
                self._scatter.setData(x=x, y=y)

    def set_channels(self, channels: List[Channel]) -> None:
        self.channels = channels
        self._refresh_combos()
