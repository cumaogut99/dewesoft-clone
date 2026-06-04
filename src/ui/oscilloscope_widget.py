"""
Oscilloscope widget — real-time multi-channel waveform display.

Features:
  - MinMax decimation (100k+ samples → ≤ 4000 points)
  - Edge trigger with hysteresis
  - Two vertical cursors + delta time / delta freq readout
  - Per-channel visibility toggles
"""
from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ..core.channel import Channel
from ..core.decimator import auto_decimate
from ..core.trigger import TriggerConfig, align_to_trigger

MAX_DISPLAY_POINTS = 4000


class OscilloscopeWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._time_window = 2.0
        self._decimate = True
        self._trigger = TriggerConfig()
        self._cursors_enabled = False
        self._cursor_a: pg.InfiniteLine | None = None
        self._cursor_b: pg.InfiniteLine | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- top toolbar ----
        top = QHBoxLayout()
        top.addWidget(QLabel("Window:"))
        self._tw_spin = QDoubleSpinBox()
        self._tw_spin.setRange(0.01, 120.0)
        self._tw_spin.setValue(self._time_window)
        self._tw_spin.setSuffix(" s")
        self._tw_spin.valueChanged.connect(lambda v: setattr(self, "_time_window", v))
        top.addWidget(self._tw_spin)

        self._dec_cb = QCheckBox("Decimate")
        self._dec_cb.setChecked(True)
        self._dec_cb.toggled.connect(lambda v: setattr(self, "_decimate", v))
        top.addWidget(self._dec_cb)

        self._cur_cb = QCheckBox("Cursors")
        self._cur_cb.toggled.connect(self._toggle_cursors)
        top.addWidget(self._cur_cb)

        top.addStretch()
        for ch in self.channels:
            cb = QCheckBox(ch.name)
            cb.setChecked(ch.enabled)
            cb.setStyleSheet(f"color: {ch.color};")
            cb.toggled.connect(lambda checked, c=ch: self._toggle_channel(c, checked))
            top.addWidget(cb)
        layout.addLayout(top)

        # ---- trigger row ----
        trig = QHBoxLayout()
        trig.addWidget(QLabel("Trigger:"))
        self._trig_mode = QComboBox()
        self._trig_mode.addItems(["Free run", "Rising ↑", "Falling ↓", "Either ↕"])
        self._trig_mode.currentTextChanged.connect(self._on_trig_mode)
        trig.addWidget(self._trig_mode)

        trig.addWidget(QLabel("Channel:"))
        self._trig_ch = QComboBox()
        for ch in self.channels:
            self._trig_ch.addItem(ch.name, userData=ch.id)
        trig.addWidget(self._trig_ch)

        trig.addWidget(QLabel("Level:"))
        self._trig_level = QDoubleSpinBox()
        self._trig_level.setRange(-1e6, 1e6)
        self._trig_level.setValue(0.0)
        self._trig_level.setSingleStep(0.1)
        self._trig_level.valueChanged.connect(lambda v: setattr(self._trigger, "level", v))
        trig.addWidget(self._trig_level)

        trig.addWidget(QLabel("Hyst:"))
        self._trig_hyst = QDoubleSpinBox()
        self._trig_hyst.setRange(0.0, 1e4)
        self._trig_hyst.setValue(0.02)
        self._trig_hyst.setSingleStep(0.01)
        self._trig_hyst.valueChanged.connect(lambda v: setattr(self._trigger, "hysteresis", v))
        trig.addWidget(self._trig_hyst)

        # Cursor readout
        self._cur_lbl = QLabel("")
        self._cur_lbl.setStyleSheet("color: #cba6f7; font-family: monospace;")
        trig.addStretch()
        trig.addWidget(self._cur_lbl)
        layout.addLayout(trig)

        # ---- plot ----
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

        # Trigger level marker
        self._trig_marker = pg.InfiniteLine(
            pos=0.0, angle=0,
            pen=pg.mkPen("#ff6b6b", width=1, style=Qt.PenStyle.DashLine),
            movable=True,
        )
        self._trig_marker.sigPositionChanged.connect(
            lambda ln: setattr(self._trigger, "level", ln.value())
        )
        self._trig_marker.sigPositionChanged.connect(
            lambda ln: self._trig_level.setValue(ln.value())
        )
        self._plot.addItem(self._trig_marker)

        layout.addWidget(self._plot)

    # ------------------------------------------------------------------
    def update_plots(self) -> None:
        trig_ch_id = self._trig_ch.currentData()

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

            # Window slice
            t_end = ts[-1]
            mask = ts >= (t_end - self._time_window)
            ts_w, data_w = ts[mask], data[mask]

            # Trigger alignment (on the designated trigger channel)
            if self._trigger.enabled and ch.id == trig_ch_id:
                n_disp = len(ts_w)
                ts_w, data_w = align_to_trigger(ts_w, data_w, self._trigger, n_disp)

            # Decimate
            if self._decimate:
                ts_w, data_w = auto_decimate(ts_w, data_w, MAX_DISPLAY_POINTS)

            t0 = ts_w[0] if len(ts_w) > 0 else 0.0
            curve.setData(ts_w - t0, data_w)
            curve.setVisible(True)

        self._update_cursor_readout()

    def _on_trig_mode(self, text: str) -> None:
        if text == "Free run":
            self._trigger.enabled = False
        else:
            self._trigger.enabled = True
            edge_map = {"Rising ↑": "rising", "Falling ↓": "falling", "Either ↕": "either"}
            self._trigger.edge = edge_map.get(text, "rising")

    # ------------------------------------------------------------------
    # Cursors
    # ------------------------------------------------------------------
    def _toggle_cursors(self, enabled: bool) -> None:
        self._cursors_enabled = enabled
        if enabled:
            if self._cursor_a is None:
                self._cursor_a = pg.InfiniteLine(
                    pos=0.2, angle=90, movable=True,
                    pen=pg.mkPen("#89dceb", width=1.5, style=Qt.PenStyle.DashLine),
                    label="A", labelOpts={"color": "#89dceb"},
                )
                self._cursor_b = pg.InfiniteLine(
                    pos=0.8, angle=90, movable=True,
                    pen=pg.mkPen("#f9e2af", width=1.5, style=Qt.PenStyle.DashLine),
                    label="B", labelOpts={"color": "#f9e2af"},
                )
                self._cursor_a.sigPositionChanged.connect(self._update_cursor_readout)
                self._cursor_b.sigPositionChanged.connect(self._update_cursor_readout)
                self._plot.addItem(self._cursor_a)
                self._plot.addItem(self._cursor_b)
        else:
            if self._cursor_a:
                self._plot.removeItem(self._cursor_a)
                self._plot.removeItem(self._cursor_b)
                self._cursor_a = None
                self._cursor_b = None
            self._cur_lbl.setText("")

    def _update_cursor_readout(self) -> None:
        if not self._cursors_enabled or self._cursor_a is None:
            return
        a = self._cursor_a.value()
        b = self._cursor_b.value()
        dt = abs(b - a)
        freq = (1.0 / dt) if dt > 1e-9 else float("inf")
        self._cur_lbl.setText(
            f"A={a:.4f}s  B={b:.4f}s  ΔT={dt:.4f}s  1/ΔT={freq:.2f}Hz"
        )

    # ------------------------------------------------------------------
    def _toggle_channel(self, ch: Channel, enabled: bool) -> None:
        ch.enabled = enabled
        curve = self._curves.get(ch.id)
        if curve:
            curve.setVisible(enabled)

    def set_channels(self, channels: List[Channel]) -> None:
        self.channels = channels
        self._plot.clear()
        self._curves.clear()
        self._trig_marker = pg.InfiniteLine(
            pos=0.0, angle=0,
            pen=pg.mkPen("#ff6b6b", width=1, style=Qt.PenStyle.DashLine),
            movable=True,
        )
        self._plot.addItem(self._trig_marker)
        for ch in channels:
            pen = pg.mkPen(color=QColor(ch.color), width=1.5)
            curve = self._plot.plot(pen=pen, name=ch.name)
            self._curves[ch.id] = curve
        # Refresh trigger channel combo
        self._trig_ch.clear()
        for ch in channels:
            self._trig_ch.addItem(ch.name, userData=ch.id)
