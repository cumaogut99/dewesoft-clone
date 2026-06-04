"""
Analyze widget — post-acquisition analysis view.

Features:
  - Stacked strip-chart of all channels (or overlaid — switchable)
  - LinearRegionItem for A/B range selection → drives StatisticsWidget
  - Zoom / pan toolbar
  - Export selected range to HDF5 or CSV
  - FFT of selection
  - Linked X axes across all sub-plots
"""
from __future__ import annotations
from typing import List
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QCheckBox, QComboBox, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from ..core.channel import Channel
from ..core.decimator import auto_decimate
from ..core.fft_analyzer import compute_fft
from ..io.data_writer import save_hdf5, save_csv

CHANNEL_COLORS = [
    "#00ff88", "#ff6b6b", "#4ecdc4", "#ffe66d",
    "#a8e6cf", "#ff8b94", "#b8b8ff", "#ffd3b6",
]


class AnalyzeWidget(QWidget):
    """
    Post-processing analysis panel.
    Emits range_changed(t_start, t_end) when the selection region moves.
    """
    range_changed = pyqtSignal(float, float)

    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._plots: list[pg.PlotWidget] = []
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._region: pg.LinearRegionItem | None = None
        self._overlay_mode = False
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- toolbar ----
        tb = QHBoxLayout()

        self._overlay_cb = QCheckBox("Overlay channels")
        self._overlay_cb.toggled.connect(self._on_overlay_toggle)
        tb.addWidget(self._overlay_cb)

        tb.addWidget(QLabel("  "))

        self._region_cb = QCheckBox("Selection region (A/B)")
        self._region_cb.setChecked(True)
        self._region_cb.toggled.connect(self._toggle_region)
        tb.addWidget(self._region_cb)

        tb.addWidget(QLabel("  "))

        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setFixedWidth(40)
        zoom_in_btn.setToolTip("Zoom in (X axis)")
        zoom_in_btn.clicked.connect(self._zoom_in)
        tb.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setFixedWidth(40)
        zoom_out_btn.setToolTip("Zoom out (X axis)")
        zoom_out_btn.clicked.connect(self._zoom_out)
        tb.addWidget(zoom_out_btn)

        zoom_fit_btn = QPushButton("Fit all")
        zoom_fit_btn.clicked.connect(self._zoom_fit)
        tb.addWidget(zoom_fit_btn)

        tb.addStretch()

        self._sel_lbl = QLabel("Select a range with the blue region")
        self._sel_lbl.setStyleSheet("color: #89dceb; font-style: italic;")
        tb.addWidget(self._sel_lbl)

        tb.addStretch()

        export_hdf5_btn = QPushButton("💾 Export selection (HDF5)")
        export_hdf5_btn.clicked.connect(self._export_hdf5)
        tb.addWidget(export_hdf5_btn)

        export_csv_btn = QPushButton("📄 Export selection (CSV)")
        export_csv_btn.clicked.connect(self._export_csv)
        tb.addWidget(export_csv_btn)

        root.addLayout(tb)

        # ---- plots area ----
        self._plot_area = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_area)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_layout.setSpacing(2)

        root.addWidget(self._plot_area)

        self._rebuild_plots()

    # ------------------------------------------------------------------
    def _rebuild_plots(self) -> None:
        # Remove old plots
        for pw in self._plots:
            self._plot_layout.removeWidget(pw)
            pw.deleteLater()
        self._plots.clear()
        self._curves.clear()
        self._region = None

        if self._overlay_mode:
            self._build_overlay()
        else:
            self._build_stacked()

    def _build_overlay(self) -> None:
        pw = pg.PlotWidget()
        pw.setBackground("#1e1e2e")
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setLabel("bottom", "Time", units="s")
        pw.addLegend(offset=(10, 10))
        pw.getPlotItem().setMouseEnabled(x=True, y=True)

        for i, ch in enumerate(self.channels):
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            curve = pw.plot(
                pen=pg.mkPen(color=QColor(color), width=1.5),
                name=ch.name,
            )
            self._curves[ch.id] = curve

        self._add_region(pw)
        self._plot_layout.addWidget(pw)
        self._plots.append(pw)

    def _build_stacked(self) -> None:
        first_pw: pg.PlotWidget | None = None

        for i, ch in enumerate(self.channels):
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            pw = pg.PlotWidget()
            pw.setBackground("#1e1e2e")
            pw.showGrid(x=True, y=True, alpha=0.2)
            pw.setLabel("left", ch.name, units=ch.unit, color=color)
            pw.setMaximumHeight(200)
            pw.setMinimumHeight(80)
            pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            pw.getPlotItem().setMouseEnabled(x=True, y=True)

            # Link X axis to first plot
            if first_pw is None:
                first_pw = pw
            else:
                pw.setXLink(first_pw)

            curve = pw.plot(pen=pg.mkPen(color=QColor(color), width=1.5))
            self._curves[ch.id] = curve

            if i == 0:
                self._add_region(pw)

            self._plot_layout.addWidget(pw)
            self._plots.append(pw)

    def _add_region(self, pw: pg.PlotWidget) -> None:
        if not self._region_cb.isChecked():
            return
        region = pg.LinearRegionItem(
            values=[0.2, 0.8],
            brush=pg.mkBrush("#89dceb22"),
            pen=pg.mkPen("#89dceb", width=1),
            movable=True,
        )
        region.sigRegionChanged.connect(self._on_region_changed)
        pw.addItem(region)
        self._region = region

    # ------------------------------------------------------------------
    def update_plots(self) -> None:
        for ch in self.channels:
            curve = self._curves.get(ch.id)
            if curve is None or not ch.enabled:
                continue
            ts, data = ch.get_data()
            if len(ts) < 2:
                continue
            ts_d, data_d = auto_decimate(ts, data, max_points=8000)
            curve.setData(ts_d, data_d)

        if self._region:
            self._on_region_changed(self._region)

    # ------------------------------------------------------------------
    def _on_region_changed(self, region: pg.LinearRegionItem) -> None:
        lo, hi = region.getRegion()
        self._sel_lbl.setText(f"A = {lo:.4f} s   B = {hi:.4f} s   Δ = {hi - lo:.4f} s")
        self.range_changed.emit(lo, hi)

    def _toggle_region(self, enabled: bool) -> None:
        if self._region:
            self._region.setVisible(enabled)
        if not enabled:
            self._sel_lbl.setText("Region hidden — showing full-range statistics")
            self.range_changed.emit(0.0, 0.0)

    def _on_overlay_toggle(self, checked: bool) -> None:
        self._overlay_mode = checked
        self._rebuild_plots()
        self.update_plots()

    # ------------------------------------------------------------------
    def _zoom_in(self) -> None:
        for pw in self._plots:
            vr = pw.viewRect()
            cx = (vr.left() + vr.right()) / 2
            hw = (vr.right() - vr.left()) / 4
            pw.setXRange(cx - hw, cx + hw, padding=0)

    def _zoom_out(self) -> None:
        for pw in self._plots:
            vr = pw.viewRect()
            cx = (vr.left() + vr.right()) / 2
            hw = (vr.right() - vr.left())
            pw.setXRange(cx - hw, cx + hw, padding=0)

    def _zoom_fit(self) -> None:
        for pw in self._plots:
            pw.enableAutoRange()

    # ------------------------------------------------------------------
    def _selection_range(self) -> tuple[float, float] | None:
        if self._region and self._region_cb.isChecked():
            return self._region.getRegion()
        return None

    def _slice_channels(self) -> list[Channel]:
        """Return temporary Channel copies containing only the selected range."""
        sel = self._selection_range()
        result = []
        for ch in self.channels:
            ts, data = ch.get_data()
            if len(data) == 0:
                continue
            if sel:
                lo, hi = sel
                mask = (ts >= lo) & (ts <= hi)
                ts, data = ts[mask], data[mask]
            if len(data) == 0:
                continue
            tmp = Channel(id=ch.id, name=ch.name, unit=ch.unit,
                          sample_rate=ch.sample_rate, color=ch.color,
                          _buffer_size=len(data) + 1)
            tmp.push(data, ts)
            result.append(tmp)
        return result

    def _export_hdf5(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export selection (HDF5)",
            str(Path.home() / "selection.h5"), "HDF5 (*.h5 *.hdf5)"
        )
        if path:
            save_hdf5(self._slice_channels(), path)

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export selection (CSV)",
            str(Path.home() / "selection.csv"), "CSV (*.csv)"
        )
        if path:
            save_csv(self._slice_channels(), path)

    # ------------------------------------------------------------------
    def set_channels(self, channels: List[Channel]) -> None:
        self.channels = channels
        self._rebuild_plots()
