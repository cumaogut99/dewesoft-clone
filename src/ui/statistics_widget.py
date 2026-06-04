"""
Statistics table widget — shows per-channel descriptive statistics.
Updates live in Measure mode; updates on selection change in Analyze mode.
"""
from __future__ import annotations
from typing import List
import csv
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from ..core.channel import Channel
from ..core.statistics import compute_stats_all, ChannelStats

_COLS = ["Channel", "Unit", "N", "Mean", "Std", "RMS", "Min", "Max", "Pk-Pk", "Crest", "Duration"]
_FMT  = [None,      None,   "d", ".4f", ".4f", ".4f", ".4f",".4f", ".4f",  ".3f",  ".3f"]


class StatisticsWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._t_start: float | None = None
        self._t_end:   float | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        hdr = QHBoxLayout()
        self._range_lbl = QLabel("Range: all data")
        self._range_lbl.setStyleSheet("color: #cba6f7; font-style: italic;")
        hdr.addWidget(self._range_lbl)
        hdr.addStretch()

        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.clicked.connect(self._export)
        hdr.addWidget(self._export_btn)
        layout.addLayout(hdr)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setFont(QFont("Monospace", 9))
        self._table.setStyleSheet(
            "QTableWidget { gridline-color: #313244; }"
            "QHeaderView::section { background-color: #313244; }"
        )
        layout.addWidget(self._table)

    # ------------------------------------------------------------------
    def set_range(self, t_start: float | None, t_end: float | None) -> None:
        self._t_start = t_start
        self._t_end   = t_end
        if t_start is not None and t_end is not None:
            self._range_lbl.setText(
                f"Range: {t_start:.3f} s → {t_end:.3f} s  (Δ {t_end - t_start:.3f} s)"
            )
        else:
            self._range_lbl.setText("Range: all data")

    def update_plots(self) -> None:
        stats = compute_stats_all(self.channels, self._t_start, self._t_end)
        self._table.setRowCount(len(stats))

        for row, s in enumerate(stats):
            values = [
                s.name, s.unit, s.n_samples,
                s.mean, s.std, s.rms,
                s.minimum, s.maximum, s.peak_to_peak,
                s.crest_factor, s.duration,
            ]
            for col, (val, fmt) in enumerate(zip(values, _FMT)):
                if fmt is None:
                    text = str(val)
                elif fmt == "d":
                    text = f"{int(val):,}"
                else:
                    text = format(float(val), fmt)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row, col, item)

    def set_channels(self, channels: List[Channel]) -> None:
        self.channels = channels

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export statistics", str(Path.home() / "statistics.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        stats = compute_stats_all(self.channels, self._t_start, self._t_end)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(_COLS)
            for s in stats:
                w.writerow([
                    s.name, s.unit, s.n_samples,
                    f"{s.mean:.6f}", f"{s.std:.6f}", f"{s.rms:.6f}",
                    f"{s.minimum:.6f}", f"{s.maximum:.6f}",
                    f"{s.peak_to_peak:.6f}", f"{s.crest_factor:.4f}",
                    f"{s.duration:.4f}",
                ])
