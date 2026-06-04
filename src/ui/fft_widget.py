"""
FFT spectrum analyzer widget.
Shows magnitude spectrum, peak markers, and window selector.
"""
from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QCheckBox, QTableWidget, QTableWidgetItem, QSplitter
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from ..core.channel import Channel
from ..core.fft_analyzer import compute_fft, find_spectral_peaks, WindowType


CHANNEL_COLORS = [
    "#00ff88", "#ff6b6b", "#4ecdc4", "#ffe66d",
    "#a8e6cf", "#ff8b94", "#b8b8ff", "#ffd3b6",
]


class FFTWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._peak_markers: dict[int, list] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Window:"))
        self._win_combo = QComboBox()
        self._win_combo.addItems(["hann", "hamming", "blackman", "flattop", "rectangular"])
        toolbar.addWidget(self._win_combo)

        toolbar.addWidget(QLabel("FFT size:"))
        self._fft_size = QComboBox()
        self._fft_size.addItems(["256", "512", "1024", "2048", "4096"])
        self._fft_size.setCurrentText("1024")
        toolbar.addWidget(self._fft_size)

        self._peaks_cb = QCheckBox("Show peaks")
        self._peaks_cb.setChecked(True)
        toolbar.addWidget(self._peaks_cb)

        self._db_cb = QCheckBox("dB scale")
        self._db_cb.setChecked(True)
        toolbar.addWidget(self._db_cb)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Splitter: plot on top, peak table on bottom
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1e1e2e")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left", "Magnitude", units="dBV")
        self._plot.setLogMode(x=False, y=False)
        self._plot.addLegend(offset=(10, 10))
        splitter.addWidget(self._plot)

        # Peak table
        self._peak_table = QTableWidget(0, 3)
        self._peak_table.setHorizontalHeaderLabels(["Channel", "Frequency (Hz)", "Magnitude (dB)"])
        self._peak_table.setMaximumHeight(140)
        self._peak_table.setFont(QFont("Monospace", 9))
        splitter.addWidget(self._peak_table)

        layout.addWidget(splitter)

        for i, ch in enumerate(self.channels):
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            pen = pg.mkPen(color=QColor(color), width=1.5)
            curve = self._plot.plot(pen=pen, name=ch.name)
            self._curves[ch.id] = curve

    # ------------------------------------------------------------------
    def update_plots(self) -> None:
        window = self._win_combo.currentText()
        fft_n = int(self._fft_size.currentText())
        db_scale = self._db_cb.isChecked()
        show_peaks = self._peaks_cb.isChecked()

        self._peak_table.setRowCount(0)

        for ch in self.channels:
            if not ch.enabled:
                self._curves[ch.id].setVisible(False)
                continue

            _, data = ch.get_data(n_samples=fft_n)
            if len(data) < 8:
                continue

            freqs, mag = compute_fft(data[-fft_n:], ch.sample_rate, window=window, db_scale=db_scale)
            self._curves[ch.id].setData(freqs, mag)
            self._curves[ch.id].setVisible(True)

            self._plot.setLabel("left", "Magnitude", units="dBV" if db_scale else "V")

            if show_peaks:
                peaks = find_spectral_peaks(freqs, mag, n_peaks=3)
                for freq, amp in peaks:
                    row = self._peak_table.rowCount()
                    self._peak_table.insertRow(row)
                    self._peak_table.setItem(row, 0, QTableWidgetItem(ch.name))
                    self._peak_table.setItem(row, 1, QTableWidgetItem(f"{freq:.2f}"))
                    self._peak_table.setItem(row, 2, QTableWidgetItem(f"{amp:.1f}"))
