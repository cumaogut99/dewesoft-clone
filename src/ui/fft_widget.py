"""
FFT spectrum analyzer widget.

Features:
  - Magnitude spectrum with configurable window and FFT size
  - Peak detection table
  - THD / THD+N / SNR / SINAD / SFDR panel (per channel)
  - Visual harmonic markers on the spectrum (F1, 2F, 3F, …)
  - dB / linear scale toggle, log-X frequency axis option
"""
from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QSplitter,
    QGroupBox, QGridLayout, QSpinBox, QTabWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from ..core.channel import Channel
from ..core.fft_analyzer import (
    compute_fft, find_spectral_peaks, compute_thd,
    THDResult, WindowType,
)

CHANNEL_COLORS = [
    "#00ff88", "#ff6b6b", "#4ecdc4", "#ffe66d",
    "#a8e6cf", "#ff8b94", "#b8b8ff", "#ffd3b6",
]

_METRIC_ROWS = [
    ("Fundamental",  "fundamental_freq",      "{:.2f} Hz"),
    ("V₁ (peak)",    "fundamental_amplitude", "{:.4f} V"),
    ("THD",          "thd_percent",           "{:.3f} %"),
    ("THD (dB)",     "thd_db",                "{:.2f} dB"),
    ("THD+N",        "thd_n_percent",         "{:.3f} %"),
    ("THD+N (dB)",   "thd_n_db",              "{:.2f} dB"),
    ("SNR",          "snr_db",                "{:.1f} dB"),
    ("SINAD",        "sinad_db",              "{:.1f} dB"),
    ("SFDR",         "sfdr_dbc",              "{:.1f} dBc"),
    ("# harmonics",  "n_harmonics",           "{:d}"),
]


class FFTWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._harmonic_lines: dict[int, list[pg.InfiniteLine]] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- Toolbar ----
        tb = QHBoxLayout()

        tb.addWidget(QLabel("Window:"))
        self._win_combo = QComboBox()
        self._win_combo.addItems(["hann", "hamming", "blackman", "flattop", "rectangular"])
        tb.addWidget(self._win_combo)

        tb.addWidget(QLabel("  FFT size:"))
        self._fft_size = QComboBox()
        self._fft_size.addItems(["256", "512", "1024", "2048", "4096", "8192"])
        self._fft_size.setCurrentText("1024")
        tb.addWidget(self._fft_size)

        self._peaks_cb = QCheckBox("Peaks")
        self._peaks_cb.setChecked(True)
        tb.addWidget(self._peaks_cb)

        self._harmonics_cb = QCheckBox("Harmonic markers")
        self._harmonics_cb.setChecked(True)
        tb.addWidget(self._harmonics_cb)

        self._db_cb = QCheckBox("dB scale")
        self._db_cb.setChecked(True)
        tb.addWidget(self._db_cb)

        self._logx_cb = QCheckBox("Log freq")
        self._logx_cb.toggled.connect(
            lambda v: self._plot.setLogMode(x=v, y=False)
        )
        tb.addWidget(self._logx_cb)

        tb.addStretch()
        root.addLayout(tb)

        # ---- Main splitter ----
        outer = QSplitter(Qt.Orientation.Horizontal)

        # Left: spectrum + peaks
        left_w = QWidget()
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(0, 0, 0, 0)

        vsplit = QSplitter(Qt.Orientation.Vertical)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#1e1e2e")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left", "Magnitude", units="dBV")
        self._plot.addLegend(offset=(10, 10))
        vsplit.addWidget(self._plot)

        self._peak_table = QTableWidget(0, 3)
        self._peak_table.setHorizontalHeaderLabels(["Channel", "Freq (Hz)", "Mag (dB)"])
        self._peak_table.setMaximumHeight(120)
        self._peak_table.setFont(QFont("Monospace", 9))
        vsplit.addWidget(self._peak_table)

        left_lay.addWidget(vsplit)
        outer.addWidget(left_w)

        # Right: THD / metrics panel
        right_w = QWidget()
        right_w.setMaximumWidth(320)
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(4, 0, 0, 0)

        right_lay.addWidget(QLabel("Distortion / Noise Metrics"))

        self._thd_ch_combo = QComboBox()
        for ch in self.channels:
            self._thd_ch_combo.addItem(ch.name, userData=ch.id)
        right_lay.addWidget(self._thd_ch_combo)

        self._n_harm_spin = QSpinBox()
        self._n_harm_spin.setRange(1, 20)
        self._n_harm_spin.setValue(9)
        self._n_harm_spin.setPrefix("Harmonics: ")
        right_lay.addWidget(self._n_harm_spin)

        # Metrics grid
        metrics_box = QGroupBox("Metrics")
        grid = QGridLayout(metrics_box)
        grid.setSpacing(3)
        self._metric_labels: dict[str, QLabel] = {}
        for row, (label, attr, _) in enumerate(_METRIC_ROWS):
            lbl_key = QLabel(label + ":")
            lbl_key.setStyleSheet("color: #7f849c; font-size: 10px;")
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet("color: #cdd6f4; font-family: monospace; font-size: 10px;")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(lbl_key, row, 0)
            grid.addWidget(lbl_val, row, 1)
            self._metric_labels[attr] = lbl_val
        right_lay.addWidget(metrics_box)

        # Harmonic table
        harm_box = QGroupBox("Harmonics")
        harm_lay = QVBoxLayout(harm_box)
        self._harm_table = QTableWidget(0, 3)
        self._harm_table.setHorizontalHeaderLabels(["Order", "Freq (Hz)", "Amp (dBV)"])
        self._harm_table.setFont(QFont("Monospace", 9))
        harm_lay.addWidget(self._harm_table)
        right_lay.addWidget(harm_box)
        right_lay.addStretch()

        outer.addWidget(right_w)
        outer.setStretchFactor(0, 3)
        outer.setStretchFactor(1, 1)
        root.addWidget(outer)

        # Build spectrum curves
        for i, ch in enumerate(self.channels):
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            pen = pg.mkPen(color=QColor(color), width=1.5)
            curve = self._plot.plot(pen=pen, name=ch.name)
            self._curves[ch.id] = curve
            self._harmonic_lines[ch.id] = []

    # ------------------------------------------------------------------
    def update_plots(self) -> None:
        window = self._win_combo.currentText()
        fft_n = int(self._fft_size.currentText())
        db_scale = self._db_cb.isChecked()
        show_peaks = self._peaks_cb.isChecked()
        show_harmonics = self._harmonics_cb.isChecked()

        self._peak_table.setRowCount(0)
        self._clear_harmonic_markers()

        for ch in self.channels:
            curve = self._curves.get(ch.id)
            if curve is None:
                continue
            if not ch.enabled:
                curve.setVisible(False)
                continue

            _, data = ch.get_data(n_samples=fft_n)
            if len(data) < 16:
                continue

            block = data[-fft_n:]
            freqs, mag = compute_fft(block, ch.sample_rate, window=window, db_scale=db_scale)
            curve.setData(freqs, mag)
            curve.setVisible(True)
            self._plot.setLabel("left", "Magnitude", units="dBV" if db_scale else "V")

            if show_peaks:
                peaks = find_spectral_peaks(freqs, mag, n_peaks=3)
                for freq, amp in peaks:
                    r = self._peak_table.rowCount()
                    self._peak_table.insertRow(r)
                    self._peak_table.setItem(r, 0, QTableWidgetItem(ch.name))
                    self._peak_table.setItem(r, 1, QTableWidgetItem(f"{freq:.2f}"))
                    self._peak_table.setItem(r, 2, QTableWidgetItem(f"{amp:.1f}"))

        # THD analysis for the selected channel
        thd_idx = self._thd_ch_combo.currentIndex()
        if 0 <= thd_idx < len(self.channels):
            self._run_thd(self.channels[thd_idx], fft_n, window, show_harmonics)

    # ------------------------------------------------------------------
    def _run_thd(
        self,
        ch: Channel,
        fft_n: int,
        window: str,
        show_markers: bool,
    ) -> None:
        _, data = ch.get_data(n_samples=fft_n)
        if len(data) < 64:
            return

        result = compute_thd(
            data[-fft_n:],
            ch.sample_rate,
            window=window,           # type: ignore[arg-type]
            n_harmonics=self._n_harm_spin.value(),
        )

        # Update metric labels
        for _, attr, fmt in _METRIC_ROWS:
            val = getattr(result, attr)
            try:
                text = fmt.format(val)
            except (TypeError, ValueError):
                text = str(val)
            lbl = self._metric_labels[attr]
            lbl.setText(text)

            # Colour THD red if high
            if attr == "thd_percent":
                if val > 10.0:
                    lbl.setStyleSheet("color: #f38ba8; font-family: monospace; font-size: 10px;")
                elif val > 1.0:
                    lbl.setStyleSheet("color: #f9e2af; font-family: monospace; font-size: 10px;")
                else:
                    lbl.setStyleSheet("color: #a6e3a1; font-family: monospace; font-size: 10px;")

        # Harmonic table
        self._harm_table.setRowCount(0)
        f0 = result.fundamental_freq
        if f0 > 0:
            # Fundamental row
            self._harm_table.insertRow(0)
            self._harm_table.setItem(0, 0, QTableWidgetItem("F₁"))
            self._harm_table.setItem(0, 1, QTableWidgetItem(f"{f0:.2f}"))
            v1_db = 20.0 * np.log10(max(result.fundamental_amplitude, 1e-12))
            self._harm_table.setItem(0, 2, QTableWidgetItem(f"{v1_db:.1f}"))

            for k, (hf, ha) in enumerate(
                zip(result.harmonic_freqs, result.harmonic_amplitudes), start=2
            ):
                r = self._harm_table.rowCount()
                self._harm_table.insertRow(r)
                self._harm_table.setItem(r, 0, QTableWidgetItem(f"{k}F"))
                self._harm_table.setItem(r, 1, QTableWidgetItem(f"{hf:.2f}"))
                ha_db = 20.0 * np.log10(max(ha, 1e-12))
                self._harm_table.setItem(r, 2, QTableWidgetItem(f"{ha_db:.1f}"))

        # Harmonic marker lines on spectrum
        if show_markers and result.fundamental_freq > 0:
            color = CHANNEL_COLORS[
                next((i for i, c in enumerate(self.channels) if c.id == ch.id), 0)
                % len(CHANNEL_COLORS)
            ]
            lines = self._harmonic_lines.setdefault(ch.id, [])
            freqs_to_mark = (
                [result.fundamental_freq] + result.harmonic_freqs
            )
            labels = ["F₁"] + [f"{k+2}F" for k in range(len(result.harmonic_freqs))]
            for freq, label in zip(freqs_to_mark, labels):
                line = pg.InfiniteLine(
                    pos=freq, angle=90,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine),
                    label=label,
                    labelOpts={"color": color, "position": 0.95,
                               "rotateAxis": (1, 0), "fill": "#1e1e2e88"},
                    movable=False,
                )
                self._plot.addItem(line)
                lines.append(line)

    def _clear_harmonic_markers(self) -> None:
        for lines in self._harmonic_lines.values():
            for line in lines:
                self._plot.removeItem(line)
            lines.clear()

    # ------------------------------------------------------------------
    def set_channels(self, channels: List[Channel]) -> None:
        self.channels = channels
        self._plot.clear()
        self._curves.clear()
        self._harmonic_lines.clear()
        self._thd_ch_combo.clear()
        for i, ch in enumerate(channels):
            color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            pen = pg.mkPen(color=QColor(color), width=1.5)
            curve = self._plot.plot(pen=pen, name=ch.name)
            self._curves[ch.id] = curve
            self._harmonic_lines[ch.id] = []
            self._thd_ch_combo.addItem(ch.name, userData=ch.id)
