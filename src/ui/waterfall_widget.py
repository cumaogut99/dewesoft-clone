"""
Waterfall / Spectrogram widget — rolling 2D FFT display.
Each horizontal row = one FFT snapshot in time; newest row at the bottom.
Color encodes magnitude (dB scale, configurable colormap).
"""
from __future__ import annotations
from typing import List
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

from ..core.channel import Channel
from ..core.fft_analyzer import compute_fft, WindowType

COLORMAPS = ["CET-L9", "viridis", "plasma", "inferno", "CET-R4", "thermal"]


class WaterfallWidget(QWidget):
    def __init__(self, channels: List[Channel], parent=None) -> None:
        super().__init__(parent)
        self.channels = channels
        self._n_rows = 100
        self._fft_size = 512
        self._img: np.ndarray | None = None
        self._row_ptr = 0
        self._build_ui()
        self._init_buffer()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Channel:"))
        self._ch_combo = QComboBox()
        self._ch_combo.addItems([ch.name for ch in self.channels])
        self._ch_combo.currentIndexChanged.connect(lambda _: self._init_buffer())
        toolbar.addWidget(self._ch_combo)

        toolbar.addWidget(QLabel("  Window:"))
        self._win_combo = QComboBox()
        self._win_combo.addItems(["hann", "hamming", "blackman", "flattop"])
        toolbar.addWidget(self._win_combo)

        toolbar.addWidget(QLabel("  FFT size:"))
        self._fft_combo = QComboBox()
        self._fft_combo.addItems(["128", "256", "512", "1024", "2048"])
        self._fft_combo.setCurrentText("512")
        self._fft_combo.currentTextChanged.connect(self._on_fft_size_changed)
        toolbar.addWidget(self._fft_combo)

        toolbar.addWidget(QLabel("  Rows:"))
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(20, 500)
        self._rows_spin.setValue(100)
        self._rows_spin.editingFinished.connect(self._on_rows_changed)
        toolbar.addWidget(self._rows_spin)

        toolbar.addWidget(QLabel("  Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(COLORMAPS)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        toolbar.addWidget(self._cmap_combo)

        toolbar.addWidget(QLabel("  Min dB:"))
        self._min_db = QDoubleSpinBox()
        self._min_db.setRange(-120, 0)
        self._min_db.setValue(-80)
        toolbar.addWidget(self._min_db)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # pyqtgraph image plot
        self._view = pg.GraphicsLayoutWidget()
        self._view.setBackground("#1e1e2e")
        self._plt = self._view.addPlot()
        self._plt.setLabel("left", "Time (rows)")
        self._plt.setLabel("bottom", "Frequency", units="Hz")

        self._img_item = pg.ImageItem()
        self._plt.addItem(self._img_item)

        # Colorbar
        self._bar = pg.ColorBarItem(
            values=(-80, 0),
            colorMap=pg.colormap.get("CET-L9"),
            label="dB",
        )
        self._bar.setImageItem(self._img_item, insert_in=self._plt)

        layout.addWidget(self._view)

    def _init_buffer(self) -> None:
        n_freq = self._fft_size // 2 + 1
        self._img = np.full((self._n_rows, n_freq), -80.0)
        self._row_ptr = 0

    def _on_fft_size_changed(self, text: str) -> None:
        self._fft_size = int(text)
        self._init_buffer()

    def _on_rows_changed(self) -> None:
        self._n_rows = self._rows_spin.value()
        self._init_buffer()

    def _on_cmap_changed(self, name: str) -> None:
        try:
            cmap = pg.colormap.get(name)
            self._bar.setColorMap(cmap)
        except Exception:
            pass

    def update_plots(self) -> None:
        if self._img is None:
            return
        idx = self._ch_combo.currentIndex()
        if idx < 0 or idx >= len(self.channels):
            return

        ch = self.channels[idx]
        _, data = ch.get_data(n_samples=self._fft_size)
        if len(data) < self._fft_size // 2:
            return

        window = self._win_combo.currentText()
        freqs, mag = compute_fft(
            data[-self._fft_size:], ch.sample_rate,
            window=window, db_scale=True
        )
        if len(mag) == 0:
            return

        n_freq = self._img.shape[1]
        row = np.interp(
            np.linspace(0, len(mag) - 1, n_freq),
            np.arange(len(mag)),
            mag
        )

        # Scroll: shift buffer up one row and insert new row at bottom
        self._img[:-1] = self._img[1:]
        self._img[-1] = row

        min_db = self._min_db.value()
        display = np.clip(self._img, min_db, 0.0)

        # Scale for ImageItem: x=freq, y=time row
        self._img_item.setImage(
            display.T,
            autoLevels=False,
            levels=(min_db, 0.0),
        )
        # Set frequency axis scale
        sr = ch.sample_rate
        self._img_item.resetTransform()
        freq_scale = (sr / 2.0) / n_freq
        self._img_item.scale(freq_scale, 1.0)

    def set_channels(self, channels: List[Channel]) -> None:
        self.channels = channels
        self._ch_combo.clear()
        self._ch_combo.addItems([ch.name for ch in channels])
        self._init_buffer()
