"""
Playback controls bar — embedded in the main window when playback mode is active.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton,
    QSlider, QLabel, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..core.playback_engine import PlaybackEngine


class PlaybackBar(QWidget):
    """Compact transport controls for the playback engine."""

    load_requested = pyqtSignal()

    def __init__(self, engine: PlaybackEngine, parent=None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._duration = 0.0
        self._dragging = False
        self._build_ui()
        engine.progress.connect(self._on_progress)
        engine.status_changed.connect(self._on_status)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._open_btn = QPushButton("📂 Open…")
        self._open_btn.clicked.connect(self.load_requested)
        layout.addWidget(self._open_btn)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)

        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedWidth(36)
        self._stop_btn.clicked.connect(self._stop)
        layout.addWidget(self._stop_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(0)
        self._slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
        self._slider.sliderReleased.connect(self._on_seek)
        layout.addWidget(self._slider, stretch=1)

        self._time_lbl = QLabel("0:00 / 0:00")
        layout.addWidget(self._time_lbl)

        layout.addWidget(QLabel("  Speed:"))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.1, 32.0)
        self._speed_spin.setValue(1.0)
        self._speed_spin.setSuffix("×")
        self._speed_spin.setSingleStep(0.5)
        self._speed_spin.valueChanged.connect(self._engine.set_speed)
        layout.addWidget(self._speed_spin)

    def set_duration(self, seconds: float) -> None:
        self._duration = seconds
        self._update_time_label(0.0)

    def _toggle_play(self) -> None:
        if self._engine.isRunning():
            self._engine.pause()
        else:
            self._engine.start_playback()

    def _stop(self) -> None:
        self._engine.stop_playback()
        self._slider.setValue(0)
        self._play_btn.setText("▶")

    def _on_seek(self) -> None:
        self._dragging = False
        frac = self._slider.value() / 1000.0
        self._engine.seek(frac)

    def _on_progress(self, frac: float) -> None:
        if not self._dragging:
            self._slider.setValue(int(frac * 1000))
        self._update_time_label(frac * self._duration)

    def _on_status(self, status: str) -> None:
        if status == "playing":
            self._play_btn.setText("⏸")
        elif status in ("paused", "stopped", "finished"):
            self._play_btn.setText("▶")

    def _update_time_label(self, elapsed: float) -> None:
        def fmt(s):
            return f"{int(s)//60}:{int(s)%60:02d}"
        self._time_lbl.setText(f"{fmt(elapsed)} / {fmt(self._duration)}")
