"""
Main application window — DeweSoft-style tabbed measurement environment.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QStatusBar, QTabWidget, QFileDialog,
    QMessageBox, QLabel, QSplitter, QDockWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QColor, QPalette, QFont

from ..core.channel import Channel
from ..core.signal_generator import SignalGenerator, WaveformType
from ..core.acquisition_engine import AcquisitionEngine
from ..io.data_writer import save_hdf5, save_csv

from .oscilloscope_widget import OscilloscopeWidget
from .fft_widget import FFTWidget
from .recorder_widget import RecorderWidget
from .digital_meter_widget import DigitalMeterWidget
from .channel_setup_widget import ChannelSetupWidget


CHANNEL_DEFAULTS = [
    ("CH1", "V",   10.0, 1.0, WaveformType.SINE,     "#00ff88"),
    ("CH2", "V",   25.0, 0.7, WaveformType.SINE,     "#ff6b6b"),
    ("CH3", "m/s²", 5.0, 2.0, WaveformType.CHIRP,    "#4ecdc4"),
    ("CH4", "°C",   1.0, 0.5, WaveformType.TRIANGLE, "#ffe66d"),
]


def _apply_dark_theme(app) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(30, 30, 46))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(205, 214, 244))
    palette.setColor(QPalette.ColorRole.Base,            QColor(24, 24, 37))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(30, 30, 46))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(24, 24, 37))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(205, 214, 244))
    palette.setColor(QPalette.ColorRole.Text,            QColor(205, 214, 244))
    palette.setColor(QPalette.ColorRole.Button,          QColor(49, 50, 68))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(205, 214, 244))
    palette.setColor(QPalette.ColorRole.BrightText,      QColor(243, 139, 168))
    palette.setColor(QPalette.ColorRole.Link,            QColor(137, 180, 250))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(137, 180, 250))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(24, 24, 37))
    app.setPalette(palette)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeweSoft Clone — Python DAQ Studio")
        self.resize(1400, 900)

        self._recording = False
        self._record_path: Path | None = None

        self._init_channels()
        self._init_engine()
        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._connect_engine()

        # UI refresh timer (independent of acquisition rate)
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(50)  # 20 FPS
        self._ui_timer.timeout.connect(self._refresh_ui)
        self._ui_timer.start()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_channels(self) -> None:
        self.channels: List[Channel] = []
        self.generators: List[SignalGenerator] = []

        for i, (name, unit, freq, amp, wave, color) in enumerate(CHANNEL_DEFAULTS):
            ch = Channel(id=i + 1, name=name, unit=unit, sample_rate=1000.0, color=color)
            gen = SignalGenerator(waveform=wave, frequency=freq, amplitude=amp)
            self.channels.append(ch)
            self.generators.append(gen)

    def _init_engine(self) -> None:
        self._engine = AcquisitionEngine(
            channels=self.channels,
            generators=self.generators,
            sample_rate=1000.0,
            update_interval_ms=50,
            parent=self,
        )

    def _connect_engine(self) -> None:
        self._engine.status_changed.connect(self._on_engine_status)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # --- Oscilloscope tab
        self._scope = OscilloscopeWidget(self.channels)
        self._tabs.addTab(self._scope, "Oscilloscope")

        # --- FFT tab
        self._fft = FFTWidget(self.channels)
        self._tabs.addTab(self._fft, "FFT Analyzer")

        # --- Recorder tab
        self._recorder = RecorderWidget(self.channels)
        self._tabs.addTab(self._recorder, "Recorder")

        # --- Digital meters tab
        self._meters = DigitalMeterWidget(self.channels)
        self._tabs.addTab(self._meters, "Digital Meters")

        # --- Channel setup tab
        self._ch_setup = ChannelSetupWidget(self.channels, self.generators)
        self._ch_setup.channels_changed.connect(self._on_channels_changed)
        self._tabs.addTab(self._ch_setup, "Channel Setup")

        root.addWidget(self._tabs)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        act_save_hdf5 = QAction("Save as HDF5…", self)
        act_save_hdf5.setShortcut(QKeySequence.StandardKey.Save)
        act_save_hdf5.triggered.connect(self._save_hdf5)
        file_menu.addAction(act_save_hdf5)

        act_save_csv = QAction("Export as CSV…", self)
        act_save_csv.triggered.connect(self._save_csv)
        file_menu.addAction(act_save_csv)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Acquisition
        acq_menu = mb.addMenu("&Acquisition")
        self._act_start = QAction("▶  Start", self)
        self._act_start.setShortcut("F5")
        self._act_start.triggered.connect(self._start_acquisition)
        acq_menu.addAction(self._act_start)

        self._act_stop = QAction("■  Stop", self)
        self._act_stop.setShortcut("F6")
        self._act_stop.triggered.connect(self._stop_acquisition)
        self._act_stop.setEnabled(False)
        acq_menu.addAction(self._act_stop)

        acq_menu.addSeparator()
        act_reset = QAction("Reset buffers", self)
        act_reset.triggered.connect(self._reset)
        acq_menu.addAction(act_reset)

        # Help
        help_menu = mb.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._tb_start = QAction("▶  Start", self)
        self._tb_start.triggered.connect(self._start_acquisition)
        tb.addAction(self._tb_start)

        self._tb_stop = QAction("■  Stop", self)
        self._tb_stop.triggered.connect(self._stop_acquisition)
        self._tb_stop.setEnabled(False)
        tb.addAction(self._tb_stop)

        tb.addSeparator()

        self._tb_save_hdf5 = QAction("💾 Save HDF5", self)
        self._tb_save_hdf5.triggered.connect(self._save_hdf5)
        tb.addAction(self._tb_save_hdf5)

        self._tb_save_csv = QAction("📄 Export CSV", self)
        self._tb_save_csv.triggered.connect(self._save_csv)
        tb.addAction(self._tb_save_csv)

        tb.addSeparator()
        tb.addWidget(QLabel("  Sample rate: 1000 Hz  |  Channels: 4  "))

    def _build_statusbar(self) -> None:
        self._status_lbl = QLabel("Ready")
        self._time_lbl = QLabel("T = 0.000 s")
        sb = self.statusBar()
        sb.addWidget(self._status_lbl, 1)
        sb.addPermanentWidget(self._time_lbl)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    @pyqtSlot()
    def _start_acquisition(self) -> None:
        if self._engine.isRunning():
            return
        self._engine.start_acquisition()
        self._act_start.setEnabled(False)
        self._act_stop.setEnabled(True)
        self._tb_start.setEnabled(False)
        self._tb_stop.setEnabled(True)

    @pyqtSlot()
    def _stop_acquisition(self) -> None:
        self._engine.stop_acquisition()
        self._act_start.setEnabled(True)
        self._act_stop.setEnabled(False)
        self._tb_start.setEnabled(True)
        self._tb_stop.setEnabled(False)

    @pyqtSlot()
    def _reset(self) -> None:
        was_running = self._engine.isRunning()
        if was_running:
            self._stop_acquisition()
        self._engine.reset()
        if was_running:
            self._start_acquisition()

    @pyqtSlot(str)
    def _on_engine_status(self, status: str) -> None:
        self._status_lbl.setText(f"Acquisition: {status}")

    @pyqtSlot()
    def _refresh_ui(self) -> None:
        tab = self._tabs.currentIndex()
        if tab == 0:
            self._scope.update_plots()
        elif tab == 1:
            self._fft.update_plots()
        elif tab == 2:
            self._recorder.update_plots()
        elif tab == 3:
            self._meters.update_plots()

        self._time_lbl.setText(f"T = {self._engine.elapsed_time:.3f} s")

    @pyqtSlot()
    def _on_channels_changed(self) -> None:
        self._scope.set_channels(self.channels)
        self._meters.set_channels(self.channels)

    @pyqtSlot()
    def _save_hdf5(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5", str(Path.home() / "recording.h5"),
            "HDF5 files (*.h5 *.hdf5)"
        )
        if path:
            try:
                save_hdf5(self.channels, path)
                self._status_lbl.setText(f"Saved: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Save error", str(exc))

    @pyqtSlot()
    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(Path.home() / "recording.csv"),
            "CSV files (*.csv)"
        )
        if path:
            try:
                save_csv(self.channels, path)
                self._status_lbl.setText(f"Exported: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export error", str(exc))

    @pyqtSlot()
    def _about(self) -> None:
        QMessageBox.about(
            self, "About DeweSoft Clone",
            "<h3>DeweSoft Clone — Python DAQ Studio</h3>"
            "<p>Personal learning project — not for commercial use.</p>"
            "<p><b>Stack:</b> PyQt6 · pyqtgraph · NumPy · SciPy · h5py</p>"
            "<p>Simulate, visualize, and analyze time-series signals.</p>",
        )

    def closeEvent(self, event) -> None:
        if self._engine.isRunning():
            self._engine.stop_acquisition()
        event.accept()
