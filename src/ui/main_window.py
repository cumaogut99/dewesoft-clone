"""
Main application window — DeweSoft-style tabbed measurement environment.
"""
from __future__ import annotations
import importlib
from pathlib import Path
from typing import List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QToolBar,
    QFileDialog, QMessageBox, QLabel
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence, QColor, QPalette, QTabWidget

from ..core.channel import Channel
from ..core.signal_generator import SignalGenerator, WaveformType
from ..core.acquisition_engine import ThreadedAcquisitionEngine, MultiprocessAcquisitionEngine
from ..hardware.simulated_driver import SimulatedDriver
from ..io.data_writer import save_hdf5, save_csv

from .oscilloscope_widget import OscilloscopeWidget
from .fft_widget import FFTWidget
from .recorder_widget import RecorderWidget
from .digital_meter_widget import DigitalMeterWidget
from .channel_setup_widget import ChannelSetupWidget
from .hardware_dialog import HardwareDialog


CHANNEL_DEFAULTS = [
    ("CH1", "V",    10.0, 1.0,  WaveformType.SINE,     "#00ff88"),
    ("CH2", "V",    25.0, 0.7,  WaveformType.SINE,     "#ff6b6b"),
    ("CH3", "m/s²",  5.0, 2.0,  WaveformType.CHIRP,    "#4ecdc4"),
    ("CH4", "°C",    1.0, 0.5,  WaveformType.TRIANGLE, "#ffe66d"),
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

        self._engine = None
        self._init_channels()
        self._init_default_engine()
        self._build_ui()
        self._build_menu()
        self._build_toolbar()

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(50)  # 20 fps
        self._ui_timer.timeout.connect(self._refresh_ui)
        self._ui_timer.start()

    # ------------------------------------------------------------------
    def _init_channels(self) -> None:
        self.channels: List[Channel] = []
        self.generators: List[SignalGenerator] = []
        for i, (name, unit, freq, amp, wave, color) in enumerate(CHANNEL_DEFAULTS):
            ch = Channel(id=i + 1, name=name, unit=unit, sample_rate=1000.0, color=color)
            gen = SignalGenerator(waveform=wave, frequency=freq, amplitude=amp)
            self.channels.append(ch)
            self.generators.append(gen)

    def _init_default_engine(self) -> None:
        driver = SimulatedDriver(
            n_channels=len(self.channels),
            sample_rate=1000.0,
            generators=self.generators,
        )
        self._engine = ThreadedAcquisitionEngine(driver, self.channels, parent=self)
        self._engine.status_changed.connect(self._on_engine_status)

    def _init_engine_from_dialog(self, dlg: HardwareDialog) -> None:
        if self._engine:
            if isinstance(self._engine, ThreadedAcquisitionEngine):
                self._engine.stop_acquisition()
            elif isinstance(self._engine, MultiprocessAcquisitionEngine):
                self._engine.stop_acquisition()
                self._engine.close()

        # Rebuild channels to match selected n_channels
        old_n = len(self.channels)
        new_n = dlg.result_n_channels
        while len(self.channels) < new_n:
            i = len(self.channels)
            color = ["#00ff88","#ff6b6b","#4ecdc4","#ffe66d","#a8e6cf","#ff8b94","#b8b8ff","#ffd3b6"][i % 8]
            self.channels.append(Channel(id=i + 1, name=f"CH{i+1}", color=color,
                                         sample_rate=dlg.result_sample_rate))
            self.generators.append(SignalGenerator())
        self.channels = self.channels[:new_n]
        self.generators = self.generators[:new_n]
        for ch in self.channels:
            ch.sample_rate = dlg.result_sample_rate
            ch.clear()

        if dlg.result_use_multiprocess:
            self._engine = MultiprocessAcquisitionEngine(
                driver_cls_path=dlg.result_driver_path,
                driver_kwargs=dlg.result_driver_kwargs,
                channels=self.channels,
                sample_rate=dlg.result_sample_rate,
                parent=self,
            )
        else:
            mod_path, cls_name = dlg.result_driver_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            DriverCls = getattr(mod, cls_name)
            driver = DriverCls(**dlg.result_driver_kwargs)
            self._engine = ThreadedAcquisitionEngine(driver, self.channels, parent=self)

        self._engine.status_changed.connect(self._on_engine_status)
        self._scope.set_channels(self.channels)
        self._meters.set_channels(self.channels)
        self._update_toolbar_info()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._scope = OscilloscopeWidget(self.channels)
        self._tabs.addTab(self._scope, "Oscilloscope")

        self._fft = FFTWidget(self.channels)
        self._tabs.addTab(self._fft, "FFT Analyzer")

        self._recorder = RecorderWidget(self.channels)
        self._tabs.addTab(self._recorder, "Recorder")

        self._meters = DigitalMeterWidget(self.channels)
        self._tabs.addTab(self._meters, "Digital Meters")

        self._ch_setup = ChannelSetupWidget(self.channels, self.generators)
        self._ch_setup.channels_changed.connect(self._on_channels_changed)
        self._tabs.addTab(self._ch_setup, "Channel Setup")

        root.addWidget(self._tabs)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        act_save = QAction("Save as HDF5…", self, shortcut=QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._save_hdf5)
        file_menu.addAction(act_save)

        act_csv = QAction("Export as CSV…", self)
        act_csv.triggered.connect(self._save_csv)
        file_menu.addAction(act_csv)
        file_menu.addSeparator()
        file_menu.addAction(QAction("Quit", self, shortcut=QKeySequence.StandardKey.Quit,
                                    triggered=self.close))

        acq_menu = mb.addMenu("&Acquisition")
        self._act_start = QAction("▶  Start", self, shortcut="F5")
        self._act_start.triggered.connect(self._start_acquisition)
        acq_menu.addAction(self._act_start)

        self._act_stop = QAction("■  Stop", self, shortcut="F6")
        self._act_stop.triggered.connect(self._stop_acquisition)
        self._act_stop.setEnabled(False)
        acq_menu.addAction(self._act_stop)

        acq_menu.addSeparator()
        acq_menu.addAction(QAction("Hardware Setup…", self,
                                   triggered=self._open_hardware_dialog))
        acq_menu.addAction(QAction("Reset buffers", self, triggered=self._reset))

        mb.addMenu("&Help").addAction(QAction("About", self, triggered=self._about))

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
        tb.addAction(QAction("💾 Save HDF5", self, triggered=self._save_hdf5))
        tb.addAction(QAction("📄 Export CSV", self, triggered=self._save_csv))
        tb.addSeparator()
        tb.addAction(QAction("⚙ Hardware…", self, triggered=self._open_hardware_dialog))
        tb.addSeparator()

        self._info_lbl = QLabel()
        self._update_toolbar_info()
        tb.addWidget(self._info_lbl)

        self._status_lbl = QLabel("  Ready")
        self._time_lbl = QLabel("T = 0.000 s  ")
        sb = self.statusBar()
        sb.addWidget(self._status_lbl, 1)
        sb.addPermanentWidget(self._time_lbl)

    def _update_toolbar_info(self) -> None:
        n = len(self.channels)
        sr = self.channels[0].sample_rate if self.channels else 0
        drv = type(self._engine).__name__.replace("AcquisitionEngine", "") if self._engine else "—"
        self._info_lbl.setText(f"  {drv} | {n} ch | {sr:,.0f} Hz  ")

    # ------------------------------------------------------------------
    @pyqtSlot()
    def _start_acquisition(self) -> None:
        if not self._engine:
            return
        self._engine.start_acquisition()
        self._act_start.setEnabled(False)
        self._act_stop.setEnabled(True)
        self._tb_start.setEnabled(False)
        self._tb_stop.setEnabled(True)

    @pyqtSlot()
    def _stop_acquisition(self) -> None:
        if not self._engine:
            return
        self._engine.stop_acquisition()
        self._act_start.setEnabled(True)
        self._act_stop.setEnabled(False)
        self._tb_start.setEnabled(True)
        self._tb_stop.setEnabled(False)

    @pyqtSlot()
    def _reset(self) -> None:
        was_running = (
            isinstance(self._engine, ThreadedAcquisitionEngine) and
            hasattr(self._engine, '_driver') and self._engine._driver._stop_event is not None
            and not self._engine._driver._stop_event.is_set()
        )
        if was_running:
            self._stop_acquisition()
        if self._engine:
            self._engine.reset()
        if was_running:
            self._start_acquisition()

    @pyqtSlot(str)
    def _on_engine_status(self, status: str) -> None:
        self._status_lbl.setText(f"  Acquisition: {status}")

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
        if self._engine:
            self._time_lbl.setText(f"T = {self._engine.elapsed_time:.3f} s  ")

    @pyqtSlot()
    def _on_channels_changed(self) -> None:
        self._scope.set_channels(self.channels)
        self._meters.set_channels(self.channels)
        self._update_toolbar_info()

    @pyqtSlot()
    def _open_hardware_dialog(self) -> None:
        dlg = HardwareDialog(self)
        if dlg.exec() == HardwareDialog.DialogCode.Accepted:
            self._init_engine_from_dialog(dlg)

    @pyqtSlot()
    def _save_hdf5(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5", str(Path.home() / "recording.h5"), "HDF5 (*.h5 *.hdf5)"
        )
        if path:
            try:
                save_hdf5(self.channels, path)
                self._status_lbl.setText(f"  Saved: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Save error", str(exc))

    @pyqtSlot()
    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(Path.home() / "recording.csv"), "CSV (*.csv)"
        )
        if path:
            try:
                save_csv(self.channels, path)
                self._status_lbl.setText(f"  Exported: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export error", str(exc))

    @pyqtSlot()
    def _about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<h3>DeweSoft Clone — Python DAQ Studio</h3>"
            "<p>Personal learning project.</p>"
            "<b>Stack:</b> PyQt6 · pyqtgraph · NumPy · SciPy · h5py<br>"
            "<b>Hardware:</b> Simulated · sounddevice · openDAQ · Serial · Modbus<br>"
            "<b>Performance:</b> SharedMemory ring buffer · Min-Max decimation"
        )

    def closeEvent(self, event) -> None:
        if self._engine:
            if isinstance(self._engine, ThreadedAcquisitionEngine):
                self._engine.stop_acquisition()
            elif isinstance(self._engine, MultiprocessAcquisitionEngine):
                self._engine.stop_acquisition()
                self._engine.close()
        event.accept()
