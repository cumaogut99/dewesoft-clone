"""
Main application window — DeweSoft-style tabbed measurement environment.
"""
from __future__ import annotations
import importlib
from pathlib import Path
from typing import List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QToolBar,
    QFileDialog, QMessageBox, QLabel, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence, QColor, QPalette

from ..core.channel import Channel
from ..core.signal_generator import SignalGenerator, WaveformType
from ..core.acquisition_engine import ThreadedAcquisitionEngine, MultiprocessAcquisitionEngine
from ..core.math_engine import MathEngine
from ..core.playback_engine import PlaybackEngine
from ..hardware.simulated_driver import SimulatedDriver
from ..io.data_writer import save_hdf5, save_csv
from ..io.project import save_project, load_project

from .oscilloscope_widget import OscilloscopeWidget
from .fft_widget import FFTWidget
from .recorder_widget import RecorderWidget
from .digital_meter_widget import DigitalMeterWidget
from .xy_plot_widget import XYPlotWidget
from .waterfall_widget import WaterfallWidget
from .channel_setup_widget import ChannelSetupWidget
from .math_channel_dialog import MathChannelDialog
from .hardware_dialog import HardwareDialog
from .playback_widget import PlaybackBar


CHANNEL_DEFAULTS = [
    ("CH1", "V",    10.0, 1.0,  WaveformType.SINE,     "#00ff88"),
    ("CH2", "V",    25.0, 0.7,  WaveformType.SINE,     "#ff6b6b"),
    ("CH3", "m/s²",  5.0, 2.0,  WaveformType.CHIRP,    "#4ecdc4"),
    ("CH4", "°C",    1.0, 0.5,  WaveformType.TRIANGLE, "#ffe66d"),
]


def _apply_dark_theme(app) -> None:
    palette = QPalette()
    c = {
        QPalette.ColorRole.Window:          QColor(30, 30, 46),
        QPalette.ColorRole.WindowText:      QColor(205, 214, 244),
        QPalette.ColorRole.Base:            QColor(24, 24, 37),
        QPalette.ColorRole.AlternateBase:   QColor(30, 30, 46),
        QPalette.ColorRole.ToolTipBase:     QColor(24, 24, 37),
        QPalette.ColorRole.ToolTipText:     QColor(205, 214, 244),
        QPalette.ColorRole.Text:            QColor(205, 214, 244),
        QPalette.ColorRole.Button:          QColor(49, 50, 68),
        QPalette.ColorRole.ButtonText:      QColor(205, 214, 244),
        QPalette.ColorRole.BrightText:      QColor(243, 139, 168),
        QPalette.ColorRole.Link:            QColor(137, 180, 250),
        QPalette.ColorRole.Highlight:       QColor(137, 180, 250),
        QPalette.ColorRole.HighlightedText: QColor(24, 24, 37),
    }
    for role, color in c.items():
        palette.setColor(role, color)
    app.setPalette(palette)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeweSoft Clone — Python DAQ Studio")
        self.resize(1440, 900)

        self._engine = None
        self._math_engine = MathEngine()
        self._playback_engine: PlaybackEngine | None = None
        self._playback_mode = False

        self._init_channels()
        self._init_default_engine()
        self._build_ui()
        self._build_menu()
        self._build_toolbar()

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(50)
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

    def _all_channels(self) -> List[Channel]:
        """Hardware channels + math channels."""
        return self.channels + self._math_engine.output_channels

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._scope = OscilloscopeWidget(self._all_channels())
        self._tabs.addTab(self._scope, "Oscilloscope")

        self._fft = FFTWidget(self._all_channels())
        self._tabs.addTab(self._fft, "FFT Analyzer")

        self._recorder = RecorderWidget(self._all_channels())
        self._tabs.addTab(self._recorder, "Recorder")

        self._meters = DigitalMeterWidget(self._all_channels())
        self._tabs.addTab(self._meters, "Digital Meters")

        self._xy = XYPlotWidget(self._all_channels())
        self._tabs.addTab(self._xy, "XY Plot")

        self._waterfall = WaterfallWidget(self._all_channels())
        self._tabs.addTab(self._waterfall, "Waterfall")

        self._ch_setup = ChannelSetupWidget(self.channels, self.generators)
        self._ch_setup.channels_changed.connect(self._on_channels_changed)
        self._tabs.addTab(self._ch_setup, "Channel Setup")

        root.addWidget(self._tabs)

        # Playback bar (hidden by default)
        self._pb_bar_widget = QWidget()
        self._pb_bar_layout = QVBoxLayout(self._pb_bar_widget)
        self._pb_bar_layout.setContentsMargins(0, 0, 0, 0)
        self._pb_bar_widget.hide()
        root.addWidget(self._pb_bar_widget)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # --- File ---
        file_menu = mb.addMenu("&File")
        file_menu.addAction(QAction("New project", self, triggered=self._new_project))
        file_menu.addAction(QAction("Open project…", self,
                                    shortcut=QKeySequence.StandardKey.Open,
                                    triggered=self._open_project))
        file_menu.addAction(QAction("Save project…", self,
                                    shortcut=QKeySequence.StandardKey.Save,
                                    triggered=self._save_project_dialog))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Save data (HDF5)…", self,
                                    triggered=self._save_hdf5))
        file_menu.addAction(QAction("Export data (CSV)…", self,
                                    triggered=self._save_csv))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Open recording for playback…", self,
                                    triggered=self._open_playback))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Quit", self,
                                    shortcut=QKeySequence.StandardKey.Quit,
                                    triggered=self.close))

        # --- Acquisition ---
        acq = mb.addMenu("&Acquisition")
        self._act_start = QAction("▶  Start", self, shortcut="F5",
                                   triggered=self._start_acquisition)
        self._act_stop = QAction("■  Stop", self, shortcut="F6",
                                  triggered=self._stop_acquisition)
        self._act_stop.setEnabled(False)
        acq.addAction(self._act_start)
        acq.addAction(self._act_stop)
        acq.addSeparator()
        acq.addAction(QAction("Hardware Setup…", self,
                               triggered=self._open_hardware_dialog))
        acq.addAction(QAction("Reset buffers", self, triggered=self._reset))

        # --- Analysis ---
        ana = mb.addMenu("A&nalysis")
        ana.addAction(QAction("Add math channel…", self,
                               triggered=self._add_math_channel))
        ana.addAction(QAction("Remove all math channels", self,
                               triggered=self._remove_all_math))

        # --- Help ---
        mb.addMenu("&Help").addAction(
            QAction("About", self, triggered=self._about)
        )

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._tb_start = QAction("▶  Start", self, triggered=self._start_acquisition)
        self._tb_stop = QAction("■  Stop", self, triggered=self._stop_acquisition)
        self._tb_stop.setEnabled(False)
        tb.addAction(self._tb_start)
        tb.addAction(self._tb_stop)
        tb.addSeparator()
        tb.addAction(QAction("💾 HDF5", self, triggered=self._save_hdf5))
        tb.addAction(QAction("📄 CSV", self, triggered=self._save_csv))
        tb.addSeparator()
        tb.addAction(QAction("⚙ Hardware…", self, triggered=self._open_hardware_dialog))
        tb.addAction(QAction("∫ Math channel…", self, triggered=self._add_math_channel))
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
        n_hw = len(self.channels)
        n_math = len(self._math_engine.output_channels)
        sr = self.channels[0].sample_rate if self.channels else 0
        math_str = f" + {n_math} math" if n_math else ""
        self._info_lbl.setText(f"  {n_hw}{math_str} ch | {sr:,.0f} Hz  ")

    # ------------------------------------------------------------------
    @pyqtSlot()
    def _start_acquisition(self) -> None:
        if not self._engine or self._playback_mode:
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
        if self._engine:
            self._engine.reset()
        self._math_engine.update([])   # flush math channels

    @pyqtSlot(str)
    def _on_engine_status(self, status: str) -> None:
        self._status_lbl.setText(f"  {status.capitalize()}")

    @pyqtSlot()
    def _refresh_ui(self) -> None:
        # Update math channels from latest hardware data
        self._math_engine.update(self.channels)

        tab = self._tabs.currentIndex()
        if tab == 0:
            self._scope.update_plots()
        elif tab == 1:
            self._fft.update_plots()
        elif tab == 2:
            self._recorder.update_plots()
        elif tab == 3:
            self._meters.update_plots()
        elif tab == 4:
            self._xy.update_plots()
        elif tab == 5:
            self._waterfall.update_plots()

        elapsed = self._engine.elapsed_time if self._engine else 0.0
        self._time_lbl.setText(f"T = {elapsed:.3f} s  ")

    @pyqtSlot()
    def _on_channels_changed(self) -> None:
        all_ch = self._all_channels()
        self._scope.set_channels(all_ch)
        self._meters.set_channels(all_ch)
        self._xy.set_channels(all_ch)
        self._waterfall.set_channels(all_ch)
        self._update_toolbar_info()

    # ------------------------------------------------------------------
    # Math channels
    # ------------------------------------------------------------------
    @pyqtSlot()
    def _add_math_channel(self) -> None:
        existing_ids = {ch.id for ch in self.channels}
        existing_ids |= {ch.id for ch in self._math_engine.output_channels}
        next_id = max(existing_ids, default=0) + 1

        dlg = MathChannelDialog(self.channels, next_id, self)
        if dlg.exec() == MathChannelDialog.DialogCode.Accepted and dlg.result_defn:
            self._math_engine.add(dlg.result_defn)
            self._on_channels_changed()
            self._status_lbl.setText(
                f"  Added math channel: {dlg.result_defn.output_channel.name}"
            )

    @pyqtSlot()
    def _remove_all_math(self) -> None:
        for defn in list(self._math_engine.definitions):
            self._math_engine.remove(defn.output_channel.id)
        self._on_channels_changed()

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------
    @pyqtSlot()
    def _open_hardware_dialog(self) -> None:
        dlg = HardwareDialog(self)
        if dlg.exec() == HardwareDialog.DialogCode.Accepted:
            self._apply_hardware(dlg)

    def _apply_hardware(self, dlg: HardwareDialog) -> None:
        if self._engine:
            self._engine.stop_acquisition()
            if isinstance(self._engine, MultiprocessAcquisitionEngine):
                self._engine.close()

        new_n = dlg.result_n_channels
        while len(self.channels) < new_n:
            i = len(self.channels)
            color = ["#00ff88","#ff6b6b","#4ecdc4","#ffe66d",
                     "#a8e6cf","#ff8b94","#b8b8ff","#ffd3b6"][i % 8]
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
            driver = getattr(mod, cls_name)(**dlg.result_driver_kwargs)
            self._engine = ThreadedAcquisitionEngine(driver, self.channels, parent=self)

        self._engine.status_changed.connect(self._on_engine_status)
        self._on_channels_changed()
        self._update_toolbar_info()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    @pyqtSlot()
    def _open_playback(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recording", str(Path.home()),
            "Recordings (*.h5 *.hdf5 *.csv)"
        )
        if not path:
            return

        if self._engine:
            self._engine.stop_acquisition()

        for ch in self.channels:
            ch.clear()

        if self._playback_engine:
            self._playback_engine.stop_playback()

        self._playback_engine = PlaybackEngine(self.channels, parent=self)
        try:
            n = self._playback_engine.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        # Show playback bar
        pb_bar = PlaybackBar(self._playback_engine)
        pb_bar.load_requested.connect(self._open_playback)
        pb_bar.set_duration(self._playback_engine.duration)
        self._playback_engine.status_changed.connect(self._on_engine_status)

        # Replace existing bar widget content
        for i in reversed(range(self._pb_bar_layout.count())):
            self._pb_bar_layout.itemAt(i).widget().deleteLater()
        self._pb_bar_layout.addWidget(pb_bar)
        self._pb_bar_widget.show()

        # Wire playback data_ready → UI refresh (already covered by timer)
        self._playback_mode = True
        self._engine = self._playback_engine  # type: ignore[assignment]
        self._status_lbl.setText(f"  Playback: {Path(path).name}  ({n} channels)")

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------
    @pyqtSlot()
    def _new_project(self) -> None:
        self._stop_acquisition()
        self._init_channels()
        self._init_default_engine()
        for defn in list(self._math_engine.definitions):
            self._math_engine.remove(defn.output_channel.id)
        self._on_channels_changed()

    @pyqtSlot()
    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", str(Path.home()),
            "DAQ Project (*.daqproj)"
        )
        if not path:
            return
        try:
            doc = load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self._stop_acquisition()
        self.channels = doc["channels"]
        self.generators = doc["generators"]
        self._on_channels_changed()
        self._status_lbl.setText(f"  Loaded project: {Path(path).name}")

    @pyqtSlot()
    def _save_project_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", str(Path.home() / "project.daqproj"),
            "DAQ Project (*.daqproj)"
        )
        if path:
            try:
                save_project(self.channels, self.generators, path=path)
                self._status_lbl.setText(f"  Saved project: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Save error", str(exc))

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------
    @pyqtSlot()
    def _save_hdf5(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5", str(Path.home() / "recording.h5"),
            "HDF5 (*.h5 *.hdf5)"
        )
        if path:
            try:
                save_hdf5(self._all_channels(), path)
                self._status_lbl.setText(f"  Saved: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Save error", str(exc))

    @pyqtSlot()
    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(Path.home() / "recording.csv"),
            "CSV (*.csv)"
        )
        if path:
            try:
                save_csv(self._all_channels(), path)
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
            "<b>Views:</b> Oscilloscope · FFT · Recorder · Meters · XY · Waterfall<br>"
            "<b>Features:</b> Trigger · Cursors · Math channels · Playback · Projects"
        )

    def closeEvent(self, event) -> None:
        if self._engine:
            if hasattr(self._engine, "stop_acquisition"):
                self._engine.stop_acquisition()
            if isinstance(self._engine, MultiprocessAcquisitionEngine):
                self._engine.close()
        event.accept()
