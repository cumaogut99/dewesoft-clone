"""
Hardware selection dialog — shown at startup or via Acquisition > Hardware Setup.
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QFormLayout, QLineEdit, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QTextEdit,
    QDialogButtonBox, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from ..hardware.base import DriverInfo


_DRIVERS = {
    "Simulated (no hardware)": {
        "cls_path": "src.hardware.simulated_driver.SimulatedDriver",
        "probe_cls": "src.hardware.simulated_driver.SimulatedDriver",
        "available": True,
    },
    "Sound Card (sounddevice)": {
        "cls_path": "src.hardware.sounddevice_driver.SounddeviceDriver",
        "probe_cls": "src.hardware.sounddevice_driver.SounddeviceDriver",
        "available": None,  # checked at runtime
    },
    "openDAQ": {
        "cls_path": "src.hardware.opendaq_driver.OpenDAQDriver",
        "probe_cls": "src.hardware.opendaq_driver.OpenDAQDriver",
        "available": None,
    },
    "Serial / Arduino": {
        "cls_path": "src.hardware.serial_driver.SerialDriver",
        "probe_cls": "src.hardware.serial_driver.SerialDriver",
        "available": None,
    },
    "Modbus RTU/TCP": {
        "cls_path": "src.hardware.modbus_driver.ModbusDriver",
        "probe_cls": "src.hardware.modbus_driver.ModbusDriver",
        "available": None,
    },
}


def _probe(cls_path: str) -> bool:
    try:
        import importlib
        mod, cls = cls_path.rsplit(".", 1)
        m = importlib.import_module(mod)
        return getattr(m, cls).probe()
    except Exception:
        return False


class HardwareDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hardware Setup")
        self.setMinimumWidth(520)
        self.result_driver_path: str = "src.hardware.simulated_driver.SimulatedDriver"
        self.result_driver_kwargs: dict = {}
        self.result_n_channels: int = 4
        self.result_sample_rate: float = 1000.0
        self.result_use_multiprocess: bool = False

        self._build_ui()
        self._probe_drivers()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Driver selector
        drv_grp = QGroupBox("Hardware Driver")
        drv_form = QFormLayout(drv_grp)

        self._drv_combo = QComboBox()
        for name in _DRIVERS:
            self._drv_combo.addItem(name)
        self._drv_combo.currentTextChanged.connect(self._on_driver_changed)
        drv_form.addRow("Driver:", self._drv_combo)

        self._status_lbl = QLabel()
        self._status_lbl.setFont(QFont("Monospace", 9))
        drv_form.addRow("Status:", self._status_lbl)

        layout.addWidget(drv_grp)

        # Common settings
        common_grp = QGroupBox("Common Settings")
        common_form = QFormLayout(common_grp)

        self._n_ch_spin = QSpinBox()
        self._n_ch_spin.setRange(1, 32)
        self._n_ch_spin.setValue(4)
        common_form.addRow("Channels:", self._n_ch_spin)

        self._sr_spin = QDoubleSpinBox()
        self._sr_spin.setRange(1.0, 200_000.0)
        self._sr_spin.setValue(1000.0)
        self._sr_spin.setSuffix(" Hz")
        common_form.addRow("Sample rate:", self._sr_spin)

        self._mp_cb = QCheckBox("Use multiprocessing (recommended > 50 kHz)")
        self._mp_cb.setToolTip(
            "Runs the driver in a separate process via shared memory.\n"
            "Bypasses the Python GIL for true parallel acquisition."
        )
        common_form.addRow("", self._mp_cb)

        layout.addWidget(common_grp)

        # Driver-specific settings tabs
        self._tabs = QTabWidget()

        # --- Simulated
        sim = QWidget()
        sim_form = QFormLayout(sim)
        self._sim_wave = QComboBox()
        self._sim_wave.addItems(["Sine", "Square", "Triangle", "Chirp", "Noise"])
        sim_form.addRow("Waveform:", self._sim_wave)
        self._tabs.addTab(sim, "Simulated")

        # --- Sound card
        snd = QWidget()
        snd_form = QFormLayout(snd)
        self._snd_dev = QLineEdit()
        self._snd_dev.setPlaceholderText("Leave blank for default device")
        snd_form.addRow("Device:", self._snd_dev)
        self._snd_list_btn = QPushButton("List devices")
        self._snd_list_btn.clicked.connect(self._list_sound_devices)
        snd_form.addRow("", self._snd_list_btn)
        self._snd_info = QTextEdit()
        self._snd_info.setReadOnly(True)
        self._snd_info.setMaximumHeight(80)
        snd_form.addRow(self._snd_info)
        self._tabs.addTab(snd, "Sound Card")

        # --- openDAQ
        daq = QWidget()
        daq_form = QFormLayout(daq)
        self._daq_conn = QLineEdit()
        self._daq_conn.setPlaceholderText('Empty → auto-discover  |  "daq.opcua://192.168.1.100"')
        daq_form.addRow("Connection:", self._daq_conn)
        self._daq_sim = QCheckBox("Use built-in simulated device (no hardware)")
        self._daq_sim.setChecked(False)
        daq_form.addRow(self._daq_sim)
        self._tabs.addTab(daq, "openDAQ")

        # --- Serial
        ser = QWidget()
        ser_form = QFormLayout(ser)
        self._ser_port = QLineEdit("/dev/ttyUSB0")
        ser_form.addRow("Port:", self._ser_port)
        self._ser_baud = QSpinBox()
        self._ser_baud.setRange(300, 3_000_000)
        self._ser_baud.setValue(115200)
        ser_form.addRow("Baud rate:", self._ser_baud)
        self._tabs.addTab(ser, "Serial")

        # --- Modbus
        mod = QWidget()
        mod_form = QFormLayout(mod)
        self._mod_mode = QComboBox()
        self._mod_mode.addItems(["tcp", "rtu"])
        mod_form.addRow("Mode:", self._mod_mode)
        self._mod_host = QLineEdit("192.168.1.100")
        mod_form.addRow("Host/IP:", self._mod_host)
        self._mod_port = QSpinBox()
        self._mod_port.setRange(1, 65535)
        self._mod_port.setValue(502)
        mod_form.addRow("TCP Port:", self._mod_port)
        self._mod_reg = QSpinBox()
        self._mod_reg.setRange(0, 65535)
        mod_form.addRow("Start register:", self._mod_reg)
        self._mod_scale = QDoubleSpinBox()
        self._mod_scale.setRange(1e-6, 1e6)
        self._mod_scale.setValue(0.1)
        mod_form.addRow("Scale factor:", self._mod_scale)
        self._tabs.addTab(mod, "Modbus")

        layout.addWidget(self._tabs)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _probe_drivers(self) -> None:
        for name, info in _DRIVERS.items():
            if info["available"] is None:
                info["available"] = _probe(info["probe_cls"])
        self._on_driver_changed(self._drv_combo.currentText())

    def _on_driver_changed(self, name: str) -> None:
        info = _DRIVERS.get(name, {})
        avail = info.get("available", False)
        if avail:
            self._status_lbl.setText("✓ Available")
            self._status_lbl.setStyleSheet("color: #00ff88;")
        else:
            self._status_lbl.setText("✗ Not available (install required packages)")
            self._status_lbl.setStyleSheet("color: #ff6b6b;")

    def _list_sound_devices(self) -> None:
        try:
            import sounddevice as sd
            self._snd_info.setText(str(sd.query_devices()))
        except ImportError:
            self._snd_info.setText("sounddevice not installed.\nRun: pip install sounddevice")

    def _accept(self) -> None:
        drv_name = self._drv_combo.currentText()
        self.result_driver_path = _DRIVERS[drv_name]["cls_path"]
        self.result_n_channels = self._n_ch_spin.value()
        self.result_sample_rate = self._sr_spin.value()
        self.result_use_multiprocess = self._mp_cb.isChecked()

        if "Simulated" in drv_name:
            self.result_driver_kwargs = {
                "n_channels": self.result_n_channels,
                "sample_rate": self.result_sample_rate,
            }
        elif "Sound" in drv_name:
            dev_text = self._snd_dev.text().strip()
            self.result_driver_kwargs = {
                "n_channels": self.result_n_channels,
                "sample_rate": self.result_sample_rate,
                "device": int(dev_text) if dev_text.isdigit() else (dev_text or None),
            }
        elif "openDAQ" in drv_name:
            self.result_driver_kwargs = {
                "n_channels": self.result_n_channels,
                "sample_rate": self.result_sample_rate,
                "connection_string": self._daq_conn.text().strip(),
                "use_simulated": self._daq_sim.isChecked(),
            }
        elif "Serial" in drv_name:
            self.result_driver_kwargs = {
                "n_channels": self.result_n_channels,
                "sample_rate": self.result_sample_rate,
                "port": self._ser_port.text().strip(),
                "baud_rate": self._ser_baud.value(),
            }
        elif "Modbus" in drv_name:
            self.result_driver_kwargs = {
                "n_channels": self.result_n_channels,
                "sample_rate": self.result_sample_rate,
                "mode": self._mod_mode.currentText(),
                "host": self._mod_host.text().strip(),
                "port": self._mod_port.value(),
                "start_register": self._mod_reg.value(),
                "scale": self._mod_scale.value(),
            }

        self.accept()
