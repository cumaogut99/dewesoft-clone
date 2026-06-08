"""
Channel setup panel — configure channels and their signal generators.
"""
from __future__ import annotations
from typing import List, Callable
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QColorDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QFormLayout, QLineEdit, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette

from ..core.channel import Channel, ChannelType
from ..core.signal_generator import SignalGenerator, WaveformType


CHANNEL_COLORS = [
    "#00ff88", "#ff6b6b", "#4ecdc4", "#ffe66d",
    "#a8e6cf", "#ff8b94", "#b8b8ff", "#ffd3b6",
]


class ChannelSetupWidget(QWidget):
    channels_changed = pyqtSignal()

    def __init__(
        self,
        channels: List[Channel],
        generators: List[SignalGenerator],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.channels = channels
        self.generators = generators
        self._selected_row = 0
        self._build_ui()
        self._populate_table()
        self._select_row(0)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main = QHBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)

        # Left: channel list
        left = QVBoxLayout()
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["#", "Name", "Unit", "Enabled"])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.itemSelectionChanged.connect(
            lambda: self._select_row(self._table.currentRow())
        )
        self._table.setMaximumWidth(300)
        left.addWidget(self._table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(self._add_channel)
        rem_btn = QPushButton("- Remove")
        rem_btn.clicked.connect(self._remove_channel)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rem_btn)
        left.addLayout(btn_row)

        main.addLayout(left)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        main.addWidget(sep)

        # Right: channel properties
        right = QVBoxLayout()

        ch_grp = QGroupBox("Channel Properties")
        ch_form = QFormLayout(ch_grp)

        self._name_edit = QLineEdit()
        self._name_edit.editingFinished.connect(self._apply_channel)
        ch_form.addRow("Name:", self._name_edit)

        self._unit_edit = QLineEdit()
        self._unit_edit.editingFinished.connect(self._apply_channel)
        ch_form.addRow("Unit:", self._unit_edit)

        self._sr_spin = QDoubleSpinBox()
        self._sr_spin.setRange(1.0, 100_000.0)
        self._sr_spin.setValue(1000.0)
        self._sr_spin.setSuffix(" Hz")
        self._sr_spin.editingFinished.connect(self._apply_channel)
        ch_form.addRow("Sample rate:", self._sr_spin)

        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(-1e6, 1e6)
        self._scale_spin.setValue(1.0)
        self._scale_spin.editingFinished.connect(self._apply_channel)
        ch_form.addRow("Scale:", self._scale_spin)

        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-1e6, 1e6)
        self._offset_spin.setValue(0.0)
        self._offset_spin.editingFinished.connect(self._apply_channel)
        ch_form.addRow("Offset:", self._offset_spin)

        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(60, 24)
        self._color_btn.clicked.connect(self._pick_color)
        ch_form.addRow("Color:", self._color_btn)

        right.addWidget(ch_grp)

        gen_grp = QGroupBox("Signal Generator")
        gen_form = QFormLayout(gen_grp)

        self._wave_combo = QComboBox()
        self._wave_combo.addItems([w.value for w in WaveformType])
        self._wave_combo.currentTextChanged.connect(self._apply_generator)
        gen_form.addRow("Waveform:", self._wave_combo)

        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(0.001, 50_000.0)
        self._freq_spin.setValue(10.0)
        self._freq_spin.setSuffix(" Hz")
        self._freq_spin.editingFinished.connect(self._apply_generator)
        gen_form.addRow("Frequency:", self._freq_spin)

        self._amp_spin = QDoubleSpinBox()
        self._amp_spin.setRange(0.0, 1000.0)
        self._amp_spin.setValue(1.0)
        self._amp_spin.editingFinished.connect(self._apply_generator)
        gen_form.addRow("Amplitude:", self._amp_spin)

        self._dc_spin = QDoubleSpinBox()
        self._dc_spin.setRange(-1000.0, 1000.0)
        self._dc_spin.setValue(0.0)
        self._dc_spin.editingFinished.connect(self._apply_generator)
        gen_form.addRow("DC Offset:", self._dc_spin)

        self._phase_spin = QDoubleSpinBox()
        self._phase_spin.setRange(-360.0, 360.0)
        self._phase_spin.setValue(0.0)
        self._phase_spin.setSuffix(" °")
        self._phase_spin.editingFinished.connect(self._apply_generator)
        gen_form.addRow("Phase:", self._phase_spin)

        self._noise_spin = QDoubleSpinBox()
        self._noise_spin.setRange(0.0, 10.0)
        self._noise_spin.setValue(0.0)
        self._noise_spin.setSingleStep(0.01)
        self._noise_spin.editingFinished.connect(self._apply_generator)
        gen_form.addRow("Noise level:", self._noise_spin)

        right.addWidget(gen_grp)
        right.addStretch()
        main.addLayout(right)

    # ------------------------------------------------------------------
    def _populate_table(self) -> None:
        self._table.setRowCount(0)
        for i, ch in enumerate(self.channels):
            self._table.insertRow(i)
            self._table.setItem(i, 0, QTableWidgetItem(str(ch.id)))
            self._table.setItem(i, 1, QTableWidgetItem(ch.name))
            self._table.setItem(i, 2, QTableWidgetItem(ch.unit))
            enabled_item = QTableWidgetItem("✓" if ch.enabled else "")
            enabled_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(i, 3, enabled_item)

    def _select_row(self, row: int) -> None:
        if row < 0 or row >= len(self.channels):
            return
        self._selected_row = row
        ch = self.channels[row]
        gen = self.generators[row]

        self._name_edit.setText(ch.name)
        self._unit_edit.setText(ch.unit)
        self._sr_spin.setValue(ch.sample_rate)
        self._scale_spin.setValue(ch.scale)
        self._offset_spin.setValue(ch.offset)
        self._set_color_btn(ch.color)

        self._wave_combo.setCurrentText(gen.waveform.value)
        self._freq_spin.setValue(gen.frequency)
        self._amp_spin.setValue(gen.amplitude)
        self._dc_spin.setValue(gen.dc_offset)
        self._phase_spin.setValue(gen.phase_deg)
        self._noise_spin.setValue(gen.noise_level)

    def _apply_channel(self) -> None:
        if not self.channels:
            return
        ch = self.channels[self._selected_row]
        ch.name = self._name_edit.text()
        ch.unit = self._unit_edit.text()
        ch.sample_rate = self._sr_spin.value()
        ch.scale = self._scale_spin.value()
        ch.offset = self._offset_spin.value()
        self._populate_table()
        self.channels_changed.emit()

    def _apply_generator(self) -> None:
        if not self.generators:
            return
        gen = self.generators[self._selected_row]
        gen.waveform = WaveformType(self._wave_combo.currentText())
        gen.frequency = self._freq_spin.value()
        gen.amplitude = self._amp_spin.value()
        gen.dc_offset = self._dc_spin.value()
        gen.phase_deg = self._phase_spin.value()
        gen.noise_level = self._noise_spin.value()

    def _pick_color(self) -> None:
        ch = self.channels[self._selected_row]
        color = QColorDialog.getColor(QColor(ch.color), self, "Pick channel color")
        if color.isValid():
            ch.color = color.name()
            self._set_color_btn(ch.color)
            self.channels_changed.emit()

    def _set_color_btn(self, hex_color: str) -> None:
        self._color_btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555;")

    def _add_channel(self) -> None:
        new_id = max((c.id for c in self.channels), default=0) + 1
        color = CHANNEL_COLORS[(len(self.channels)) % len(CHANNEL_COLORS)]
        ch = Channel(id=new_id, name=f"CH{new_id}", color=color)
        gen = SignalGenerator()
        self.channels.append(ch)
        self.generators.append(gen)
        self._populate_table()
        self._table.setCurrentCell(len(self.channels) - 1, 0)
        self.channels_changed.emit()

    def _remove_channel(self) -> None:
        row = self._selected_row
        if row < 0 or row >= len(self.channels):
            return
        self.channels.pop(row)
        self.generators.pop(row)
        self._populate_table()
        self._selected_row = max(0, row - 1)
        if self.channels:
            self._table.setCurrentCell(self._selected_row, 0)
        self.channels_changed.emit()
