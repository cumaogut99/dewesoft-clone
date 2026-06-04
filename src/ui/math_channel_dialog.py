"""
Math channel dialog — define a new calculated channel from an existing source.
"""
from __future__ import annotations
from typing import List
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox,
    QLineEdit, QDoubleSpinBox, QSpinBox, QGroupBox,
    QDialogButtonBox, QLabel, QStackedWidget, QWidget
)
from PyQt6.QtGui import QColor

from ..core.channel import Channel
from ..core.math_engine import OPERATIONS, MathChannelDef

CHANNEL_COLORS = [
    "#c9cbff", "#f38ba8", "#a6e3a1", "#f9e2af",
    "#94e2d5", "#89dceb", "#cba6f7", "#fab387",
]


class MathChannelDialog(QDialog):
    def __init__(self, channels: List[Channel], next_id: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Math Channel")
        self.channels = channels
        self._next_id = next_id
        self.result_defn: MathChannelDef | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form_grp = QGroupBox("Math Channel Definition")
        form = QFormLayout(form_grp)

        self._name_edit = QLineEdit("Math1")
        form.addRow("Name:", self._name_edit)

        self._unit_edit = QLineEdit()
        form.addRow("Unit:", self._unit_edit)

        self._src_combo = QComboBox()
        for ch in self.channels:
            self._src_combo.addItem(ch.name, userData=ch.id)
        form.addRow("Source channel:", self._src_combo)

        self._op_combo = QComboBox()
        self._op_combo.addItems(list(OPERATIONS.keys()))
        self._op_combo.currentTextChanged.connect(self._on_op_changed)
        form.addRow("Operation:", self._op_combo)

        layout.addWidget(form_grp)

        # Dynamic parameter panel (different params per operation)
        self._param_stack = QStackedWidget()

        self._rms_pg = self._make_rms_page()
        self._filter_pg = self._make_filter_page()
        self._bp_pg = self._make_bp_page()
        self._empty_pg = QWidget()

        self._param_stack.addWidget(self._rms_pg)      # 0
        self._param_stack.addWidget(self._filter_pg)   # 1
        self._param_stack.addWidget(self._bp_pg)       # 2
        self._param_stack.addWidget(self._empty_pg)    # 3 fallback

        layout.addWidget(self._param_stack)
        self._on_op_changed(self._op_combo.currentText())

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _make_rms_page(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self._rms_win = QSpinBox()
        self._rms_win.setRange(2, 10_000)
        self._rms_win.setValue(100)
        self._rms_win.setSuffix(" samples")
        f.addRow("Window size:", self._rms_win)
        return w

    def _make_filter_page(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self._filt_cutoff = QDoubleSpinBox()
        self._filt_cutoff.setRange(0.01, 100_000.0)
        self._filt_cutoff.setValue(100.0)
        self._filt_cutoff.setSuffix(" Hz")
        f.addRow("Cutoff frequency:", self._filt_cutoff)
        return w

    def _make_bp_page(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self._bp_low = QDoubleSpinBox()
        self._bp_low.setRange(0.01, 100_000.0)
        self._bp_low.setValue(10.0)
        self._bp_low.setSuffix(" Hz")
        f.addRow("Low cutoff:", self._bp_low)
        self._bp_high = QDoubleSpinBox()
        self._bp_high.setRange(0.01, 100_000.0)
        self._bp_high.setValue(500.0)
        self._bp_high.setSuffix(" Hz")
        f.addRow("High cutoff:", self._bp_high)
        return w

    def _on_op_changed(self, op: str) -> None:
        if op == "RMS":
            self._param_stack.setCurrentIndex(0)
        elif op in ("Low-pass", "High-pass"):
            self._param_stack.setCurrentIndex(1)
        elif op == "Band-pass":
            self._param_stack.setCurrentIndex(2)
        else:
            self._param_stack.setCurrentIndex(3)

    def _accept(self) -> None:
        op = self._op_combo.currentText()
        src_id = self._src_combo.currentData()
        name = self._name_edit.text().strip() or f"Math{self._next_id}"
        unit = self._unit_edit.text().strip()
        color = CHANNEL_COLORS[(self._next_id - 1) % len(CHANNEL_COLORS)]

        params: dict = {}
        if op == "RMS":
            params["window"] = self._rms_win.value()
        elif op in ("Low-pass", "High-pass"):
            params["cutoff"] = self._filt_cutoff.value()
        elif op == "Band-pass":
            params["low"] = self._bp_low.value()
            params["high"] = self._bp_high.value()

        # Find source channel sample rate
        src_sr = 1000.0
        for ch in self.channels:
            if ch.id == src_id:
                src_sr = ch.sample_rate
                break

        out_ch = Channel(
            id=self._next_id, name=name, unit=unit,
            sample_rate=src_sr, color=color,
        )

        self.result_defn = MathChannelDef(
            source_id=src_id,
            operation=op,
            output_channel=out_ch,
            params=params,
        )
        self.accept()
