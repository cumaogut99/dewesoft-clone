#!/usr/bin/env python3
"""
DeweSoft Clone — Python DAQ Studio
Entry point.
"""
import sys
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.ui.main_window import MainWindow, _apply_dark_theme


def main() -> None:
    pg.setConfigOptions(antialias=True, useOpenGL=False)
    pg.setConfigOption("background", "#1e1e2e")
    pg.setConfigOption("foreground", "#cdd6f4")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _apply_dark_theme(app)
    app.setApplicationName("DeweSoft Clone")
    app.setOrganizationName("Personal")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
