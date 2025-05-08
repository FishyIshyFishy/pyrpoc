import sys
from gui.gui_handler import AppState, StateSignalBus
from gui.gui import MainWindow
from PyQt6.QtWidgets import QApplication

if __name__ == '__main__':
    app_state = AppState()
    signals = StateSignalBus()
    signals.bind_controllers(app_state)

    app = QApplication(sys.argv)
    win = MainWindow(app_state, signals)
    win.show()
    sys.exit(app.exec())