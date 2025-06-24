import sys
from pyrpoc.gui.gui_handler import AppState, StateSignalBus
from pyrpoc.gui.gui import MainWindow
from PyQt6.QtWidgets import QApplication

if __name__ == '__main__':
    app_state = AppState() # can initialize GUI configs with this
    signals = StateSignalBus()

    app = QApplication(sys.argv)
    win = MainWindow(app_state, signals)
    
    signals.bind_controllers(app_state, win)
    win.show()
    sys.exit(app.exec())