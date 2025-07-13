import sys
from pyrpoc.gui_handler import AppState, StateSignalBus
from pyrpoc.gui import MainWindow
from PyQt6.QtWidgets import QApplication

def main():
    app_state = AppState() # can initialize GUI configs with this
    signals = StateSignalBus()

    app = QApplication(sys.argv)
    win = MainWindow(app_state, signals)
    
    signals.bind_controllers(app_state, win)
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()