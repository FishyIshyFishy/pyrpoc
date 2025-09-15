from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
import PyQt6Ads as qtads

qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.DisableTabTextEliding, True)
# qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.FocusHighlighting, True)
qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.OpaqueSplitterResize, False)

from main_widgets.acquisition_mgr import AcquisitionManagerWidget
from main_widgets.instrument_mgr import InstrumentManagerWidget
from main_widgets.laser_mod_mgr import LaserModManagerWidget
from main_widgets.console import ConsoleWidget
from main_widgets.menubar import MainMenuBar
from pyrpoc.gui.signals.signals import UISignals

from styles.theme_manager import ThemeManager


class MainGUI(QWidget):
    def __init__(self, ui_signals: UISignals):
        super().__init__()
        self.setWindowTitle("pyrpoc")

        self.ui_signals = ui_signals
        self.theme_mgr = ThemeManager()

        self.dock_manager = qtads.CDockManager(self)
        self.dock_manager.setStyleSheet('') # ehhhhhhhhhhhh 

        self.docks: list[qtads.CDockWidget] = []
        self.menubar = MainMenuBar(self.ui_signals, self)

        dock_acq = self.add_dock("Acquisition Manager", AcquisitionManagerWidget(), qtads.DockWidgetArea.LeftDockWidgetArea)
        dock_instr = self.add_dock("Instrument Manager", InstrumentManagerWidget(), qtads.DockWidgetArea.LeftDockWidgetArea, tab_with=dock_acq)
        dock_laser = self.add_dock("Laser Modulation Manager", LaserModManagerWidget(), qtads.DockWidgetArea.LeftDockWidgetArea, tab_with=dock_acq)
        dock_console = self.add_dock("Console", ConsoleWidget(), qtads.DockWidgetArea.BottomDockWidgetArea)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menubar)
        layout.addWidget(self.dock_manager)

        self.menubar.populate_view_menu(self.docks)
        self.menubar.populate_style_menu(self.theme_mgr.get_available_themes())

        # all signal connection goes here connecting stuff
        self.ui_signals.style_selected.connect(self.set_style)
        
        self.set_style('qdarkstyle-dark')

    def add_dock(self, title: str, widget: QWidget, area: qtads.DockWidgetArea, tab_with: qtads.CDockWidget | None = None) -> qtads.CDockWidget:
        '''
        description:
            add a widget into the class
            to be used in the initialization, and also when instruments/displays get created

        args:  
            title: the title to give the tab of the widget
            widget: the QWidget to be added
            area: the ADS area to add the widget into
            tab_with: the widget with which this should be added

        returns: the dockable widget itself

        example:
            dock_acq = self.add_dock("Acquisition Manager", AcquisitionManagerWidget(), qtads.DockWidgetArea.LeftDockWidgetArea)
            dock_instr = self.add_dock("Instrument Manager", InstrumentManagerWidget(), qtads.DockWidgetArea.LeftDockWidgetArea, tab_with=dock_acq)
        '''
        dock = qtads.CDockWidget(title)
        dock.setWidget(widget)
        self.docks.append(dock)

        if tab_with is None:
            self.dock_manager.addDockWidget(area, dock) # first dock in an area
        else:
            self.dock_manager.addDockWidgetTab(area, dock) # add as a tab with an existing dock

        return dock

    def set_style(self, theme_name: str):
        try:
            qss = self.theme_mgr.load_theme(theme_name)
            self.setStyleSheet(qss)
        except FileNotFoundError:
            print(f"Theme '{theme_name}' not found")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    ui_signals = UISignals()
    win = MainGUI(ui_signals)
    win.resize(1400, 850)
    win.show()
    sys.exit(app.exec())
