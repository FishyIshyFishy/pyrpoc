from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

class MainMenuBar(QMenuBar):
    def __init__(self, ui_signals, parent=None):
        super().__init__(parent)
        self.ui_signals = ui_signals

        self.file_menu = QMenu('&File', self)
        self.file_menu.addAction(QAction("New", self))
        self.file_menu.addAction(QAction("Open...", self))
        self.file_menu.addAction(QAction("Save", self))
        self.file_menu.addAction(QAction("Save As...", self))
        self.file_menu.addSeparator()
        self.file_menu.addAction(QAction("Exit", self))
        self.addMenu(self.file_menu)

        self.view_menu = QMenu('View', self)
        self.addMenu(self.view_menu)

        self.prefs_menu = QMenu('&Preferences', self)
        self.prefs_menu.addAction(QAction("Application Settings", self))
        self.prefs_menu.addAction(QAction("Keyboard Shortcuts", self))
        self.addMenu(self.prefs_menu)

        self.style_menu = QMenu('&Style', self)
        self.addMenu(self.style_menu)


    def populate_view_menu(self, docks: list):
        '''
        description:
            fills the view menu under the menubar with all available dockable widgets
            checking and unchecking then hides/shows them

        args: 
            docks: a list of dockable widgets

        returns: none

        example:
        '''
        self.view_menu.clear()
        for dock in docks:
            self.view_menu.addAction(dock.toggleViewAction())

    def populate_style_menu(self, themes: list[str]):
        '''
        description:
            fills the style menu under the menubar with all available styles from the theme manager 
            the main GUI handles actually setting the theme because thats a global thing anyways

        args: 
            themes: the list of themes obtained from styles/theme_manager.py

        returns: none

        example:
        '''
        self.style_menu.clear()
        for theme in themes:
            action = QAction(theme.capitalize(), self.style_menu)
            self.style_menu.addAction(action)
            action.triggered.connect(
                lambda checked, t=theme: self.ui_signals.style_selected.emit(t)
            )
