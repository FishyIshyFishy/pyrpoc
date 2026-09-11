from __future__ import annotations

from PyQt6.QtCore import QFile, QSettings, QTextStream
from PyQt6.QtWidgets import QApplication

from . import breeze_all

_SETTINGS_ORG = 'pyrpoc'
_SETTINGS_APP = 'pyrpoc'
_SETTINGS_KEY_THEME_MODE = 'ui/theme_mode'
DEFAULT_THEME = 'dark-pink'
available_breeze_themes = [
    'dark-blue', 
    'dark-blue-alt', 
    'dark-cyan', 
    'dark-cyan-alt', 
    'dark-green', 
    'dark-green-alt', 
    'dark-pink', 
    'dark-pink-alt', 
    'dark-purple', 
    'dark-purple-alt', 
    'dark-red', 
    'dark-red-alt', 
    'light-blue', 
    'light-blue-alt', 
    'light-cyan', 
    'light-cyan-alt', 
    'light-green', 
    'light-green-alt', 
    'light-pink', 
    'light-pink-alt', 
    'light-purple', 
    'light-purple-alt', 
    'light-red', 
    'light-red-alt'
]



class ThemeController:
    def __init__(self, app: QApplication):
        self.app = app
        self.settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

    def get_saved_mode(self) -> str:
        raw = self.settings.value(_SETTINGS_KEY_THEME_MODE, DEFAULT_THEME)
        theme = str(raw).strip().lower()
        if theme in available_breeze_themes:
            return theme
        return DEFAULT_THEME

    def apply_saved_or_default(self) -> str:
        return self.apply(self.get_saved_mode(), persist=False)

    def load_breeze_stylesheet(self, theme: str) -> str:
        qss_path = f':/{theme}/stylesheet.qss'

        file = QFile(qss_path)
        ok = file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text)
        if not ok:
            raise RuntimeError(f'failed to open breeze stylesheet: {qss_path}')

        stream = QTextStream(file)
        return stream.readAll()

    def apply(self, theme: str, persist: bool = True) -> str:
        normalized = theme.strip().lower()
        if normalized not in available_breeze_themes:
            normalized = DEFAULT_THEME
        selected_theme = normalized

        if persist:
            self.settings.setValue(_SETTINGS_KEY_THEME_MODE, selected_theme)

        base_qss = self.load_breeze_stylesheet(selected_theme)
        self.app.setStyleSheet(base_qss)
        return selected_theme