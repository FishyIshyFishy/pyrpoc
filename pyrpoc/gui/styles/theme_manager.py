from __future__ import annotations

from typing import Literal

from PyQt6.QtCore import QFile, QSettings, QTextStream
from PyQt6.QtWidgets import QApplication

from . import breeze_all

ThemeMode = Literal['dark', 'light']

_SETTINGS_ORG = 'pyrpoc'
_SETTINGS_APP = 'pyrpoc'
_SETTINGS_KEY_THEME_MODE = 'ui/theme_mode'
AVAILABLE_BREEZE_THEMES = [
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
    AVAILABLE_MODES: tuple[ThemeMode, ...] = ('dark', 'light')

    def __init__(self, app: QApplication):
        self.app = app
        self.settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

    def get_available_modes(self) -> list[ThemeMode]:
        return list(self.AVAILABLE_MODES)

    def get_saved_mode(self) -> ThemeMode:
        raw = self.settings.value(_SETTINGS_KEY_THEME_MODE, 'dark')
        mode = str(raw).strip().lower()
        if mode in self.AVAILABLE_MODES:
            return mode
        return 'dark'

    def apply_saved_or_default(self) -> ThemeMode:
        return self.apply(self.get_saved_mode(), persist=False)

    def _load_breeze_stylesheet(self, mode: ThemeMode) -> str:
        qss_path = ':/dark-pink/stylesheet.qss'

        file = QFile(qss_path)
        ok = file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text)
        if not ok:
            raise RuntimeError(f'failed to open breeze stylesheet: {qss_path}')

        stream = QTextStream(file)
        return stream.readAll()

    def apply(self, mode: str, persist: bool = True) -> ThemeMode:
        normalized = mode.strip().lower()
        if normalized not in self.AVAILABLE_MODES:
            normalized = 'dark'
        selected_mode: ThemeMode = normalized

        if persist:
            self.settings.setValue(_SETTINGS_KEY_THEME_MODE, selected_mode)

        base_qss = self._load_breeze_stylesheet(selected_mode)
        self.app.setStyleSheet(base_qss)
        return selected_mode