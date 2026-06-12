from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QSettings, QStandardPaths
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from .theme_engine import (
    ThemeError,
    builtin_themes_dir,
    render_theme,
    resolve_theme_tokens,
)

settings_org = "pyrpoc"
settings_app = "pyrpoc"
settings_key_theme = "ui/theme"
default_theme = "dark-pink"

rgba_re = re.compile(r"rgba?\(([^)]*)\)")


def app_data_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(base) if base else Path(".")


def parse_color(value: str) -> QColor:
    """Parse a theme color value: #hex or rgba(r, g, b, fraction)."""
    match = rgba_re.fullmatch(value.strip())
    if match is None:
        return QColor(value.strip())
    parts = [part.strip() for part in match.group(1).split(",")]
    red, green, blue = (int(float(part)) for part in parts[:3])
    alpha = float(parts[3]) if len(parts) == 4 else 1.0
    return QColor(red, green, blue, round(alpha * 255))


def build_palette(tokens: dict[str, str]) -> QPalette:
    """Build a QPalette from theme tokens so palette(...) references in
    per-widget stylesheets follow the theme."""
    role = QPalette.ColorRole
    palette = QPalette()
    assignments = {
        role.Window: "background",
        role.WindowText: "foreground",
        role.Base: "view:background",
        role.AlternateBase: "background:alternate",
        role.Text: "foreground",
        role.Button: "background",
        role.ButtonText: "foreground",
        role.BrightText: "foreground:light",
        role.Highlight: "highlight",
        role.HighlightedText: "foreground",
        role.Link: "highlight",
        role.ToolTipBase: "background",
        role.ToolTipText: "foreground",
        role.PlaceholderText: "midtone",
        role.Light: "midtone:light",
        role.Midlight: "midtone:light",
        role.Mid: "midtone",
        role.Dark: "midtone:dark",
    }
    for color_role, token in assignments.items():
        palette.setColor(color_role, parse_color(tokens[token]))
    disabled = QPalette.ColorGroup.Disabled
    for color_role in (role.Text, role.WindowText, role.ButtonText, role.HighlightedText):
        palette.setColor(disabled, color_role, parse_color(tokens["midtone"]))
    return palette


class ThemeController:
    """Discovers themes, applies them to the QApplication, and persists the choice.

    Themes are JSON token files: the ones shipped with pyrpoc live in
    ``pyrpoc/gui/styles/themes/``, user-defined ones in ``user_themes_dir``
    (a user theme with the same file name overrides a built-in one).
    """

    def __init__(
        self,
        app: QApplication,
        user_themes_dir: Path | None = None,
        icon_cache_dir: Path | None = None,
        settings: QSettings | None = None,
    ):
        self.app = app
        self.settings = settings if settings is not None else QSettings(settings_org, settings_app)
        self.user_themes_dir = (
            user_themes_dir if user_themes_dir is not None else app_data_dir() / "themes"
        )
        self.icon_cache_dir = (
            icon_cache_dir if icon_cache_dir is not None else app_data_dir() / "theme_cache"
        )

    def theme_files(self) -> dict[str, Path]:
        files = {path.stem: path for path in sorted(builtin_themes_dir.glob("*.json"))}
        if self.user_themes_dir.is_dir():
            files.update(
                {path.stem: path for path in sorted(self.user_themes_dir.glob("*.json"))}
            )
        return files

    def available_themes(self) -> list[str]:
        return sorted(self.theme_files())

    def get_saved_theme(self) -> str:
        raw = self.settings.value(settings_key_theme, default_theme)
        return str(raw).strip()

    def apply(self, name: str, persist: bool = True) -> str:
        """Apply a theme by name; raises ThemeError if it is unknown or invalid."""
        tokens = resolve_theme_tokens(name, self.theme_files())
        stylesheet = render_theme(tokens, self.icon_cache_dir / name)
        self.app.setPalette(build_palette(tokens))
        self.app.setStyleSheet(stylesheet)
        if persist:
            self.settings.setValue(settings_key_theme, name)
        return name

    def apply_or_default(self, name: str, persist: bool = True) -> str:
        """Apply a theme by name, falling back to the default theme on any error."""
        try:
            return self.apply(name, persist=persist)
        except ThemeError:
            return self.apply(default_theme, persist=persist)

    def apply_saved_or_default(self) -> str:
        return self.apply_or_default(self.get_saved_theme(), persist=False)
