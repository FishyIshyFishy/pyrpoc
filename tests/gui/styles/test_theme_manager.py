from __future__ import annotations

import json

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPalette

from pyrpoc.gui.styles.theme_engine import ThemeError, builtin_themes_dir, load_theme_tokens
from pyrpoc.gui.styles.theme_manager import (
    ThemeController,
    build_palette,
    default_theme,
    parse_color,
    settings_key_theme,
)


@pytest.fixture
def controller(qapp, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    ctl = ThemeController(
        qapp,
        user_themes_dir=tmp_path / "themes",
        icon_cache_dir=tmp_path / "cache",
        settings=settings,
    )
    yield ctl
    # the qapp fixture is session-scoped: undo global styling for other tests
    qapp.setStyleSheet("")
    qapp.setPalette(QPalette())


def test_available_themes_lists_builtins(controller):
    themes = controller.available_themes()
    assert len(themes) == 12
    assert "dark-pink" in themes
    assert "light-blue" in themes


def test_apply_sets_stylesheet_palette_and_persists(controller, qapp):
    tokens = load_theme_tokens(builtin_themes_dir / "light-blue.json")
    applied = controller.apply("light-blue")

    assert applied == "light-blue"
    assert tokens["highlight"] in qapp.styleSheet()
    assert qapp.palette().color(QPalette.ColorRole.Window) == parse_color(tokens["background"])
    assert controller.settings.value(settings_key_theme) == "light-blue"
    assert (controller.icon_cache_dir / "light-blue" / "down_arrow.svg").exists()


def test_apply_without_persist_keeps_settings(controller):
    controller.apply("dark-green", persist=False)
    assert controller.settings.value(settings_key_theme) is None


def test_apply_unknown_theme_raises(controller):
    with pytest.raises(ThemeError, match="unknown theme"):
        controller.apply("does-not-exist")


def test_apply_or_default_falls_back(controller, qapp):
    assert controller.apply_or_default("does-not-exist") == default_theme
    dark_pink = load_theme_tokens(builtin_themes_dir / "dark-pink.json")
    assert dark_pink["highlight"] in qapp.styleSheet()


def test_apply_saved_or_default_with_stale_setting(controller):
    # legacy settings stored modes like "dark"; they fall back to the default
    controller.settings.setValue(settings_key_theme, "dark")
    assert controller.apply_saved_or_default() == default_theme


def test_saved_theme_round_trip(controller):
    controller.apply("light-red")
    assert controller.get_saved_theme() == "light-red"
    assert controller.apply_saved_or_default() == "light-red"


def test_user_theme_discovered_and_applied(controller, qapp):
    controller.user_themes_dir.mkdir(parents=True)
    custom = {"base": "dark-pink", "highlight": "#123abc"}
    (controller.user_themes_dir / "my-theme.json").write_text(
        json.dumps(custom), encoding="utf-8"
    )

    assert "my-theme" in controller.available_themes()
    controller.apply("my-theme")
    assert "#123abc" in qapp.styleSheet()


def test_user_theme_overrides_builtin_with_same_name(controller):
    controller.user_themes_dir.mkdir(parents=True)
    override = controller.user_themes_dir / "dark-pink.json"
    override.write_text(json.dumps({"base": "dark-blue"}), encoding="utf-8")
    assert controller.theme_files()["dark-pink"] == override


def test_broken_user_theme_reports_error(controller):
    controller.user_themes_dir.mkdir(parents=True)
    (controller.user_themes_dir / "broken.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(ThemeError, match="failed to read"):
        controller.apply("broken")


def test_parse_color_hex_and_rgba():
    assert parse_color("#ff0000").getRgb() == (255, 0, 0, 255)
    assert parse_color("rgba(218, 60, 218, 0.25)").getRgb() == (218, 60, 218, 64)
    assert parse_color("rgb(1, 2, 3)").getRgb() == (1, 2, 3, 255)


def test_build_palette_maps_core_roles():
    tokens = load_theme_tokens(builtin_themes_dir / "dark-pink.json")
    palette = build_palette(tokens)
    assert palette.color(QPalette.ColorRole.Window) == parse_color(tokens["background"])
    assert palette.color(QPalette.ColorRole.Highlight) == parse_color(tokens["highlight"])
    assert palette.color(QPalette.ColorRole.Mid) == parse_color(tokens["midtone"])
    disabled = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
    assert disabled == parse_color(tokens["midtone"])
