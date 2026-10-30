"""The one stylesheet, filled from a Theme.

Every interactive widget gets the same four states: a visible resting border,
a hover that moves toward the accent, a pressed or checked fill in the accent,
and a dimmed disabled look. Arrows and check marks are small SVGs written per
theme, because a stylesheet can only draw them from image files. Labels opt
into the type scale with a ``role`` property (heading, caption).
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from pyrpoc.app.theme.palette import Theme

_CHEVRON_DOWN = "M2.5 4.5 6 8l3.5-3.5"
_CHEVRON_UP = "M2.5 7.5 6 4l3.5 3.5"
_CHECK = "M2.8 6.3 5 8.5 9.2 3.8"

_TEMPLATE = Template("""
QToolTip {
    background: $raised; color: $text; border: 1px solid $border; padding: 4px 6px;
}

QMenuBar { background: $window; color: $text; }
QMenuBar::item { padding: 4px 10px; background: transparent; border-radius: 4px; }
QMenuBar::item:selected { background: $hover; }
QMenu {
    background: $raised; color: $text; border: 1px solid $border; padding: 4px;
}
QMenu::item { padding: 5px 22px 5px 14px; border-radius: 4px; }
QMenu::item:selected { background: $accent; color: $on_accent; }
QMenu::item:disabled { color: $disabled; background: transparent; }
QMenu::separator { height: 1px; background: $border; margin: 4px 6px; }

QLabel[role="heading"] { font-weight: 600; }
QLabel[role="caption"] { color: $muted; font-size: 9pt; }

QGroupBox {
    border: 1px solid $border; border-radius: 6px; margin-top: 10px; padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 8px; padding: 0 4px; color: $muted; font-weight: 600;
}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: $base; border: 1px solid $border; border-radius: 4px; padding: 3px 6px;
    selection-background-color: $accent; selection-color: $on_accent;
}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover, QSpinBox:hover,
QDoubleSpinBox:hover, QComboBox:hover { border-color: $muted; }
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus { border-color: $accent; }
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled, QComboBox:disabled {
    background: $window; color: $disabled; border-color: $border;
}
QLineEdit:read-only { background: $window; }

QComboBox { padding-right: 24px; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none;
}
QComboBox::down-arrow { image: url($chevron_down); width: 10px; height: 10px; }
QComboBox::down-arrow:disabled { image: url($chevron_down_disabled); }
QComboBox QAbstractItemView {
    background: $raised; color: $text; border: 1px solid $border; outline: 0;
    selection-background-color: $accent; selection-color: $on_accent;
}

QSpinBox, QDoubleSpinBox { padding-right: 22px; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right; width: 18px;
    border: none; border-left: 1px solid $border; border-top-right-radius: 3px;
    background: transparent;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right; width: 18px;
    border: none; border-left: 1px solid $border; border-bottom-right-radius: 3px;
    background: transparent;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: $hover; }
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed { background: $accent; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url($chevron_up); width: 8px; height: 8px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url($chevron_down); width: 8px; height: 8px;
}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {
    image: url($chevron_up_disabled);
}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {
    image: url($chevron_down_disabled);
}

QPushButton {
    background: $raised; border: 1px solid $border; border-radius: 4px; padding: 4px 12px;
}
QPushButton:hover { background: $hover; border-color: $accent; }
QPushButton:pressed { background: $accent; color: $on_accent; border-color: $accent; }
QPushButton:checked { background: $hover; border-color: $accent; }
QPushButton:focus { border-color: $accent; }
QPushButton:disabled { color: $disabled; background: $window; border-color: $border; }

QToolButton {
    background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 2px;
}
QToolButton:hover { background: $hover; border-color: $border; }
QToolButton:pressed, QToolButton:checked { background: $hover; border-color: $accent; }
QToolButton:disabled { color: $disabled; }

QCheckBox, QRadioButton { spacing: 8px; }
QCheckBox:disabled, QRadioButton:disabled { color: $disabled; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px; height: 16px; border: 1px solid $muted; background: $base;
}
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 9px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: $accent; }
QCheckBox::indicator:checked { background: $accent; border-color: $accent; image: url($check); }
QRadioButton::indicator:checked {
    border-color: $accent;
    background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5,
        stop: 0 $accent, stop: 0.45 $accent, stop: 0.55 $base, stop: 1 $base);
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border-color: $border; background: $window;
}
QCheckBox::indicator:checked:disabled { background: $disabled; image: url($check); }

QSlider::groove:horizontal { height: 4px; background: $border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    background: $text; border: none;
}
QSlider::handle:horizontal:hover { background: $accent; }
QSlider::groove:horizontal:disabled, QSlider::sub-page:horizontal:disabled { background: $border; }
QSlider::handle:horizontal:disabled { background: $disabled; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle {
    background: $border; border-radius: 4px; min-height: 28px; min-width: 28px; margin: 2px;
}
QScrollBar::handle:hover { background: $muted; }
QScrollBar::handle:pressed { background: $accent; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QHeaderView::section {
    background: $window; color: $muted; border: none; border-right: 1px solid $border;
    border-bottom: 1px solid $border; padding: 4px 8px; font-weight: 600;
}
QHeaderView::section:hover { background: $hover; color: $text; }
QTableView, QListView, QTreeView {
    background: $base; border: 1px solid $border; border-radius: 4px; gridline-color: $border;
    alternate-background-color: $window; outline: 0;
}
QTableView::item:hover, QListView::item:hover, QTreeView::item:hover { background: $hover; }
QTableView::item:selected, QListView::item:selected, QTreeView::item:selected {
    background: $accent; color: $on_accent;
}
QTableView::item:selected:!active, QListView::item:selected:!active,
QTreeView::item:selected:!active { background: $hover; color: $text; }

QTableCornerButton::section { background: $window; border: none; border-bottom: 1px solid $border; }

QSplitter::handle { background: $border; }
QSplitter::handle:hover { background: $accent; }

ads--CDockManager { background: $window; }
ads--CDockSplitter::handle { background: $border; }
ads--CDockSplitter::handle:hover { background: $accent; }
ads--CDockAreaWidget { background: $window; border: 1px solid $border; border-radius: 4px; }
ads--CDockAreaTitleBar { background: $window; }
ads--CDockWidget { background: $window; }
ads--CDockWidgetTab {
    background: $base; border: 1px solid $border; border-radius: 0; margin-right: 2px;
    padding: 0 8px;
}
ads--CDockWidgetTab:hover { background: $hover; }
ads--CDockWidgetTab[activeTab="true"] {
    background: $window; border: 1px solid $accent; border-bottom: none;
}
ads--CDockWidgetTab QAbstractButton, ads--CDockAreaTitleBar QAbstractButton {
    background: transparent; border: none; border-radius: 4px; padding: 2px;
}
ads--CDockWidgetTab QAbstractButton:hover, ads--CDockAreaTitleBar QAbstractButton:hover {
    background: $hover;
}
ads--CDockWidgetTab QLabel { color: $muted; }
ads--CDockWidgetTab:hover QLabel { color: $text; }
ads--CDockWidgetTab[activeTab="true"] QLabel { color: $text; font-weight: 600; }
""")


def _write_glyph(directory: Path, name: str, path: str, color: str) -> str:
    """An SVG stroke glyph, returned as a path the stylesheet can url(). The
    color is in the file name: Qt caches images by path, so a recolored glyph
    must be a new file to show after a theme switch."""
    file = directory / f"{name}-{color.lstrip('#')}.svg"
    file.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12">'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>',
        encoding="utf-8",
    )
    return file.as_posix()


def build_qss(theme: Theme, glyph_dir: Path) -> str:
    s = theme.surfaces
    glyph_dir.mkdir(parents=True, exist_ok=True)
    return _TEMPLATE.substitute(
        window=s.window,
        base=s.base,
        raised=s.raised,
        hover=s.hover,
        border=s.border,
        text=s.text,
        muted=s.muted,
        disabled=s.disabled,
        accent=theme.accent_color.name(),
        on_accent=theme.on_accent.name(),
        chevron_down=_write_glyph(glyph_dir, "chevron-down", _CHEVRON_DOWN, s.text),
        chevron_down_disabled=_write_glyph(glyph_dir, "chevron-down", _CHEVRON_DOWN, s.disabled),
        chevron_up=_write_glyph(glyph_dir, "chevron-up", _CHEVRON_UP, s.text),
        chevron_up_disabled=_write_glyph(glyph_dir, "chevron-up", _CHEVRON_UP, s.disabled),
        check=_write_glyph(glyph_dir, "check", _CHECK, theme.on_accent.name()),
    )
