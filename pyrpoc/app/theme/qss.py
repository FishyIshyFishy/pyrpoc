"""The one stylesheet, filled from a Theme.

Fusion draws buttons, checkboxes, spin boxes and combos from the palette, so
this styles only what the palette cannot: docks, tabs, menus, scrollbars,
tooltips and the type scale. Labels opt into the scale with a ``role``
property (heading, caption).
"""

from __future__ import annotations

from string import Template

from pyrpoc.app.theme.palette import Theme

_TEMPLATE = Template("""
QToolTip {
    background: $raised; color: $text; border: 1px solid $border; padding: 4px 6px;
}

QMenuBar { background: $window; color: $text; }
QMenuBar::item { padding: 4px 10px; background: transparent; border-radius: 4px; }
QMenuBar::item:selected { background: $raised; }
QMenu {
    background: $raised; color: $text; border: 1px solid $border; padding: 4px;
}
QMenu::item { padding: 5px 22px 5px 14px; border-radius: 4px; }
QMenu::item:selected { background: $accent; color: $on_accent; }
QMenu::separator { height: 1px; background: $border; margin: 4px 6px; }

QLabel[role="heading"] { font-weight: 600; }
QLabel[role="caption"] { color: $muted; font-size: 9pt; }

QGroupBox {
    border: 1px solid $border; border-radius: 6px; margin-top: 10px; padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 8px; padding: 0 4px; color: $muted; font-weight: 600;
}

QLineEdit, QPlainTextEdit, QTextEdit {
    border: 1px solid $border; border-radius: 4px; padding: 3px 6px;
    selection-background-color: $accent; selection-color: $on_accent;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus { border: 1px solid $accent; }

QPushButton {
    border: 1px solid $border; border-radius: 4px; padding: 4px 12px; background: $raised;
}
QPushButton:hover { border-color: $accent; }
QPushButton:pressed { background: $accent; color: $on_accent; }
QPushButton:disabled { color: $disabled; background: transparent; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle { background: $border; border-radius: 4px; min-height: 24px; min-width: 24px; }
QScrollBar::handle:hover { background: $muted; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QHeaderView::section {
    background: $window; color: $muted; border: none;
    border-bottom: 1px solid $border; padding: 4px 8px; font-weight: 600;
}
QTableView, QListView, QTreeView {
    border: 1px solid $border; border-radius: 4px; gridline-color: $border;
    alternate-background-color: $window;
}
QTableView::item:hover, QListView::item:hover, QTreeView::item:hover { background: $raised; }
QTableView::item:selected, QListView::item:selected, QTreeView::item:selected {
    background: $accent; color: $on_accent;
}

QSplitter::handle { background: $border; }

ads--CDockManager { background: $window; }
ads--CDockSplitter::handle { background: $border; }
ads--CDockAreaWidget { background: $window; border: 1px solid $border; border-radius: 4px; }
ads--CDockAreaTitleBar { background: $window; border-bottom: 1px solid $border; }
ads--CDockWidget { background: $window; }
ads--CDockWidgetTab {
    background: $window; border: none; border-bottom: 2px solid transparent; padding: 0 8px;
}
ads--CDockWidgetTab QAbstractButton, ads--CDockAreaTitleBar QAbstractButton {
    background: transparent; border: none; border-radius: 4px; padding: 2px;
}
ads--CDockWidgetTab QAbstractButton:hover, ads--CDockAreaTitleBar QAbstractButton:hover {
    background: $raised;
}
ads--CDockWidgetTab QLabel { color: $muted; }
ads--CDockWidgetTab:hover QLabel { color: $text; }
ads--CDockWidgetTab[activeTab="true"] { border-bottom: 2px solid $accent; }
ads--CDockWidgetTab[activeTab="true"] QLabel { color: $text; font-weight: 600; }
""")


def build_qss(theme: Theme) -> str:
    s = theme.surfaces
    return _TEMPLATE.substitute(
        window=s.window,
        base=s.base,
        raised=s.raised,
        border=s.border,
        text=s.text,
        muted=s.muted,
        disabled=s.disabled,
        accent=theme.accent_color.name(),
        on_accent=theme.on_accent.name(),
    )
