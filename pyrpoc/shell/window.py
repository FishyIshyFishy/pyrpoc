"""The dock manager: three fixed panels plus one dock per open view.

Moved from gui/main_gui.py. The ADS handling is carried over as-is, including
the object-name-before-add ordering that its save/restore lookup depends on and
the guard that stops restoreState() reshuffling from mutating view inventory.

Data and views used to be a panel each. They are one dock now, split, because
choosing what a display shows means looking at what has been acquired -- and as
tabs, only one of the two was ever on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PyQt6 import sip
from PyQt6.QtCore import QByteArray, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget
import PyQt6Ads as qtads

from .app import Application
from .data_panel import DataPanel
from .devices_panel import DevicesPanel
from .launcher import LauncherPanel
from .menubar import MainMenuBar
from .theme.manager import ThemeController
from .views_panel import ViewsPanel

qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.DisableTabTextEliding, True)
qtads.CDockManager.setConfigFlag(qtads.CDockManager.eConfigFlag.OpaqueSplitterResize, False)


class DockKey(str, Enum):
    ACQUISITION = "acquisition"
    DEVICES = "devices"
    DATA = "data"


@dataclass(frozen=True)
class DockSpec:
    key: DockKey
    title: str
    object_name: str


PANELS = [
    DockSpec(DockKey.ACQUISITION, "Acquisition", "dock.acquisition"),
    DockSpec(DockKey.DEVICES, "Devices", "dock.devices"),
    # The object name is the one the Views dock had. ADS restores by object
    # name, so a session saved before the merge puts this dock where its Views
    # tab was rather than dropping it; the vacated Library dock is skipped.
    DockSpec(DockKey.DATA, "Data & Views", "dock.views"),
]


def titled(title: str, widget: QWidget) -> QWidget:
    """A panel under a heading, for the dock that holds more than one.

    The dock tab used to name what was inside it. Two panels in one dock need
    to say so themselves.
    """
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 6, 0, 0)
    layout.setSpacing(0)
    heading = QLabel(title, box)
    heading.setStyleSheet("font-weight: 600; padding: 0 8px;")
    layout.addWidget(heading)
    layout.addWidget(widget, 1)
    return box


def stacked(*sections: tuple[str, QWidget]) -> QWidget:
    """Several titled panels in one dock, the split between them draggable.

    How much room a view list needs depends on how many views are open, so the
    division is the user's to make rather than a fixed ratio.
    """
    splitter = QSplitter(Qt.Orientation.Vertical)
    splitter.setChildrenCollapsible(False)
    for title, widget in sections:
        splitter.addWidget(titled(title, widget))
    for index in range(splitter.count()):
        splitter.setStretchFactor(index, 1)
    return splitter


class MainWindow(QWidget):
    closing = pyqtSignal()

    def __init__(self, app: Application, theme_controller: ThemeController):
        super().__init__()
        self.setWindowTitle("pyrpoc")
        self.app = app
        self.theme_controller = theme_controller

        self.dock_manager = qtads.CDockManager(self)
        self.dock_manager.setStyleSheet("")
        # Guards the dock close/toggle handlers while restoreState() reshuffles
        # docks, so ADS visibility changes during restore don't mutate inventory.
        self.restoring_layout = False
        self.dock_by_key: dict[DockKey, qtads.CDockWidget] = {}
        self.view_docks: dict[QWidget, qtads.CDockWidget] = {}
        self.view_actions: dict[QWidget, QAction] = {}

        self.menubar = MainMenuBar(self)
        self.build_panels()

        self.app.views_changed.connect(self.sync_view_docks)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menubar)
        layout.addWidget(self.dock_manager)

        self.autosave = None
        self.refresh_view_menu()
        self.menubar.populate_style_menu(self.theme_controller.get_saved_mode())
        self.menubar.style_selected.connect(self.set_style)

    def bind_session(self, autosave) -> None:
        """Connect the File menu and the close event to session persistence."""
        self.autosave = autosave
        self.menubar.new_requested.connect(autosave.reset)
        self.menubar.open_requested.connect(autosave.restore)
        self.menubar.save_requested.connect(autosave.save_now)
        self.menubar.save_as_requested.connect(autosave.save_now)
        self.closing.connect(autosave.save_now)

    # -- panels -------------------------------------------------------------- #

    def build_panels(self) -> None:
        self.data_panel = DataPanel(self.app)
        self.views_panel = ViewsPanel(self.app)
        widgets = {
            DockKey.ACQUISITION: LauncherPanel(self.app),
            DockKey.DEVICES: DevicesPanel(self.app),
            DockKey.DATA: stacked(("Data", self.data_panel), ("Views", self.views_panel)),
        }
        first: qtads.CDockWidget | None = None
        for spec in PANELS:
            dock = self.add_dock(spec.title, widgets[spec.key], spec.object_name, tab_with=first)
            self.dock_by_key[spec.key] = dock
            if first is None:
                first = dock

    def add_dock(
        self,
        title: str,
        widget: QWidget,
        object_name: str,
        tab_with: qtads.CDockWidget | None = None,
    ) -> qtads.CDockWidget:
        dock = qtads.CDockWidget(title)
        # ADS keys its save/restore lookup map by the object name at
        # addDockWidget time (falling back to the title if unset), so this MUST
        # precede the add.
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        area = qtads.DockWidgetArea.LeftDockWidgetArea
        if tab_with is None:
            self.dock_manager.addDockWidget(area, dock)
        else:
            self.dock_manager.addDockWidgetTab(area, dock)
        return dock

    # -- view docks ---------------------------------------------------------- #

    def sync_view_docks(self) -> None:
        for view in list(self.view_docks):
            if view not in self.app.views:
                self.remove_view_dock(view)
        for view in self.app.views:
            if view not in self.view_docks:
                self.add_view_dock(view)
        self.refresh_view_menu()

    def view_title(self, view: QWidget) -> str:
        return getattr(view, "user_label", None) or getattr(view, "display_name", "View")

    def view_object_name(self, view: QWidget) -> str:
        raw = str(getattr(view, "instance_id", "") or id(view))
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
        return f"dock.view.{safe}"

    def add_view_dock(self, view: QWidget) -> None:
        dock = qtads.CDockWidget(self.view_title(view))
        dock.setObjectName(self.view_object_name(view))
        dock.setWidget(view)
        try:
            self.dock_manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, dock)
        except Exception:
            dock.deleteLater()
            return
        self.view_docks[view] = dock

        action = QAction(self.view_title(view), self)
        action.setCheckable(True)
        action.setChecked(True)
        action.toggled.connect(lambda checked, v=view: self.on_view_toggled(v, checked))
        self.view_actions[view] = action

        if hasattr(dock, "closed"):
            dock.closed.connect(lambda *_args, v=view: self.on_view_dock_closed(v))

    def remove_view_dock(self, view: QWidget) -> None:
        dock = self.view_docks.pop(view, None)
        action = self.view_actions.pop(view, None)

        if action is not None and not sip.isdeleted(action):
            try:
                action.toggled.disconnect()
            except Exception:
                pass
            try:
                self.menubar.view_menu.removeAction(action)
            except Exception:
                pass
            action.setParent(None)
            action.deleteLater()

        if dock is not None and not sip.isdeleted(dock):
            try:
                self.dock_manager.removeDockWidget(dock)
            except Exception:
                pass
            try:
                detached = dock.takeWidget()
                if detached is not None:
                    detached.setParent(None)
            except Exception:
                pass
            dock.deleteLater()

    def on_view_toggled(self, view: QWidget, visible: bool) -> None:
        if self.restoring_layout:
            return
        dock = self.view_docks.get(view)
        if dock is not None and not sip.isdeleted(dock):
            dock.toggleView(visible)
        if hasattr(view, "docked_visible"):
            view.docked_visible = visible

    def on_view_dock_closed(self, view: QWidget) -> None:
        """Closing a view's dock no longer destroys its data.

        The dataset stays in the library; reopening the view rebinds to it.
        """
        if self.restoring_layout:
            return
        if hasattr(view, "docked_visible"):
            view.docked_visible = False
        action = self.view_actions.get(view)
        if action is not None and not sip.isdeleted(action) and action.isChecked():
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)

    # -- layout, menu, theme -------------------------------------------------- #

    def save_dock_layout(self) -> str | None:
        try:
            state = self.dock_manager.saveState()
        except Exception:
            return None
        if state.isEmpty():
            return None
        return state.toBase64().data().decode("ascii")

    def restore_dock_layout(self, layout_base64: str | None) -> None:
        if not layout_base64:
            return
        data = QByteArray.fromBase64(layout_base64.encode("ascii"))
        if data.isEmpty():
            return
        self.restoring_layout = True
        try:
            self.dock_manager.restoreState(data)
        except Exception:
            pass
        finally:
            self.restoring_layout = False
        self.refresh_view_menu()

    def refresh_view_menu(self) -> None:
        for view in list(self.view_actions):
            action = self.view_actions.get(view)
            if action is None or sip.isdeleted(action):
                self.view_actions.pop(view, None)
        self.menubar.populate_view_menu(
            list(self.dock_by_key.values()), list(self.view_actions.values())
        )

    def set_style(self, theme_mode: str) -> None:
        self.menubar.set_active_style(self.theme_controller.apply(theme_mode))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closing.emit()
        super().closeEvent(event)
