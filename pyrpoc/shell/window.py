"""The dock manager: three fixed panels plus one dock per added panel.

Moved from gui/main_gui.py. The ADS handling is carried over as-is, including
the object-name-before-add ordering that its save/restore lookup depends on and
the guard that stops restoreState() reshuffling from mutating view inventory.

Panels are chosen from the menu bar rather than from inside a panel. The three
fixed ones are toggled there -- unchecking hides the dock, the panel is still
there -- and the rest are added under Add and destroyed by unchecking. That
split is why a view has no hidden state any more: a view exists exactly as long
as its dock does, whether it goes away by its tab's close button or by its menu
entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PyQt6 import sip
from PyQt6.QtCore import QByteArray, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget
import PyQt6Ads as qtads

from pyrpoc.views.registry import view_registry

from .app import Application
from .data_panel import DataPanel
from .devices_panel import DevicesPanel
from .launcher import LauncherPanel
from .menubar import MainMenuBar
from .theme.manager import ThemeController

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
    # "dock.views" is a historical name, kept because ADS restores by object
    # name: rename it and every saved layout stops placing this dock, which
    # reads as the panel having vanished. The title is what the user sees.
    DockSpec(DockKey.DATA, "Data Library", "dock.views"),
]


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
        # The same guard for teardown: ADS closes every dock on the way out,
        # and closing a view's dock now deletes the view. Without this the
        # session would be saved and then emptied.
        self.shutting_down = False
        self.dock_by_key: dict[DockKey, qtads.CDockWidget] = {}
        self.view_docks: dict[QWidget, qtads.CDockWidget] = {}
        self.view_actions: dict[QWidget, QAction] = {}
        #: Which instance of its type each view is, so two of the same kind can
        #: be told apart. Held here rather than on the view: what a panel is
        #: called among its siblings is the window's business, not the
        #: renderer's.
        self.view_ordinals: dict[QWidget, int] = {}

        self.menubar = MainMenuBar(self)
        self.menubar.populate_add_menu(
            [(key, view_registry.get(key).display_name) for key in view_registry.keys()]
        )
        self.menubar.panel_requested.connect(self.add_panel_of_type)
        self.build_panels()

        self.app.views_changed.connect(self.sync_view_docks)

        layout = QVBoxLayout(self)
        layout.setMenuBar(self.menubar)
        layout.addWidget(self.dock_manager)

        self.autosave = None
        self.refresh_panels_menu()
        self.menubar.populate_style_menu(self.theme_controller.get_saved_mode())
        self.menubar.style_selected.connect(self.set_style)

    def bind_session(self, autosave) -> None:
        """Connect the close event to session persistence."""
        self.autosave = autosave
        self.closing.connect(autosave.save_now)

    # -- panels -------------------------------------------------------------- #

    def build_panels(self) -> None:
        self.data_panel = DataPanel(self.app)
        widgets = {
            DockKey.ACQUISITION: LauncherPanel(self.app),
            DockKey.DEVICES: DevicesPanel(self.app),
            DockKey.DATA: self.data_panel,
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

    def add_panel_of_type(self, key: str) -> None:
        """Add one, from the menu. The dock follows from views_changed."""
        try:
            view = view_registry.get(key)()
        except Exception:
            return
        self.app.add_view(view)

    # -- view docks ---------------------------------------------------------- #

    def sync_view_docks(self) -> None:
        for view in list(self.view_docks):
            if view not in self.app.views:
                self.remove_view_dock(view)
        for view in self.app.views:
            if view not in self.view_docks:
                self.add_view_dock(view)
        self.refresh_panels_menu()

    def assign_ordinal(self, view: QWidget) -> int:
        """The lowest number this type has free.

        Lowest free rather than a running count, so the numbers stay short
        after panels have been opened and closed a few times. Assigned once and
        kept: a dock that renumbered itself because an earlier one was closed
        would be renaming the panel someone is working in.
        """
        type_key = getattr(view, "type_key", "")
        taken = {
            number
            for other, number in self.view_ordinals.items()
            if getattr(other, "type_key", "") == type_key
        }
        number = 1
        while number in taken:
            number += 1
        self.view_ordinals[view] = number
        return number

    def view_title(self, view: QWidget) -> str:
        label = getattr(view, "user_label", None)
        if label:
            return label
        name = getattr(view, "display_name", "View")
        number = self.view_ordinals.get(view, 1)
        return name if number == 1 else f"{name} {number}"

    def view_object_name(self, view: QWidget) -> str:
        raw = str(getattr(view, "instance_id", "") or id(view))
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
        return f"dock.view.{safe}"

    def add_view_dock(self, view: QWidget) -> None:
        self.assign_ordinal(view)
        dock = qtads.CDockWidget(self.view_title(view))
        dock.setObjectName(self.view_object_name(view))
        dock.setWidget(view)
        try:
            self.dock_manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, dock)
        except Exception:
            self.view_ordinals.pop(view, None)
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
        self.view_ordinals.pop(view, None)

        if action is not None and not sip.isdeleted(action):
            try:
                action.toggled.disconnect()
            except Exception:
                pass
            try:
                self.menubar.panels_menu.removeAction(action)
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

    def discard_view(self, view: QWidget) -> None:
        """Drop an added panel, from whichever gesture asked for it.

        Deferred, because both callers are inside a signal from the thing this
        is about to delete -- the dock's own close, or its menu action being
        unchecked. Idempotent, because a close button press unchecks nothing
        and an uncheck closes nothing: whichever arrives second finds the view
        already gone.
        """
        if self.restoring_layout or self.shutting_down:
            return
        if view not in self.app.views:
            return
        QTimer.singleShot(0, lambda v=view: self.app.remove_view(v))

    def on_view_toggled(self, view: QWidget, visible: bool) -> None:
        """Unchecking an added panel deletes it. There is nothing to re-check.

        The fixed three hide instead, and that difference is the whole reason
        the menu draws a rule between them.
        """
        if visible:
            return
        self.discard_view(view)

    def on_view_dock_closed(self, view: QWidget) -> None:
        """The tab's close button means what unchecking it means.

        The data it was showing is untouched: that lives in the library, and
        adding the panel back binds to it again.
        """
        self.discard_view(view)

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
        self.refresh_panels_menu()

    def refresh_panels_menu(self) -> None:
        for view in list(self.view_actions):
            action = self.view_actions.get(view)
            if action is None or sip.isdeleted(action):
                self.view_actions.pop(view, None)
        self.menubar.populate_panels_menu(
            list(self.dock_by_key.values()), list(self.view_actions.values())
        )

    def set_style(self, theme_mode: str) -> None:
        self.menubar.set_active_style(self.theme_controller.apply(theme_mode))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutting_down = True
        self.closing.emit()
        super().closeEvent(event)
