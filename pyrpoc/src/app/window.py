"""The dock manager: three fixed panels plus one dock per added panel.

Moved from gui/main_gui.py. The ADS handling is carried over as-is, including
the object-name-before-add ordering that its save/restore lookup depends on and
the guard that stops restoreState() reshuffling from mutating panel inventory.

Panels are chosen from the menu bar rather than from inside a panel. The three
fixed ones are toggled there -- unchecking hides the dock, the panel is still
there -- and the rest are added under Add and destroyed by unchecking. That
split is why an added panel has no hidden state any more: it exists exactly as
long as its dock does, whether it goes away by its tab's close button or by its
menu entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PyQt6 import sip
from PyQt6.QtCore import QByteArray, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import QVBoxLayout, QWidget
import PyQt6Ads as qtads

from pyrpoc.src.panels import AcquisitionPanel, DataLibraryPanel, DevicesPanel, panel_registry

from .application import Application
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
        # and closing a panel's dock now deletes the panel. Without this the
        # session would be saved and then emptied.
        self.shutting_down = False
        self.dock_by_key: dict[DockKey, qtads.CDockWidget] = {}
        self.panel_docks: dict[QWidget, qtads.CDockWidget] = {}
        self.panel_actions: dict[QWidget, QAction] = {}
        #: Which instance of its type each panel is, so two of the same kind
        #: can be told apart. Held here rather than on the panel: what a panel
        #: is called among its siblings is the window's business, not the
        #: renderer's.
        self.panel_ordinals: dict[QWidget, int] = {}

        self.menubar = MainMenuBar(self)
        self.menubar.populate_add_menu(
            [(key, panel_registry.get(key).display_name) for key in panel_registry.keys()]
        )
        self.menubar.panel_requested.connect(self.add_panel_of_type)
        self.build_panels()

        self.app.panels_changed.connect(self.sync_panel_docks)

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
        self.data_library_panel = DataLibraryPanel(self.app)
        widgets = {
            DockKey.ACQUISITION: AcquisitionPanel(self.app),
            DockKey.DEVICES: DevicesPanel(self.app),
            DockKey.DATA: self.data_library_panel,
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
        """Add one, from the menu. The dock follows from panels_changed."""
        try:
            panel = panel_registry.get(key)()
        except Exception:
            return
        self.app.add_panel(panel)

    # -- panel docks ----------------------------------------------------------- #

    def sync_panel_docks(self) -> None:
        for panel in list(self.panel_docks):
            if panel not in self.app.panels:
                self.remove_panel_dock(panel)
        for panel in self.app.panels:
            if panel not in self.panel_docks:
                self.add_panel_dock(panel)
        self.refresh_panels_menu()

    def assign_ordinal(self, panel: QWidget) -> int:
        """The lowest number this type has free.

        Lowest free rather than a running count, so the numbers stay short
        after panels have been opened and closed a few times. Assigned once and
        kept: a dock that renumbered itself because an earlier one was closed
        would be renaming the panel someone is working in.
        """
        type_key = getattr(panel, "type_key", "")
        taken = {
            number
            for other, number in self.panel_ordinals.items()
            if getattr(other, "type_key", "") == type_key
        }
        number = 1
        while number in taken:
            number += 1
        self.panel_ordinals[panel] = number
        return number

    def panel_title(self, panel: QWidget) -> str:
        label = getattr(panel, "user_label", None)
        if label:
            return label
        name = getattr(panel, "display_name", "Panel")
        number = self.panel_ordinals.get(panel, 1)
        return name if number == 1 else f"{name} {number}"

    def panel_object_name(self, panel: QWidget) -> str:
        raw = str(getattr(panel, "instance_id", "") or id(panel))
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
        # "dock.view." rather than "dock.panel.": these prefix every added
        # panel's object name in a saved layout blob, same reason "dock.views"
        # above is kept -- renaming it strands every already-saved layout.
        return f"dock.view.{safe}"

    def add_panel_dock(self, panel: QWidget) -> None:
        self.assign_ordinal(panel)
        dock = qtads.CDockWidget(self.panel_title(panel))
        dock.setObjectName(self.panel_object_name(panel))
        dock.setWidget(panel)
        try:
            self.dock_manager.addDockWidget(qtads.DockWidgetArea.RightDockWidgetArea, dock)
        except Exception:
            self.panel_ordinals.pop(panel, None)
            dock.deleteLater()
            return
        self.panel_docks[panel] = dock

        action = QAction(self.panel_title(panel), self)
        action.setCheckable(True)
        action.setChecked(True)
        action.toggled.connect(lambda checked, p=panel: self.on_panel_toggled(p, checked))
        self.panel_actions[panel] = action

        if hasattr(dock, "closed"):
            dock.closed.connect(lambda *_args, p=panel: self.on_panel_dock_closed(p))

    def remove_panel_dock(self, panel: QWidget) -> None:
        dock = self.panel_docks.pop(panel, None)
        action = self.panel_actions.pop(panel, None)
        self.panel_ordinals.pop(panel, None)

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

    def discard_panel(self, panel: QWidget) -> None:
        """Drop an added panel, from whichever gesture asked for it.

        Deferred, because both callers are inside a signal from the thing this
        is about to delete -- the dock's own close, or its menu action being
        unchecked. Idempotent, because a close button press unchecks nothing
        and an uncheck closes nothing: whichever arrives second finds the panel
        already gone.
        """
        if self.restoring_layout or self.shutting_down:
            return
        if panel not in self.app.panels:
            return
        QTimer.singleShot(0, lambda p=panel: self.app.remove_panel(p))

    def on_panel_toggled(self, panel: QWidget, visible: bool) -> None:
        """Unchecking an added panel deletes it. There is nothing to re-check.

        The fixed three hide instead, and that difference is the whole reason
        the menu draws a rule between them.
        """
        if visible:
            return
        self.discard_panel(panel)

    def on_panel_dock_closed(self, panel: QWidget) -> None:
        """The tab's close button means what unchecking it means.

        The data it was showing is untouched: that lives in the library, and
        adding the panel back binds to it again.
        """
        self.discard_panel(panel)

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
        for panel in list(self.panel_actions):
            action = self.panel_actions.get(panel)
            if action is None or sip.isdeleted(action):
                self.panel_actions.pop(panel, None)
        self.menubar.populate_panels_menu(
            list(self.dock_by_key.values()), list(self.panel_actions.values())
        )

    def set_style(self, theme_mode: str) -> None:
        self.menubar.set_active_style(self.theme_controller.apply(theme_mode))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutting_down = True
        self.closing.emit()
        super().closeEvent(event)
