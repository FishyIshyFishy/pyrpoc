"""Capturing and applying a session: the wiring, which belongs to the shell.

``session/`` knows the file format. It does not know what a device or a view is,
and it must not import Qt. Turning live objects into a SessionState and back is
connection logic, so it lives here.

Replaces services/session_coordinator.py.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QTimer

from pyrpoc.devices.registry import device_registry
from pyrpoc.views.registry import view_registry
from pyrpoc.session.state import DeviceState, SaveState, SessionState, ViewState
from pyrpoc.session.store import SessionStore

from . import catalog
from .app import Application


def capture(app: Application, window=None, theme_mode: str = "system") -> SessionState:
    devices = [
        DeviceState(
            key=device_registry.key_for(type(device)),
            instance_id=device.instance_id,
            user_label=device.user_label,
            state=device.export_state(),
        )
        for device in app.devices
    ]
    views = [
        ViewState(
            key=view.type_key,
            instance_id=str(getattr(view, "instance_id", "")),
            user_label=getattr(view, "user_label", None),
            visible=bool(getattr(view, "docked_visible", True)),
            state=view.export_persistence_state(),
        )
        for view in app.views
    ]
    return SessionState(
        theme_mode=theme_mode,
        devices=devices,
        views=views,
        selected_program=app.selected_program,
        params_by_program=app.params_state(),
        save=SaveState(
            name=app.save.name,
            directory=app.save.directory,
            enabled=app.save.enabled,
        ),
        ads_layout=window.save_dock_layout() if window is not None else None,
    )


def apply(state: SessionState, app: Application, window=None) -> None:
    """Rebuild runtime state from a saved session.

    Anything that cannot be recreated -- a device type that no longer exists, a
    view whose class was removed -- is skipped rather than blocking the launch.
    """
    app.clear_views()
    app.clear_devices()

    for row in state.devices:
        try:
            device = app.add_device(row.key, instance_id=row.instance_id or None,
                                    user_label=row.user_label)
            device.import_state(row.state)
        except Exception:
            continue

    for row in state.views:
        try:
            view = view_registry.get(row.key)()
            if row.instance_id:
                view.instance_id = row.instance_id
            view.user_label = row.user_label
            view.docked_visible = row.visible
            view.import_persistence_state(row.state)
            app.add_view(view)
        except Exception:
            continue

    app.load_params_state(state.params_by_program)
    app.set_save(
        name=state.save.name,
        directory=state.save.directory,
        enabled=state.save.enabled,
    )

    key = state.selected_program
    if key not in catalog.keys():
        key = catalog.CATALOG[0].key if catalog.CATALOG else None
    if key is not None:
        app.select_program(key)

    # Every dock exists now; the saved layout goes on last.
    if window is not None:
        window.restore_dock_layout(state.ads_layout)


def seed_defaults(app: Application) -> None:
    """A fresh workbench needs a card and a galvo, or nothing can run.

    v3.0's confocal required no instruments at all; v3.1's declares
    ``uses = [Galvo, DAQ]``, so without this the schema-7 reset would leave the
    user with a dead play button and no obvious cause.
    """
    if app.devices:
        return
    app.add_device("daq")
    app.add_device("galvo")


class Autosave(QObject):
    """Debounced save on any state change, plus explicit save/reset actions."""

    def __init__(self, app: Application, window, theme_controller,
                 store: SessionStore | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self.app = app
        self.window = window
        self.theme_controller = theme_controller
        self.store = store if store is not None else SessionStore()
        self.suspended = False

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.save_now)

        app.state_changed.connect(self.schedule)
        app.views_changed.connect(self.schedule)
        app.devices_changed.connect(self.schedule)

    def schedule(self) -> None:
        if self.suspended:
            return
        self.timer.start()

    def save_now(self) -> None:
        if self.suspended:
            return
        try:
            self.store.save(
                capture(self.app, self.window, self.theme_controller.get_saved_mode())
            )
        except Exception:
            pass  # a failed autosave must never interrupt an experiment

    def restore(self) -> None:
        self.suspended = True
        try:
            apply(self.store.load(), self.app, self.window)
            seed_defaults(self.app)
        finally:
            self.suspended = False
        self.save_now()

    def reset(self) -> None:
        self.suspended = True
        try:
            self.app.clear_views()
            self.app.clear_devices()
            self.app.params_by_program.clear()
            self.app.set_save(name=SaveState().name, directory="", enabled=False)
            seed_defaults(self.app)
            if catalog.CATALOG:
                self.app.select_program(catalog.CATALOG[0].key)
        finally:
            self.suspended = False
        self.save_now()
