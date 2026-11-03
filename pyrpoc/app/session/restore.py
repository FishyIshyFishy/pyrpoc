"""Converting between the live application and a ``SessionState``: capture it
to save, apply it back at launch, and the devices a first launch gets."""

from __future__ import annotations

import logging

from pyrpoc.app.application import Application
from pyrpoc.app.window import MainWindow
from pyrpoc.core.device import Device
from pyrpoc.core.panels import has_panel
from pyrpoc.plugins import data_panel_registry, device_registry

from .file import (
    BAD_STATE,
    DeviceState,
    LibraryState,
    SaveState,
    ScreenState,
    SessionState,
    ViewState,
    WorkingState,
)

log = logging.getLogger(__name__)


def capture(app: Application, window: MainWindow) -> SessionState:
    return SessionState(working=capture_working(app), screen=capture_screen(window))


def capture_working(app: Application) -> WorkingState:
    devices = [
        DeviceState(
            key=device_registry.key_for(type(device)),
            instance_id=device.instance_id,
            panel_open=device in app.inventory.open_panels,
            state=device.export_state(),
        )
        for device in app.inventory.devices
    ]
    save = app.runs.save
    return WorkingState(
        devices=devices,
        program_inputs=app.inputs.to_dict(),
        save=SaveState(name=save.name, directory=save.directory, enabled=save.enabled),
        library=LibraryState(auto_purge=app.library.auto_purge),
    )


def capture_screen(window: MainWindow) -> ScreenState:
    views = [
        ViewState(
            key=panel.type_key,
            instance_id=panel.instance_id,
            user_label=panel.user_label,
            state=panel.export_persistence_state(),
        )
        for panel in window.panels
    ]
    return ScreenState(
        views=views, panels=window.fixed_panel_states(), ads_layout=window.save_dock_layout()
    )


def restore_devices(state: WorkingState, app: Application) -> list[Device]:
    """Add the saved devices, returning the ones whose panel was open."""
    reopen: list[Device] = []
    for row in state.devices:
        try:
            device = app.inventory.add(row.key, instance_id=row.instance_id or None)
            device.import_state(row.state)
        except BAD_STATE:
            log.warning("skipping saved device %r", row.key, exc_info=True)
            continue
        if row.panel_open and has_panel(device):
            reopen.append(device)
    return reopen


def restore_panels(state: ScreenState, app: Application, window: MainWindow) -> None:
    for row in state.views:
        try:
            panel = data_panel_registry.get(row.key)(app.library, app.inputs)
            if row.instance_id:
                panel.instance_id = row.instance_id
            panel.user_label = row.user_label
            panel.import_persistence_state(row.state)
        except BAD_STATE:
            log.warning("skipping saved panel %r", row.key, exc_info=True)
            continue
        window.add_panel(panel)


def apply(state: SessionState, app: Application, window: MainWindow) -> None:
    """Rebuild the live application from a saved session, skipping any device or panel type
    that no longer exists."""
    working, screen = state.working, state.screen
    window.clear_panels()
    app.inventory.clear()
    reopen = restore_devices(working, app)
    # Before the panels, since a panel's saved target names a slot of an input's value.
    app.inputs.clear()
    app.inputs.load_dict(working.program_inputs)
    restore_panels(screen, app, window)
    for device in reopen:
        window.device_docks.open(device)
    app.runs.set_save(
        name=working.save.name, directory=working.save.directory, enabled=working.save.enabled
    )
    app.library.set_auto_purge(working.library.auto_purge)
    window.restore_fixed_panel_states(screen.panels)
    # Every dock exists now; the saved layout goes on last.
    window.restore_dock_layout(screen.ads_layout)


def seed_defaults(app: Application) -> None:
    """A fresh workbench gets a DAQ and a galvo, which the imaging programs
    need; without them the play button is dead with no obvious cause."""
    if app.inventory.devices:
        return
    app.inventory.add("daq")
    app.inventory.add("galvo")
