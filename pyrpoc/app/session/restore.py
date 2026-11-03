"""Converting between the live application and a ``SessionState``: capture
it to save, apply it back at launch, and the defaults a first launch gets."""

from __future__ import annotations

import logging

from pyrpoc.app.application import Application
from pyrpoc.app.window import MainWindow
from pyrpoc.core.device import Device
from pyrpoc.core.panels import has_panel
from pyrpoc.plugins.devices import device_registry
from pyrpoc.plugins.panels import data_panel_registry
from pyrpoc.plugins.programs import program_registry

from .file import BAD_STATE, DeviceState, LibraryState, SaveState, SessionState, ViewState

log = logging.getLogger(__name__)


def capture(app: Application, window: MainWindow) -> SessionState:
    devices = [
        DeviceState(
            key=device_registry.key_for(type(device)),
            instance_id=device.instance_id,
            panel_open=device in app.inventory.open_panels,
            state=device.export_state(),
        )
        for device in app.inventory.devices
    ]
    panels = [
        ViewState(
            key=panel.type_key,
            instance_id=panel.instance_id,
            user_label=panel.user_label,
            state=panel.export_persistence_state(),
        )
        for panel in window.panels
    ]
    return SessionState(
        devices=devices,
        views=panels,
        selected_program=app.acquisition.selected_program,
        param_blocks=app.acquisition.params_state(),
        save=SaveState(
            name=app.acquisition.save.name,
            directory=app.acquisition.save.directory,
            enabled=app.acquisition.save.enabled,
        ),
        library=LibraryState(auto_purge=app.library.auto_purge),
        ads_layout=window.save_dock_layout(),
    )


def restore_devices(state: SessionState, app: Application) -> list[Device]:
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


def restore_panels(state: SessionState, app: Application, window: MainWindow) -> None:
    for row in state.views:
        try:
            panel = data_panel_registry.get(row.key)(app.library, app.requests)
            if row.instance_id:
                panel.instance_id = row.instance_id
            panel.user_label = row.user_label
            panel.import_persistence_state(row.state)
        except BAD_STATE:
            log.warning("skipping saved panel %r", row.key, exc_info=True)
            continue
        window.add_panel(panel)


def apply(state: SessionState, app: Application, window: MainWindow) -> None:
    """Rebuild the live application from a saved session. A device or panel
    type that no longer exists is skipped rather than blocking launch."""
    window.clear_panels()
    app.inventory.clear()
    reopen = restore_devices(state, app)
    restore_panels(state, app, window)
    for device in reopen:
        window.device_docks.open(device)
    app.acquisition.load_params_state(state.param_blocks)
    app.acquisition.set_save(
        name=state.save.name, directory=state.save.directory, enabled=state.save.enabled
    )
    app.library.set_auto_purge(state.library.auto_purge)
    key = state.selected_program
    app.acquisition.select_program(
        key if key in program_registry.entries else program_registry.keys()[0]
    )
    # Every dock exists now; the saved layout goes on last.
    window.restore_dock_layout(state.ads_layout)


def seed_defaults(app: Application) -> None:
    """A fresh workbench gets a DAQ and a galvo, which the imaging programs
    need; without them the play button is dead with no obvious cause."""
    if app.inventory.devices:
        return
    app.inventory.add("daq")
    app.inventory.add("galvo")
