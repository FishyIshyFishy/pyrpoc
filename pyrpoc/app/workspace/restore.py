"""Converting between the live application and a ``WorkspaceState``: capture
it to save, apply it back at launch, and the defaults a first launch gets."""

from __future__ import annotations

import logging

from pyrpoc.devices import device_registry
from pyrpoc.panels import data_panel_registry
from pyrpoc.programs import program_registry

from ..gui.window import MainWindow
from ..model.application import Application
from .file import BAD_STATE, DeviceState, LibraryState, SaveState, ViewState, WorkspaceState

log = logging.getLogger(__name__)


def capture(app: Application, window: MainWindow) -> WorkspaceState:
    devices = [
        DeviceState(
            key=device_registry.key_for(type(device)),
            instance_id=device.instance_id,
            user_label=device.user_label,
            state=device.export_state(),
        )
        for device in app.devices
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
    return WorkspaceState(
        devices=devices,
        views=panels,
        selected_program=app.selected_program,
        param_blocks=app.params_state(),
        save=SaveState(name=app.save.name, directory=app.save.directory, enabled=app.save.enabled),
        library=LibraryState(auto_purge=app.library.auto_purge),
        ads_layout=window.save_dock_layout(),
    )


def restore_devices(state: WorkspaceState, app: Application) -> None:
    for row in state.devices:
        try:
            device = app.add_device(
                row.key, instance_id=row.instance_id or None, user_label=row.user_label
            )
            device.import_state(row.state)
        except BAD_STATE:
            log.warning("skipping saved device %r", row.key, exc_info=True)


def restore_panels(state: WorkspaceState, app: Application, window: MainWindow) -> None:
    for row in state.views:
        try:
            panel = data_panel_registry.get(row.key)(app.library)
            if row.instance_id:
                panel.instance_id = row.instance_id
            panel.user_label = row.user_label
            panel.import_persistence_state(row.state)
        except BAD_STATE:
            log.warning("skipping saved panel %r", row.key, exc_info=True)
            continue
        window.add_panel(panel)


def apply(state: WorkspaceState, app: Application, window: MainWindow) -> None:
    """Rebuild the live application from a saved workspace. A device or panel
    type that no longer exists is skipped rather than blocking launch."""
    window.clear_panels()
    app.clear_devices()
    restore_devices(state, app)
    restore_panels(state, app, window)
    app.load_params_state(state.param_blocks)
    app.set_save(name=state.save.name, directory=state.save.directory, enabled=state.save.enabled)
    app.library.set_auto_purge(state.library.auto_purge)
    key = state.selected_program
    app.select_program(key if key in program_registry.entries else program_registry.keys()[0])
    # Every dock exists now; the saved layout goes on last.
    window.restore_dock_layout(state.ads_layout)


def seed_defaults(app: Application) -> None:
    """A fresh workbench gets a DAQ and a galvo, which the imaging programs
    need; without them the play button is dead with no obvious cause."""
    if app.devices:
        return
    app.add_device("daq")
    app.add_device("galvo")
