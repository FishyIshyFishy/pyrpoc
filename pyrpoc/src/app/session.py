"""The session: what a saved workbench holds, the file it lives in, and the wiring.

Configuration and layout only, as small JSON; acquired data lives in TIFF. The
state dataclasses and the store know the file format and no Qt; ``capture`` and
``apply`` convert the live application; ``Autosave`` is the Qt timer driving it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QTimer

from pyrpoc.src.panels import panel_registry
from pyrpoc.src.structs.params import ParameterError
from pyrpoc.src.structs.registries import device_registry

from . import catalog
from .application import Application
from .window import MainWindow

log = logging.getLogger(__name__)

# Bump when the shape changes. A file of another version loads as defaults,
# since there is no converter between shapes.
SCHEMA_VERSION = 8

# What malformed session JSON raises on its way into the application.
BAD_STATE = (KeyError, TypeError, ValueError, ParameterError)


@dataclass
class DeviceState:
    key: str
    instance_id: str = ""
    user_label: str | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewState:
    """One added panel. Named for the on-disk field ``views``, kept so saved
    layouts still load."""

    key: str
    instance_id: str = ""
    user_label: str | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class SaveState:
    name: str = "acquisition"
    directory: str = ""
    enabled: bool = False


@dataclass
class SessionState:
    schema_version: int = SCHEMA_VERSION
    devices: list[DeviceState] = field(default_factory=list)
    views: list[ViewState] = field(default_factory=list)
    selected_program: str | None = None
    # Every parameter block, keyed by class name: one entry per block, shared.
    param_blocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    save: SaveState = field(default_factory=SaveState)
    ads_layout: str | None = None


def default_session_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "pyrpoc" / "session.json"


class SessionStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> SessionState:
        """The saved session, or defaults if there is not a usable one. A
        corrupt file is logged rather than blocking launch."""
        if not self.path.exists():
            return SessionState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("could not read %s; starting fresh", self.path, exc_info=True)
            return SessionState()
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            return SessionState()
        try:
            return decode(raw)
        except BAD_STATE:
            log.warning("could not decode %s; starting fresh", self.path, exc_info=True)
            return SessionState()

    def save(self, state: SessionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(state), indent=2, default=str)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)


def decode_rows(rows: Any) -> list[dict[str, Any]]:
    """The entries of a saved list that are objects with a key."""
    return [row for row in rows if isinstance(row, dict) and row.get("key")]


def decode(raw: dict[str, Any]) -> SessionState:
    devices = [
        DeviceState(
            key=str(row["key"]),
            instance_id=str(row.get("instance_id", "")),
            user_label=row.get("user_label"),
            state=dict(row.get("state") or {}),
        )
        for row in decode_rows(raw.get("devices", []))
    ]
    views = [
        ViewState(
            key=str(row["key"]),
            instance_id=str(row.get("instance_id", "")),
            user_label=row.get("user_label"),
            state=dict(row.get("state") or {}),
        )
        for row in decode_rows(raw.get("views", []))
    ]
    blocks = {
        str(key): dict(value)
        for key, value in (raw.get("param_blocks") or {}).items()
        if isinstance(value, dict)
    }
    layout = raw.get("ads_layout")
    return SessionState(
        devices=devices,
        views=views,
        selected_program=raw.get("selected_program"),
        param_blocks=blocks,
        save=decode_save(raw.get("save")),
        ads_layout=layout if isinstance(layout, str) else None,
    )


def decode_save(raw: Any) -> SaveState:
    if not isinstance(raw, dict):
        return SaveState()
    default = SaveState()
    return SaveState(
        name=str(raw.get("name", default.name)),
        directory=str(raw.get("directory", default.directory)),
        enabled=bool(raw.get("enabled", default.enabled)),
    )


def capture(app: Application, window: MainWindow) -> SessionState:
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
        for panel in app.panels
    ]
    return SessionState(
        devices=devices,
        views=panels,
        selected_program=app.selected_program,
        param_blocks=app.params_state(),
        save=SaveState(name=app.save.name, directory=app.save.directory, enabled=app.save.enabled),
        ads_layout=window.save_dock_layout(),
    )


def restore_devices(state: SessionState, app: Application) -> None:
    for row in state.devices:
        try:
            device = app.add_device(
                row.key, instance_id=row.instance_id or None, user_label=row.user_label
            )
            device.import_state(row.state)
        except BAD_STATE:
            log.warning("skipping saved device %r", row.key, exc_info=True)


def restore_panels(state: SessionState, app: Application) -> None:
    for row in state.views:
        try:
            panel = panel_registry.get(row.key)(app.library)
            if row.instance_id:
                panel.instance_id = row.instance_id
            panel.user_label = row.user_label
            panel.import_persistence_state(row.state)
        except BAD_STATE:
            log.warning("skipping saved panel %r", row.key, exc_info=True)
            continue
        app.add_panel(panel)


def apply(state: SessionState, app: Application, window: MainWindow) -> None:
    """Rebuild runtime state from a saved session. A device or panel type that
    no longer exists is skipped rather than blocking launch."""
    app.clear_panels()
    app.clear_devices()
    restore_devices(state, app)
    restore_panels(state, app)
    app.load_params_state(state.param_blocks)
    app.set_save(name=state.save.name, directory=state.save.directory, enabled=state.save.enabled)
    key = state.selected_program
    app.select_program(key if key in catalog.keys() else catalog.CATALOG[0].key)
    # Every dock exists now; the saved layout goes on last.
    window.restore_dock_layout(state.ads_layout)


def seed_defaults(app: Application) -> None:
    """A fresh workbench gets a DAQ and a galvo, which the imaging programs
    need; without them the play button is dead with no obvious cause."""
    if app.devices:
        return
    app.add_device("daq")
    app.add_device("galvo")


class Autosave(QObject):
    """Debounced save on any state change, plus explicit save/reset actions."""

    def __init__(self, app: Application, window: MainWindow, store: SessionStore, parent: QObject):
        super().__init__(parent)
        self.app = app
        self.window = window
        self.store = store
        self.suspended = False

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.save_now)

        app.state_changed.connect(self.schedule)
        app.panels_changed.connect(self.schedule)
        app.devices_changed.connect(self.schedule)

    def schedule(self) -> None:
        if not self.suspended:
            self.timer.start()

    def save_now(self) -> None:
        if self.suspended:
            return
        # A file boundary: a failed autosave is logged and must never
        # interrupt an experiment.
        try:
            self.store.save(capture(self.app, self.window))
        except OSError:
            log.warning("could not save the session to %s", self.store.path, exc_info=True)

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
            self.app.clear_panels()
            self.app.clear_devices()
            self.app.blocks.clear()
            self.app.set_save(name=SaveState().name, directory="", enabled=False)
            seed_defaults(self.app)
            self.app.select_program(catalog.CATALOG[0].key)
        finally:
            self.suspended = False
        self.save_now()
