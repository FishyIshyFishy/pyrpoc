"""The session: what a saved workbench holds, the file it lives in, and the wiring.

Configuration only: what exists, how it is configured, and the layout -- small,
JSON, and there so the workbench comes back on relaunch. Acquired data is
deliberately not here. It is large, lives in TIFF, and exists because it is the
experimental result; merge them and the session file starts trying to hold
arrays.

Three parts, top to bottom. The state dataclasses and the store know the file
format and nothing else -- no Qt, and the path is supplied rather than looked up,
so they stay testable headless. ``capture``/``apply`` turn the live application
into a ``SessionState`` and back, which is connection logic. ``Autosave`` is the
Qt timer that drives it.

``SessionState.views``/``ViewState`` keep their name from before panels/ was
panels/: it is the on-disk field name, and renaming it would reset every
saved session's added-panel layout for no functional gain. It holds the
added panels only -- image_2d, overlay, mask_editor, spectrum -- the same set
``Application.panels`` does; the three fixed panels are not saved here at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QTimer

from pyrpoc.src.structs.panel import panel_registry
from pyrpoc.src.structs.registries import device_registry

from . import catalog
from .application import Application


# --------------------------------------------------------------------------- #
# What a saved session holds                                                   #
# --------------------------------------------------------------------------- #

#: Bumped from 7. Parameters are no longer stored per program: a block is
#: shared by every modality that declares it, so there is one flat state dict
#: keyed by block class name instead of a nested dict keyed by program. A v7
#: file's ``params_by_program`` has no single answer to map onto -- three
#: programs could each hold a different ScanGroup -- so there is no converter
#: and a v7 session loads as defaults, once.
SCHEMA_VERSION = 8


@dataclass
class DeviceState:
    key: str
    instance_id: str = ""
    user_label: str | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewState:
    key: str
    instance_id: str = ""
    user_label: str | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class SaveState:
    """What acquisitions are called and where they are written.

    One block, not one per program: saving is not something a program decides.
    """

    name: str = "acquisition"
    directory: str = ""
    enabled: bool = False


@dataclass
class SessionState:
    schema_version: int = SCHEMA_VERSION
    devices: list[DeviceState] = field(default_factory=list)
    views: list[ViewState] = field(default_factory=list)
    selected_program: str | None = None
    #: Every parameter block, keyed by class name. The state dict: one entry
    #: per block, not one per program, because a block is shared.
    param_blocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    save: SaveState = field(default_factory=SaveState)
    ads_layout: str | None = None

    def is_empty(self) -> bool:
        return not self.devices and not self.views and not self.param_blocks


# --------------------------------------------------------------------------- #
# Reading and writing the session file                                         #
# --------------------------------------------------------------------------- #


def default_session_path() -> Path:
    """Where the session lives when the caller does not say."""
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "pyrpoc" / "session.json"


class SessionStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else default_session_path()
        self.last_load_error: str | None = None

    # -- reading ------------------------------------------------------------ #

    def load(self) -> SessionState:
        """Return the saved session, or defaults if there is not a usable one.

        A version mismatch is not an error to report at the user: parameter
        storage has changed shape twice and an old file simply resets.
        Anything else that goes wrong is recorded in ``last_load_error``.
        """
        self.last_load_error = None
        if not self.path.exists():
            return SessionState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not block launch
            self.last_load_error = f"could not read {self.path}: {exc}"
            return SessionState()

        if not isinstance(raw, dict):
            self.last_load_error = f"{self.path} does not contain a session"
            return SessionState()
        if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
            return SessionState()

        try:
            return decode(raw)
        except Exception as exc:  # noqa: BLE001
            self.last_load_error = f"could not decode {self.path}: {exc}"
            return SessionState()

    # -- writing ------------------------------------------------------------ #

    def save(self, state: SessionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(state), indent=2, default=str)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)


def decode(raw: dict[str, Any]) -> SessionState:
    devices = [
        DeviceState(
            key=str(row["key"]),
            instance_id=str(row.get("instance_id", "")),
            user_label=row.get("user_label"),
            state=dict(row.get("state") or {}),
        )
        for row in raw.get("devices", [])
        if isinstance(row, dict) and row.get("key")
    ]
    views = [
        ViewState(
            key=str(row["key"]),
            instance_id=str(row.get("instance_id", "")),
            user_label=row.get("user_label"),
            state=dict(row.get("state") or {}),
        )
        for row in raw.get("views", [])
        if isinstance(row, dict) and row.get("key")
    ]
    blocks = {
        str(key): dict(value)
        for key, value in (raw.get("param_blocks") or {}).items()
        if isinstance(value, dict)
    }
    layout = raw.get("ads_layout")
    return SessionState(
        schema_version=SCHEMA_VERSION,
        devices=devices,
        views=views,
        selected_program=raw.get("selected_program"),
        param_blocks=blocks,
        save=decode_save(raw.get("save")),
        ads_layout=layout if isinstance(layout, str) else None,
    )


def decode_save(raw: Any) -> SaveState:
    """The save block, or defaults. Absent in every file written before it."""
    if not isinstance(raw, dict):
        return SaveState()
    default = SaveState()
    return SaveState(
        name=str(raw.get("name", default.name)),
        directory=str(raw.get("directory", default.directory)),
        enabled=bool(raw.get("enabled", default.enabled)),
    )


# --------------------------------------------------------------------------- #
# Live application <-> session                                                 #
# --------------------------------------------------------------------------- #


def capture(app: Application, window=None) -> SessionState:
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
            instance_id=str(getattr(panel, "instance_id", "")),
            user_label=getattr(panel, "user_label", None),
            state=panel.export_persistence_state(),
        )
        for panel in app.panels
    ]
    return SessionState(
        devices=devices,
        views=panels,
        selected_program=app.selected_program,
        param_blocks=app.params_state(),
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
    panel whose class was removed -- is skipped rather than blocking the launch.
    """
    app.clear_panels()
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
            panel = panel_registry.get(row.key)()
            if row.instance_id:
                panel.instance_id = row.instance_id
            panel.user_label = row.user_label
            panel.import_persistence_state(row.state)
            app.add_panel(panel)
        except Exception:
            continue

    app.load_params_state(state.param_blocks)
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

    def __init__(self, app: Application, window,
                 store: SessionStore | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self.app = app
        self.window = window
        self.store = store if store is not None else SessionStore()
        self.suspended = False

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.save_now)

        app.state_changed.connect(self.schedule)
        app.panels_changed.connect(self.schedule)
        app.devices_changed.connect(self.schedule)

    def schedule(self) -> None:
        if self.suspended:
            return
        self.timer.start()

    def save_now(self) -> None:
        if self.suspended:
            return
        try:
            self.store.save(capture(self.app, self.window))
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
            self.app.clear_panels()
            self.app.clear_devices()
            self.app.blocks.clear()
            self.app.set_save(name=SaveState().name, directory="", enabled=False)
            seed_defaults(self.app)
            if catalog.CATALOG:
                self.app.select_program(catalog.CATALOG[0].key)
        finally:
            self.suspended = False
        self.save_now()
