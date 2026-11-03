"""The saved session, split into working state and screen state, and the file
it lives in."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pyrpoc.core.program_inputs import ParameterError

log = logging.getLogger(__name__)

# Bump when the shape changes, and add the step from the old shape to UPGRADES.
SCHEMA_VERSION = 9

# What malformed session JSON raises on its way into the application.
BAD_STATE = (KeyError, TypeError, ValueError, ParameterError)


@dataclass
class DeviceState:
    key: str
    instance_id: str = ""
    panel_open: bool = False
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewState:
    """One added panel."""

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
class LibraryState:
    auto_purge: bool = False


@dataclass
class WorkingState:
    """What the app is: its devices, every program input's value, where runs
    save, and the library's settings."""

    devices: list[DeviceState] = field(default_factory=list)
    program_inputs: dict[str, Any] = field(default_factory=dict)
    save: SaveState = field(default_factory=SaveState)
    library: LibraryState = field(default_factory=LibraryState)


@dataclass
class ScreenState:
    """How the app is shown: the added panels, each fixed panel's own state by
    dock key, and the dock layout."""

    views: list[ViewState] = field(default_factory=list)
    panels: dict[str, dict[str, Any]] = field(default_factory=dict)
    ads_layout: str | None = None


@dataclass
class SessionState:
    schema_version: int = SCHEMA_VERSION
    working: WorkingState = field(default_factory=WorkingState)
    screen: ScreenState = field(default_factory=ScreenState)


def default_session_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "pyrpoc" / "session.json"


class SessionFile:
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
        if not isinstance(raw, dict):
            return SessionState()
        try:
            upgraded = upgrade(raw)
            return decode(upgraded) if upgraded is not None else SessionState()
        except BAD_STATE:
            log.warning("could not decode %s; starting fresh", self.path, exc_info=True)
            return SessionState()

    def save(self, state: SessionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(state), indent=2, default=str)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)


def upgrade_v8(raw: dict[str, Any]) -> dict[str, Any]:
    """A version 8 session, one flat object with parameters by block, as version 9."""
    blocks = as_object(raw.get("param_blocks"))
    return {
        "schema_version": 9,
        "working": {
            "devices": raw.get("devices", []),
            "program_inputs": {
                f"{block}.{name}": value
                for block, values in blocks.items()
                if isinstance(values, dict)
                for name, value in values.items()
            },
            "save": raw.get("save"),
            "library": raw.get("library"),
        },
        "screen": {
            "views": raw.get("views", []),
            "panels": {"acquisition": {"program": raw.get("selected_program")}},
            "ads_layout": raw.get("ads_layout"),
        },
    }


# Each step turns a session of its key's version into the next version.
UPGRADES: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {8: upgrade_v8}


def upgrade(raw: dict[str, Any]) -> dict[str, Any] | None:
    """``raw`` brought to the current version, or None for a version with no
    path to it."""
    version = raw.get("schema_version")
    while version != SCHEMA_VERSION:
        if not isinstance(version, int) or version not in UPGRADES:
            return None
        raw = UPGRADES[version](raw)
        version = raw.get("schema_version")
    return raw


def as_object(raw: Any) -> dict[str, Any]:
    """``raw`` if it is a JSON object, otherwise an empty one."""
    return raw if isinstance(raw, dict) else {}


def decode_rows(rows: Any) -> list[dict[str, Any]]:
    """The entries of a saved list that are objects with a key."""
    return [row for row in rows if isinstance(row, dict) and row.get("key")]


def decode(raw: dict[str, Any]) -> SessionState:
    return SessionState(
        working=decode_working(as_object(raw.get("working"))),
        screen=decode_screen(as_object(raw.get("screen"))),
    )


def decode_working(raw: dict[str, Any]) -> WorkingState:
    devices = [
        DeviceState(
            key=str(row["key"]),
            instance_id=str(row.get("instance_id", "")),
            panel_open=bool(row.get("panel_open", False)),
            state=dict(row.get("state") or {}),
        )
        for row in decode_rows(raw.get("devices", []))
    ]
    return WorkingState(
        devices=devices,
        program_inputs={
            str(key): value for key, value in as_object(raw.get("program_inputs")).items()
        },
        save=decode_save(raw.get("save")),
        library=decode_library(raw.get("library")),
    )


def decode_screen(raw: dict[str, Any]) -> ScreenState:
    views = [
        ViewState(
            key=str(row["key"]),
            instance_id=str(row.get("instance_id", "")),
            user_label=row.get("user_label"),
            state=dict(row.get("state") or {}),
        )
        for row in decode_rows(raw.get("views", []))
    ]
    panels = {
        str(key): dict(value)
        for key, value in as_object(raw.get("panels")).items()
        if isinstance(value, dict)
    }
    layout = raw.get("ads_layout")
    return ScreenState(
        views=views, panels=panels, ads_layout=layout if isinstance(layout, str) else None
    )


def decode_library(raw: Any) -> LibraryState:
    if not isinstance(raw, dict):
        return LibraryState()
    return LibraryState(auto_purge=bool(raw.get("auto_purge", False)))


def decode_save(raw: Any) -> SaveState:
    if not isinstance(raw, dict):
        return SaveState()
    default = SaveState()
    return SaveState(
        name=str(raw.get("name", default.name)),
        directory=str(raw.get("directory", default.directory)),
        enabled=bool(raw.get("enabled", default.enabled)),
    )
