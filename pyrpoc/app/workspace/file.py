"""What the saved workspace holds, and reading and writing the file it lives in.

Configuration and layout only, as small JSON; acquired data lives in the run's
own files. Nothing here knows the live application or Qt; ``restore.py``
converts between the two.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pyrpoc.structs.params import ParameterError

log = logging.getLogger(__name__)

# Bump when the shape changes. A file of another version loads as defaults,
# since there is no converter between shapes.
SCHEMA_VERSION = 8

# What malformed workspace JSON raises on its way into the application.
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
class LibraryState:
    auto_purge: bool = False


@dataclass
class WorkspaceState:
    schema_version: int = SCHEMA_VERSION
    devices: list[DeviceState] = field(default_factory=list)
    views: list[ViewState] = field(default_factory=list)
    selected_program: str | None = None
    # Every parameter block, keyed by class name: one entry per block, shared.
    param_blocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    save: SaveState = field(default_factory=SaveState)
    # Optional in the file, so adding it needed no schema bump.
    library: LibraryState = field(default_factory=LibraryState)
    ads_layout: str | None = None


def default_workspace_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    # The file's old name, kept so existing workspaces still load.
    return root / "pyrpoc" / "session.json"


class WorkspaceFile:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> WorkspaceState:
        """The saved workspace, or defaults if there is not a usable one. A
        corrupt file is logged rather than blocking launch."""
        if not self.path.exists():
            return WorkspaceState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("could not read %s; starting fresh", self.path, exc_info=True)
            return WorkspaceState()
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            return WorkspaceState()
        try:
            return decode(raw)
        except BAD_STATE:
            log.warning("could not decode %s; starting fresh", self.path, exc_info=True)
            return WorkspaceState()

    def save(self, state: WorkspaceState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(state), indent=2, default=str)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)


def decode_rows(rows: Any) -> list[dict[str, Any]]:
    """The entries of a saved list that are objects with a key."""
    return [row for row in rows if isinstance(row, dict) and row.get("key")]


def decode(raw: dict[str, Any]) -> WorkspaceState:
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
    return WorkspaceState(
        devices=devices,
        views=views,
        selected_program=raw.get("selected_program"),
        param_blocks=blocks,
        save=decode_save(raw.get("save")),
        library=decode_library(raw.get("library")),
        ads_layout=layout if isinstance(layout, str) else None,
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
