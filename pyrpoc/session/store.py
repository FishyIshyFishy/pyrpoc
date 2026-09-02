"""Reading and writing the session file.

No Qt: the path is supplied rather than looked up through QStandardPaths, so
this stays importable and testable headless.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .state import SCHEMA_VERSION, DeviceState, SessionState, ViewState


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

        A version mismatch is not an error to report at the user: v3.1 changed
        how parameters are stored, so a v6 file simply resets. Anything else
        that goes wrong is recorded in ``last_load_error``.
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
            visible=bool(row.get("visible", True)),
            state=dict(row.get("state") or {}),
        )
        for row in raw.get("views", [])
        if isinstance(row, dict) and row.get("key")
    ]
    params = {
        str(key): dict(value)
        for key, value in (raw.get("params_by_program") or {}).items()
        if isinstance(value, dict)
    }
    layout = raw.get("ads_layout")
    return SessionState(
        schema_version=SCHEMA_VERSION,
        theme_mode=str(raw.get("theme_mode", "system")),
        devices=devices,
        views=views,
        selected_program=raw.get("selected_program"),
        params_by_program=params,
        ads_layout=layout if isinstance(layout, str) else None,
    )
