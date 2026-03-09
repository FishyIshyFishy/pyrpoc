from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QStandardPaths

from pyrpoc.domain.session_state import SessionState
from .session_codec import SessionCodec


class SessionRepository:
    def __init__(self):
        self.path = self._session_path()
        self.last_load_error: str | None = None

    def _session_path(self) -> Path:
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        root = Path(base) if base else Path(".")
        root.mkdir(parents=True, exist_ok=True)
        return root / "session.json"

    def load_or_default(self) -> SessionState:
        try:
            if not self.path.exists():
                self.last_load_error = None
                return SessionState()
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            state = SessionCodec.from_json_dict(raw)
            self.last_load_error = None
            return state
        except Exception as exc:
            self.last_load_error = (
                f"Failed to load session from {self.path} "
                f"({type(exc).__name__}: {exc})"
            )
            return SessionState()

    def save(self, state: SessionState) -> None:
        payload = SessionCodec.to_json_dict(state)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(self.path)
