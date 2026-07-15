from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from pyrpoc.utils.state_helpers import make_instance_id
from pyrpoc.structs.parameters import ParameterValue
from pyrpoc.structs.session import (
    DisplaySessionState,
    InstrumentSessionState,
    OptoControlSessionState,
    PresetSessionState,
    SessionState,
    schema_version,
)


class SessionCodec:
    @staticmethod
    def encode_value(value: Any) -> Any:
        if isinstance(value, Path):
            return {"__type__": "path", "value": str(value)}
        if isinstance(value, dict):
            return {str(k): SessionCodec.encode_value(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [SessionCodec.encode_value(v) for v in value]
        if isinstance(value, list):
            return [SessionCodec.encode_value(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    @staticmethod
    def decode_value(value: Any) -> Any:
        if isinstance(value, dict) and value.get("__type__") == "path":
            return Path(str(value.get("value", "")))
        if isinstance(value, dict):
            return {SessionCodec.decode_value(k): SessionCodec.decode_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SessionCodec.decode_value(item) for item in value]
        return value

    @classmethod
    def encode_param_values(cls, values: list[ParameterValue]) -> list[dict[str, Any]]:
        return [{"label": entry.label, "value": cls.encode_value(entry.value)} for entry in values]

    @classmethod
    def decode_param_values(cls, raw: list[dict[str, Any]]) -> list[ParameterValue]:
        out: list[ParameterValue] = []
        for row in raw:
            label = row.get("label")
            if not isinstance(label, str):
                raise ValueError("invalid parameter value label")
            out.append(ParameterValue(label=label, value=cls.decode_value(row.get("value"))))
        return out

    @classmethod
    def decode_config_values(cls, item: dict[str, Any]) -> list[ParameterValue]:
        raw_values = item.get("config_values", [])
        if isinstance(raw_values, list) and raw_values:
            return cls.decode_param_values(raw_values)
        return []

    @classmethod
    def pick_instance_id(cls, item: dict[str, Any], type_key: str) -> str:
        value = item.get("instance_id", "")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return make_instance_id(type_key or "item")

    @classmethod
    def to_json_dict(cls, state: SessionState) -> dict[str, Any]:
        raw = asdict(state)
        raw["instruments"] = [
            {
                "type_key": row.type_key,
                "instance_id": row.instance_id,
                "connected": row.connected,
                "persisted_state": cls.encode_value(row.persisted_state),
                "config_values": cls.encode_param_values(row.config_values),
                "user_label": row.user_label,
            }
            for row in state.instruments
        ]
        raw["optocontrols"] = [
            {
                "type_key": row.type_key,
                "instance_id": row.instance_id,
                "connected": row.connected,
                "enabled": row.enabled,
                "persisted_state": cls.encode_value(row.persisted_state),
                "config_values": cls.encode_param_values(row.config_values),
                "user_label": row.user_label,
            }
            for row in state.optocontrols
        ]
        raw["displays"] = [
            {
                "type_key": row.type_key,
                "instance_id": row.instance_id,
                "attached": row.attached,
                "dock_visible": row.dock_visible,
                "persisted_state": cls.encode_value(row.persisted_state),
                "config_values": cls.encode_param_values(row.config_values),
                "user_label": row.user_label,
            }
            for row in state.displays
        ]
        if state.preset is None:
            raw["preset"] = None
        else:
            raw["preset"] = {
                "selected_key": state.preset.selected_key,
                "params_by_preset": {
                    key: cls.encode_param_values(values)
                    for key, values in state.preset.params_by_preset.items()
                },
            }
        return raw

    @classmethod
    def decode_params_by_preset(cls, preset_raw: dict[str, Any]) -> dict[str, list[ParameterValue]]:
        raw_map = preset_raw.get("params_by_preset")
        if isinstance(raw_map, dict):
            return {
                str(key): cls.decode_param_values(values)
                for key, values in raw_map.items()
                if isinstance(values, list)
            }
        return {}

    @classmethod
    def from_json_dict(cls, raw: dict[str, Any]) -> SessionState:
        if not isinstance(raw, dict):
            raise ValueError("session data must be an object")

        version = int(raw.get("schema_version", -1))
        if version < 1 or version > schema_version:
            raise ValueError("unsupported session schema version")

        instruments: list[InstrumentSessionState] = []
        for item in raw.get("instruments", []):
            if not isinstance(item, dict):
                continue
            type_key = item.get("type_key")
            if not isinstance(type_key, str) or not type_key:
                continue
            persisted_state = item.get("persisted_state")
            persisted = cls.decode_value(persisted_state) if isinstance(persisted_state, dict) else {}
            instruments.append(
                InstrumentSessionState(
                    type_key=type_key,
                    instance_id=cls.pick_instance_id(item, type_key),
                    connected=bool(item.get("connected", False)),
                    persisted_state=persisted,
                    config_values=cls.decode_config_values(item),
                    user_label=item.get("user_label"),
                )
            )

        optocontrols: list[OptoControlSessionState] = []
        for item in raw.get("optocontrols", []):
            if not isinstance(item, dict):
                continue
            type_key = item.get("type_key")
            if not isinstance(type_key, str) or not type_key:
                continue
            persisted_state = item.get("persisted_state")
            persisted = cls.decode_value(persisted_state) if isinstance(persisted_state, dict) else {}
            optocontrols.append(
                OptoControlSessionState(
                    type_key=type_key,
                    instance_id=cls.pick_instance_id(item, type_key),
                    connected=bool(item.get("connected", False)),
                    enabled=bool(item.get("enabled", False)),
                    persisted_state=persisted,
                    config_values=cls.decode_config_values(item),
                    user_label=item.get("user_label"),
                )
            )

        displays: list[DisplaySessionState] = []
        for item in raw.get("displays", []):
            if not isinstance(item, dict):
                continue
            type_key = item.get("type_key")
            if not isinstance(type_key, str) or not type_key:
                continue
            persisted_state = item.get("persisted_state")
            persisted = cls.decode_value(persisted_state) if isinstance(persisted_state, dict) else {}
            displays.append(
                DisplaySessionState(
                    type_key=type_key,
                    instance_id=cls.pick_instance_id(item, type_key),
                    attached=bool(item.get("attached", True)),
                    dock_visible=bool(item.get("dock_visible", True)),
                    persisted_state=persisted,
                    config_values=cls.decode_config_values(item),
                    user_label=item.get("user_label"),
                )
            )

        preset_raw = raw.get("preset")
        preset: PresetSessionState | None = None
        if isinstance(preset_raw, dict):
            preset = PresetSessionState(
                selected_key=preset_raw.get("selected_key"),
                params_by_preset=cls.decode_params_by_preset(preset_raw),
            )

        ads_layout = raw.get("ads_layout")
        return SessionState(
            schema_version=version,
            theme_mode=str(raw.get("theme_mode", "system")),
            instruments=instruments,
            optocontrols=optocontrols,
            displays=displays,
            preset=preset,
            ads_layout=ads_layout if isinstance(ads_layout, str) else None,
        )


class SessionRepository:
    """Reads/writes the session JSON file at a given path (Qt-free).

    The path is supplied by the caller (the gui layer resolves the per-user
    AppData location and passes it in)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.last_load_error: str | None = None

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
                f"Failed to load session from {self.path} ({type(exc).__name__}: {exc})"
            )
            return SessionState()

    def save(self, state: SessionState) -> None:
        payload = SessionCodec.to_json_dict(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(self.path)
