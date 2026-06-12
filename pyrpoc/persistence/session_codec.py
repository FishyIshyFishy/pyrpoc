from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from pyrpoc.domain.app_state import ParameterValue
from pyrpoc.domain.session_state import (
    InstrumentSessionState,
    ModalitySessionState,
    OptoControlSessionState,
    DisplaySessionState,
    SessionState,
    schema_version,
)
from pyrpoc.backend_utils.state_helpers import make_instance_id


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
    def decode_params_by_modality(cls, modality_raw: dict[str, Any]) -> dict[str, list[ParameterValue]]:
        """Decode the per-modality parameter map (with a single-list legacy fallback)."""
        raw_map = modality_raw.get("params_by_modality")
        if isinstance(raw_map, dict):
            return {
                str(key): cls.decode_param_values(values)
                for key, values in raw_map.items()
                if isinstance(values, list)
            }
        # legacy (schema <= 5): one configured_params list belonging to selected_key
        legacy = modality_raw.get("configured_params")
        selected_key = modality_raw.get("selected_key")
        if isinstance(legacy, list) and isinstance(selected_key, str) and selected_key:
            return {selected_key: cls.decode_param_values(legacy)}
        return {}

    @classmethod
    def decode_config_values_with_legacy_fallback(cls, item: dict[str, Any]) -> list[ParameterValue]:
        """
        Accept modern and legacy config shapes.

        Supported shapes:
        - current: {"config_values": [{"label": "...", "value": ...}, ...]}
        - legacy map: {"config": {"Label": value, ...}} or {"settings": {...}}
        """
        raw_values = item.get("config_values", [])
        if isinstance(raw_values, list) and raw_values:
            try:
                return cls.decode_param_values(raw_values)
            except Exception:
                pass

        for legacy_field in ("config", "settings"):
            legacy = item.get(legacy_field)
            if isinstance(legacy, dict):
                return [
                    ParameterValue(label=str(label), value=cls.decode_value(value))
                    for label, value in legacy.items()
                ]
        return []

    @classmethod
    def legacy_config_to_state_dict(cls, item: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for entry in cls.decode_config_values_with_legacy_fallback(item):
            out[entry.label] = entry.value
        return out

    @classmethod
    def pick_type_key(cls, item: dict[str, Any], *fallback_keys: str) -> str:
        for key in ("type_key", *fallback_keys):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    @classmethod
    def pick_instance_id(cls, item: dict[str, Any], type_key: str) -> str:
        value = item.get("instance_id", item.get("id", ""))
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
                "docked_visible": row.dock_visible,
                "persisted_state": cls.encode_value(row.persisted_state),
                "config_values": cls.encode_param_values(row.config_values),
                "user_label": row.user_label,
            }
            for row in state.displays
        ]
        if state.modality is None:
            raw["modality"] = None
        else:
            raw["modality"] = {
                "selected_key": state.modality.selected_key,
                "params_by_modality": {
                    key: cls.encode_param_values(values)
                    for key, values in state.modality.params_by_modality.items()
                },
            }
        return raw

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
            type_key = cls.pick_type_key(item, "instrument_key", "key")
            if not type_key:
                continue
            persisted_state_raw = item.get("persisted_state")
            if isinstance(persisted_state_raw, dict):
                persisted_state = cls.decode_value(persisted_state_raw)
            else:
                persisted_state = cls.legacy_config_to_state_dict(item)
            instruments.append(
                InstrumentSessionState(
                    type_key=type_key,
                    instance_id=cls.pick_instance_id(item, type_key),
                    connected=bool(item.get("connected", False)),
                    persisted_state=persisted_state,
                    config_values=cls.decode_config_values_with_legacy_fallback(item),
                    user_label=item.get("user_label"),
                )
            )

        optocontrols: list[OptoControlSessionState] = []
        for item in raw.get("optocontrols", []):
            if not isinstance(item, dict):
                continue
            type_key = cls.pick_type_key(item, "opto_control_key", "key")
            if not type_key:
                continue
            persisted_state_raw = item.get("persisted_state")
            if isinstance(persisted_state_raw, dict):
                persisted_state = cls.decode_value(persisted_state_raw)
            else:
                persisted_state = cls.legacy_config_to_state_dict(item)
            optocontrols.append(
                OptoControlSessionState(
                    type_key=type_key,
                    instance_id=cls.pick_instance_id(item, type_key),
                    connected=bool(item.get("connected", False)),
                    enabled=bool(item.get("enabled", False)),
                    persisted_state=persisted_state,
                    config_values=cls.decode_config_values_with_legacy_fallback(item),
                    user_label=item.get("user_label"),
                )
            )

        displays: list[DisplaySessionState] = []
        for item in raw.get("displays", []):
            if not isinstance(item, dict):
                continue
            type_key = cls.pick_type_key(item, "display_key", "key")
            if not type_key:
                continue
            persisted_state_raw = item.get("persisted_state")
            if isinstance(persisted_state_raw, dict):
                persisted_state = cls.decode_value(persisted_state_raw)
            else:
                persisted_state = cls.legacy_config_to_state_dict(item)
            displays.append(
                DisplaySessionState(
                    type_key=type_key,
                    instance_id=cls.pick_instance_id(item, type_key),
                    attached=bool(item.get("attached", True)),
                    dock_visible=bool(item.get("dock_visible", item.get("docked_visible", True))),
                    persisted_state=persisted_state,
                    config_values=cls.decode_config_values_with_legacy_fallback(item),
                    user_label=item.get("user_label"),
                )
            )

        modality_raw = raw.get("modality")
        modality: ModalitySessionState | None = None
        if isinstance(modality_raw, dict):
            modality = ModalitySessionState(
                selected_key=modality_raw.get("selected_key"),
                params_by_modality=cls.decode_params_by_modality(modality_raw),
            )

        ads_layout = raw.get("ads_layout")
        return SessionState(
            schema_version=version,
            # older sessions stored a "theme_mode" of dark/light/system; unknown
            # names fall back to the default theme when applied
            theme=str(raw.get("theme", raw.get("theme_mode", "dark-pink"))),
            instruments=instruments,
            optocontrols=optocontrols,
            displays=displays,
            modality=modality,
            ads_layout=ads_layout if isinstance(ads_layout, str) else None,
        )
