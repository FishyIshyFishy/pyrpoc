from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from pyrpoc.domain.app_state import ParameterValue
from pyrpoc.domain.session_state import (
    GuiLayoutSessionState,
    InstrumentSessionState,
    ModalitySessionState,
    OptoControlSessionState,
    DisplaySessionState,
    SessionState,
    SCHEMA_VERSION,
)


class SessionCodec:
    @staticmethod
    def _encode_value(value: Any) -> Any:
        if isinstance(value, Path):
            return {"__type__": "path", "value": str(value)}
        return value

    @staticmethod
    def _decode_value(value: Any) -> Any:
        if isinstance(value, dict) and value.get("__type__") == "path":
            return Path(str(value.get("value", "")))
        return value

    @classmethod
    def _encode_param_values(cls, values: list[ParameterValue]) -> list[dict[str, Any]]:
        return [{"label": entry.label, "value": cls._encode_value(entry.value)} for entry in values]

    @classmethod
    def _decode_param_values(cls, raw: list[dict[str, Any]]) -> list[ParameterValue]:
        out: list[ParameterValue] = []
        for row in raw:
            label = row.get("label")
            if not isinstance(label, str):
                raise ValueError("invalid parameter value label")
            out.append(ParameterValue(label=label, value=cls._decode_value(row.get("value"))))
        return out

    @classmethod
    def to_json_dict(cls, state: SessionState) -> dict[str, Any]:
        raw = asdict(state)
        raw["instruments"] = [
            {
                "type_key": row.type_key,
                "connected": row.connected,
                "config_values": cls._encode_param_values(row.config_values),
                "user_label": row.user_label,
            }
            for row in state.instruments
        ]
        raw["optocontrols"] = [
            {
                "type_key": row.type_key,
                "connected": row.connected,
                "enabled": row.enabled,
                "config_values": cls._encode_param_values(row.config_values),
                "user_label": row.user_label,
            }
            for row in state.optocontrols
        ]
        raw["displays"] = [
            {
                "type_key": row.type_key,
                "attached": row.attached,
                "config_values": cls._encode_param_values(row.config_values),
                "user_label": row.user_label,
            }
            for row in state.displays
        ]
        if state.modality is None:
            raw["modality"] = None
        else:
            raw["modality"] = {
                "selected_key": state.modality.selected_key,
                "configured_params": cls._encode_param_values(state.modality.configured_params),
            }
        return raw

    @classmethod
    def from_json_dict(cls, raw: dict[str, Any]) -> SessionState:
        if not isinstance(raw, dict):
            raise ValueError("session data must be an object")

        schema_version = int(raw.get("schema_version", -1))
        if schema_version not in (1, SCHEMA_VERSION):
            raise ValueError("unsupported session schema version")

        instruments = [
            InstrumentSessionState(
                type_key=str(item["type_key"]),
                connected=bool(item.get("connected", False)),
                config_values=cls._decode_param_values(item.get("config_values", [])),
                user_label=item.get("user_label"),
            )
            for item in raw.get("instruments", [])
        ]
        optocontrols = [
            OptoControlSessionState(
                type_key=str(item["type_key"]),
                connected=bool(item.get("connected", False)),
                enabled=bool(item.get("enabled", False)),
                config_values=cls._decode_param_values(item.get("config_values", [])),
                user_label=item.get("user_label"),
            )
            for item in raw.get("optocontrols", [])
        ]
        displays = [
            DisplaySessionState(
                type_key=str(item["type_key"]),
                attached=bool(item.get("attached", True)),
                config_values=cls._decode_param_values(item.get("config_values", [])),
                user_label=item.get("user_label"),
            )
            for item in raw.get("displays", [])
        ]
        modality_raw = raw.get("modality")
        modality: ModalitySessionState | None = None
        if isinstance(modality_raw, dict):
            modality = ModalitySessionState(
                selected_key=modality_raw.get("selected_key"),
                configured_params=cls._decode_param_values(modality_raw.get("configured_params", [])),
            )

        gui_raw = raw.get("gui_layout", {})
        gui_layout = GuiLayoutSessionState(
            ads_state_base64=gui_raw.get("ads_state_base64"),
            dock_visibility=dict(gui_raw.get("dock_visibility", {})),
            expanded_opto_index=gui_raw.get("expanded_opto_index"),
        )
        return SessionState(
            schema_version=SCHEMA_VERSION,
            theme_mode=str(raw.get("theme_mode", "system")),
            instruments=instruments,
            optocontrols=optocontrols,
            displays=displays,
            modality=modality,
            gui_layout=gui_layout,
        )
