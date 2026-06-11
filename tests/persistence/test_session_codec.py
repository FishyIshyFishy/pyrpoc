from __future__ import annotations

from pathlib import Path

import pytest

from pyrpoc.domain.app_state import ParameterValue
from pyrpoc.domain.session_state import (
    DisplaySessionState,
    InstrumentSessionState,
    ModalitySessionState,
    OptoControlSessionState,
    SessionState,
    schema_version,
)
from pyrpoc.persistence.session_codec import SessionCodec


# --------------------------------------------------------------------------- #
# encode_value / decode_value
# --------------------------------------------------------------------------- #

def test_path_round_trips():
    encoded = SessionCodec.encode_value(Path("dir/file"))
    assert encoded["__type__"] == "path"
    assert SessionCodec.decode_value(encoded) == Path("dir/file")


def test_tuple_encodes_to_list():
    assert SessionCodec.encode_value((1, 2, 3)) == [1, 2, 3]


def test_dict_keys_stringified():
    assert SessionCodec.encode_value({1: "x"}) == {"1": "x"}


def test_primitives_pass_through():
    assert SessionCodec.encode_value(5) == 5
    assert SessionCodec.encode_value(None) is None
    assert SessionCodec.encode_value(True) is True


def test_unserializable_value_falls_back_to_str():
    assert SessionCodec.encode_value({1, 2}) == str({1, 2})


def test_nested_path_inside_dict_round_trips():
    encoded = SessionCodec.encode_value({"out": Path("a/b")})
    decoded = SessionCodec.decode_value(encoded)
    assert decoded == {"out": Path("a/b")}


# --------------------------------------------------------------------------- #
# parameter value codecs
# --------------------------------------------------------------------------- #

def test_param_values_round_trip_with_path():
    values = [ParameterValue(label="save_path", value=Path("acq")), ParameterValue(label="n", value=3)]
    encoded = SessionCodec.encode_param_values(values)
    decoded = SessionCodec.decode_param_values(encoded)
    assert decoded[0].label == "save_path"
    assert decoded[0].value == Path("acq")
    assert decoded[1].value == 3


def test_decode_param_values_rejects_bad_label():
    with pytest.raises(ValueError):
        SessionCodec.decode_param_values([{"value": 1}])


# --------------------------------------------------------------------------- #
# pick helpers
# --------------------------------------------------------------------------- #

def test_pick_type_key_prefers_type_key_then_fallbacks():
    assert SessionCodec.pick_type_key({"type_key": "a", "key": "b"}, "key") == "a"
    assert SessionCodec.pick_type_key({"key": "b"}, "key") == "b"
    assert SessionCodec.pick_type_key({}, "key") == ""


def test_pick_instance_id_uses_existing_or_generates():
    assert SessionCodec.pick_instance_id({"instance_id": "fixed"}, "t") == "fixed"
    generated = SessionCodec.pick_instance_id({}, "confocal")
    assert generated.startswith("confocal-")


# --------------------------------------------------------------------------- #
# full document round trip
# --------------------------------------------------------------------------- #

def make_state() -> SessionState:
    return SessionState(
        theme_mode="dark",
        instruments=[
            InstrumentSessionState(
                type_key="time_tagger",
                instance_id="time_tagger-1",
                connected=True,
                persisted_state={"serial": "123"},
                config_values=[ParameterValue(label="ch", value=1)],
                user_label="Tagger A",
            )
        ],
        optocontrols=[
            OptoControlSessionState(
                type_key="mask",
                instance_id="mask-1",
                enabled=True,
                persisted_state={"mask_path": str(Path("m.tiff"))},
            )
        ],
        displays=[
            DisplaySessionState(type_key="streamed_image", instance_id="disp-1", dock_visible=False)
        ],
        modality=ModalitySessionState(
            selected_key="confocal",
            params_by_modality={
                "confocal": [ParameterValue(label="X Pixels", value=256)],
                "flim": [ParameterValue(label="Laser Frequency MHz", value=80.0)],
            },
        ),
        ads_layout="ZmFrZS1sYXlvdXQ=",
    )


def test_full_round_trip_preserves_fields():
    restored = SessionCodec.from_json_dict(SessionCodec.to_json_dict(make_state()))
    assert restored.schema_version == schema_version
    assert restored.theme_mode == "dark"
    assert restored.instruments[0].type_key == "time_tagger"
    assert restored.instruments[0].connected is True
    assert restored.instruments[0].config_values[0].value == 1
    assert restored.optocontrols[0].enabled is True
    assert restored.displays[0].dock_visible is False
    assert restored.modality is not None
    assert restored.modality.selected_key == "confocal"
    # Every modality's params are remembered, not just the active one.
    assert restored.modality.params_by_modality["confocal"][0].value == 256
    assert restored.modality.params_by_modality["flim"][0].value == 80.0
    assert restored.ads_layout == "ZmFrZS1sYXlvdXQ="


def test_from_json_rejects_unsupported_version():
    with pytest.raises(ValueError):
        SessionCodec.from_json_dict({"schema_version": 99})


def test_from_json_accepts_legacy_version_and_config_map():
    raw = {
        "schema_version": 2,
        "instruments": [
            {"type_key": "time_tagger", "config": {"Channel": 4}}
        ],
    }
    restored = SessionCodec.from_json_dict(raw)
    assert restored.instruments[0].config_values[0].label == "Channel"
    assert restored.instruments[0].config_values[0].value == 4


def test_from_json_requires_object():
    with pytest.raises(ValueError):
        SessionCodec.from_json_dict([])  # type: ignore[arg-type]


def test_legacy_single_modality_params_map_under_selected_key():
    # Pre-v6 sessions stored one `configured_params` list for the active modality.
    raw = {
        "schema_version": 5,
        "modality": {
            "selected_key": "confocal",
            "configured_params": [{"label": "X Pixels", "value": 128}],
        },
    }
    restored = SessionCodec.from_json_dict(raw)
    assert restored.modality is not None
    assert restored.modality.params_by_modality["confocal"][0].value == 128


def test_ads_layout_round_trips():
    restored = SessionCodec.from_json_dict(SessionCodec.to_json_dict(SessionState(ads_layout="QUJD")))
    assert restored.ads_layout == "QUJD"


def test_ads_layout_non_string_decodes_to_none():
    restored = SessionCodec.from_json_dict({"schema_version": 6, "ads_layout": 123})
    assert restored.ads_layout is None


def test_empty_modality_map_round_trips():
    restored = SessionCodec.from_json_dict(SessionCodec.to_json_dict(SessionState()))
    assert restored.modality is None or restored.modality.params_by_modality == {}
