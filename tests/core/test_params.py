"""The parameter model: coercion, form description, unpacking, round-trip."""

from __future__ import annotations

from dataclasses import fields

from dataclasses import dataclass
from pathlib import Path

import pytest

from pyrpoc.core import params as P
from pyrpoc.core.errors import ParameterError
from pyrpoc.core.modulation import MaskBinding


@dataclass
class SampleParams:
    scan: P.ScanGroup = P.group(P.ScanGroup, "Scan")
    daq: P.DaqGroup = P.group(P.DaqGroup, "DAQ")
    modulation: P.ModulationGroup = P.group(P.ModulationGroup, "Modulation")
    num_frames: int = P.int_field("Frames", 1, minimum=1)


# --- coercion --------------------------------------------------------------


def test_int_field_bounds():
    spec = P.IntField("X Pixels", minimum=8, maximum=64)
    assert spec.coerce("16") == 16
    with pytest.raises(ParameterError, match=">= 8"):
        spec.coerce(4)
    with pytest.raises(ParameterError, match="<= 64"):
        spec.coerce(65)


def test_int_field_rejects_bool():
    with pytest.raises(ParameterError):
        P.IntField("Frames").coerce(True)


def test_float_field_bounds():
    spec = P.FloatField("Amplitude", minimum=1e-6)
    assert spec.coerce(2) == pytest.approx(2.0)
    with pytest.raises(ParameterError):
        spec.coerce(0.0)


def test_bool_field_accepts_the_strings_the_old_model_accepted():
    spec = P.BoolField("Save")
    assert spec.coerce("yes") is True
    assert spec.coerce("off") is False
    assert spec.coerce(1) is True
    with pytest.raises(ParameterError):
        spec.coerce("maybe")


def test_path_field_rejects_a_bare_directory():
    spec = P.PathField("Mask File")
    assert spec.coerce("/data/run1") == "/data/run1"
    with pytest.raises(ParameterError, match="filename"):
        spec.coerce("/data/")


def test_path_field_allows_empty_so_a_path_can_be_unset():
    """An unset path is a normal state; whoever needs one checks at use time."""
    assert P.PathField("Mask File").coerce("") == ""


def test_channels_field_sorts_and_deduplicates():
    spec = P.ChannelsField("Active AI Channels", num_channels=9)
    assert spec.coerce([3, 1, 1, 0]) == (0, 1, 3)
    with pytest.raises(ParameterError, match="outside"):
        spec.coerce([12])


def test_masks_field_builds_bindings():
    spec = P.MasksField("Masks")
    out = spec.coerce([{"path": "m.png", "port": 1, "line": 3}])
    assert out == (MaskBinding(Path("m.png"), 1, 3),)
    assert spec.encode(out) == [{"path": "m.png", "port": 1, "line": 3}]


# --- groups ----------------------------------------------------------------


def test_a_group_is_not_a_mapping():
    """``Group`` carried ``keys``/``__getitem__`` so that ``f(**p.scan)`` worked.

    Nothing splats a group any more -- callers take the group itself, so a
    renamed field is a type error rather than a TypeError on the first frame.
    The protocol was removed with the last splat; this is what stops it coming
    back by accident.
    """
    scan = P.ScanGroup(x_pixels=64, dwell_time_us=3.0)
    assert (scan.x_pixels, scan.dwell_time_us) == (64, 3.0)
    with pytest.raises(TypeError):
        dict(**scan)  # type: ignore[arg-type]


def test_scan_group_total_x():
    scan = P.ScanGroup(x_pixels=8, extra_left=3, extra_right=2)
    assert scan.total_x == 13


def test_flim_daq_group_overrides_only_the_default():
    assert P.DaqGroup().sample_rate_hz == 100_000.0
    assert P.FlimDaqGroup().sample_rate_hz == 1_000_000.0
    assert [f.name for f in fields(P.FlimDaqGroup())] == ["sample_rate_hz"]


def test_histogram_group_laser_period():
    assert P.HistogramGroup(laser_frequency_mhz=80.0).laser_period_ps == 12500


# --- form description ------------------------------------------------------


def test_sections_are_groups_in_order_then_root_scalars():
    labels = [section.label for section in P.sections(SampleParams)]
    assert labels == ["Scan", "DAQ", "Modulation", "Acquisition"]


def test_section_entries_are_dotted_paths():
    scan = next(s for s in P.sections(SampleParams) if s.label == "Scan")
    names = [path for path, _ in scan.entries]
    assert names[:2] == ["scan.x_pixels", "scan.y_pixels"]
    assert all(name.startswith("scan.") for name in names)

    acq = next(s for s in P.sections(SampleParams) if s.label == "Acquisition")
    assert [path for path, _ in acq.entries] == ["num_frames"]


def test_get_and_set_by_path():
    params = SampleParams()
    assert P.get_path(params, "scan.x_pixels") == 512
    P.set_path(params, "scan.x_pixels", 64)
    assert params.scan.x_pixels == 64
    assert P.spec_at(SampleParams, "scan.x_pixels").label == "X Pixels"


# --- serialisation ---------------------------------------------------------


def test_round_trip_preserves_values():
    params = SampleParams()
    params.scan.x_pixels = 64
    params.daq.sample_rate_hz = 250_000.0
    params.modulation.masks = (MaskBinding(Path("m.png"), 0, 3),)
    params.num_frames = 5

    raw = P.to_dict(params)
    restored = P.from_dict(SampleParams, raw)

    assert restored.scan.x_pixels == 64
    assert restored.daq.sample_rate_hz == 250_000.0
    assert restored.modulation.masks == (MaskBinding(Path("m.png"), 0, 3),)
    assert restored.num_frames == 5
    assert P.to_dict(restored) == raw


def test_to_dict_is_json_safe():
    import json

    params = SampleParams()
    params.modulation.masks = (MaskBinding(Path("m.png"), 0, 3),)
    json.dumps(P.to_dict(params))  # must not raise


def test_from_dict_uses_defaults_for_missing_keys():
    restored = P.from_dict(SampleParams, {"num_frames": 3})
    assert restored.num_frames == 3
    assert restored.scan.x_pixels == 512


def test_from_dict_coerces_and_rejects_out_of_range():
    with pytest.raises(ParameterError, match=">= 8"):
        P.from_dict(SampleParams, {"scan": {"x_pixels": 2}})


def test_coerce_is_strict_about_unknown_keys():
    with pytest.raises(ParameterError, match="unknown"):
        P.coerce(SampleParams, {"nope": 1})
    P.from_dict(SampleParams, {"nope": 1})  # non-strict ignores it


def test_validate_walks_current_values():
    params = SampleParams()
    P.validate(params)
    params.scan.x_pixels = 2
    with pytest.raises(ParameterError):
        P.validate(params)
