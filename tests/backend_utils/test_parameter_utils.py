from __future__ import annotations

from pathlib import Path

import pytest

from pyrpoc.backend_utils.contracts import Action
from pyrpoc.backend_utils.parameter_utils import (
    ChannelSelectionParameter,
    CheckboxParameter,
    ChoiceParameter,
    NumberParameter,
    ParameterValidationError,
    PathParameter,
    TextParameter,
    coerce_action_values,
    coerce_parameter_values,
    validate_action_list,
    validate_parameter_groups,
)


# --------------------------------------------------------------------------- #
# BaseParameter construction rules
# --------------------------------------------------------------------------- #

def test_display_label_defaults_to_label():
    param = TextParameter(label="Device")
    assert param.display_label == "Device"


def test_explicit_display_label_kept():
    param = TextParameter(label="dev", display_label="Device Name")
    assert param.display_label == "Device Name"


def test_empty_label_rejected():
    with pytest.raises(TypeError):
        TextParameter(label="   ")


# --------------------------------------------------------------------------- #
# Coercion per parameter type
# --------------------------------------------------------------------------- #

def test_text_coerce_stringifies():
    assert TextParameter(label="t").coerce(42) == "42"


def test_number_int_coerce_and_bounds():
    param = NumberParameter(label="n", number_type=int, minimum=0, maximum=10)
    assert param.coerce("5") == 5
    assert isinstance(param.coerce(5.9), int)
    with pytest.raises(ValueError):
        param.coerce(-1)
    with pytest.raises(ValueError):
        param.coerce(11)


def test_number_float_coerce():
    param = NumberParameter(label="n", number_type=float, minimum=1e-6)
    assert param.coerce(2) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        param.coerce(0.0)


def test_number_minimum_cannot_exceed_maximum():
    with pytest.raises(ValueError):
        NumberParameter(label="n", minimum=10, maximum=1)


def test_number_type_must_be_int_or_float():
    with pytest.raises(TypeError):
        NumberParameter(label="n", number_type=str)  # type: ignore[arg-type]


def test_checkbox_coerce_variants():
    param = CheckboxParameter(label="c")
    assert param.coerce(True) is True
    assert param.coerce(1) is True
    assert param.coerce("yes") is True
    assert param.coerce("off") is False
    with pytest.raises(ValueError):
        param.coerce("maybe")


def test_checkbox_default_must_be_bool():
    with pytest.raises(TypeError):
        CheckboxParameter(label="c", default="nope")


def test_choice_coerce_validates_membership():
    param = ChoiceParameter(label="mode", choices=["a", "b"])
    assert param.coerce("a") == "a"
    with pytest.raises(ValueError):
        param.coerce("c")


def test_choice_requires_non_empty_choices():
    with pytest.raises(ValueError):
        ChoiceParameter(label="mode", choices=[])


def test_path_coerce_rules(tmp_path):
    param = PathParameter(label="p")
    coerced = param.coerce(str(tmp_path / "acq"))
    assert isinstance(coerced, Path)
    with pytest.raises(ValueError):
        param.coerce("   ")
    with pytest.raises(ValueError):
        param.coerce("/some/dir/")  # trailing separator → no filename
    with pytest.raises(TypeError):
        param.coerce(123)


def test_channel_selection_coerce_sorts_and_dedupes():
    param = ChannelSelectionParameter(label="ai", num_channels=9)
    assert param.coerce([3, 1, 1, 2]) == [1, 2, 3]
    with pytest.raises(TypeError):
        param.coerce("0,1,2")


def test_channel_selection_default_is_all_channels():
    param = ChannelSelectionParameter(label="ai", num_channels=4)
    assert param.default == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# Group / action validation
# --------------------------------------------------------------------------- #

def test_validate_parameter_groups_ok():
    groups = {"scan": [NumberParameter(label="X", number_type=int)], "daq": [TextParameter(label="dev")]}
    validate_parameter_groups(groups)


def test_validate_parameter_groups_duplicate_label():
    groups = {"a": [TextParameter(label="dup")], "b": [TextParameter(label="dup")]}
    with pytest.raises(ValueError):
        validate_parameter_groups(groups)


def test_validate_parameter_groups_rejects_non_dict():
    with pytest.raises(TypeError):
        validate_parameter_groups([])  # type: ignore[arg-type]


def test_validate_action_list_duplicate_method():
    actions = [
        Action(label="A", method_name="run"),
        Action(label="B", method_name="run"),
    ]
    with pytest.raises(ValueError):
        validate_action_list(actions)


# --------------------------------------------------------------------------- #
# coerce_parameter_values
# --------------------------------------------------------------------------- #

def make_groups():
    return {
        "scan": [
            NumberParameter(label="X", default=512, minimum=1, number_type=int),
            CheckboxParameter(label="save", default=False),
        ]
    }


def test_coerce_applies_defaults_when_missing():
    result = coerce_parameter_values(make_groups(), {})
    assert result == {"X": 512, "save": False}


def test_coerce_overrides_provided_values():
    result = coerce_parameter_values(make_groups(), {"X": "8", "save": "yes"})
    assert result == {"X": 8, "save": True}


def test_coerce_unknown_parameter_raises():
    with pytest.raises(ParameterValidationError):
        coerce_parameter_values(make_groups(), {"bogus": 1})


def test_coerce_aggregates_validation_errors():
    with pytest.raises(ParameterValidationError) as excinfo:
        coerce_parameter_values(make_groups(), {"X": -5})
    assert "X" in excinfo.value.errors


def test_coerce_required_missing_reports_error():
    groups = {"g": [TextParameter(label="needed", required=True, default=None)]}
    with pytest.raises(ParameterValidationError) as excinfo:
        coerce_parameter_values(groups, {})
    assert excinfo.value.errors["needed"] == "value is required"


def test_coerce_action_values_roundtrip():
    action = Action(label="Move", method_name="move", parameters=[NumberParameter(label="steps", number_type=int)])
    assert coerce_action_values(action, {"steps": "3"}) == {"steps": 3}


def test_parameter_validation_error_str_includes_errors():
    err = ParameterValidationError("bad", {"x": "nope"})
    assert "x" in str(err)
