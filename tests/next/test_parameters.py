from pathlib import Path

import pytest

from pyrpoc_next.structs.parameters import (
    ChannelSelectionParameter,
    CheckboxParameter,
    ChoiceParameter,
    NumberParameter,
    ParameterError,
    PathParameter,
    TextParameter,
    coerce_parameter_values,
    validate_parameter_groups,
)


def test_text_required_rejects_empty():
    with pytest.raises(ParameterError):
        TextParameter(label="name").coerce("")


def test_text_optional_allows_empty():
    assert TextParameter(label="name", required=False).coerce("") == ""


def test_number_coerces_and_bounds():
    param = NumberParameter(label="x", minimum=1, maximum=10, number_type=int)
    assert param.coerce("5") == 5
    assert isinstance(param.coerce("5"), int)
    with pytest.raises(ParameterError):
        param.coerce(0)
    with pytest.raises(ParameterError):
        param.coerce(11)
    with pytest.raises(ParameterError):
        param.coerce("not a number")


def test_checkbox_parses_strings():
    param = CheckboxParameter(label="save")
    assert param.coerce("true") is True
    assert param.coerce("off") is False
    assert param.coerce(True) is True
    with pytest.raises(ParameterError):
        param.coerce("maybe")


def test_choice_restricts_to_set():
    param = ChoiceParameter(label="mode", choices=["a", "b"])
    assert param.coerce("a") == "a"
    with pytest.raises(ParameterError):
        param.coerce("c")


def test_path_expands_and_rejects_directory():
    result = PathParameter(label="out").coerce("~/data.tif")
    assert isinstance(result, Path)
    assert "~" not in str(result)
    with pytest.raises(ParameterError):
        PathParameter(label="out").coerce("some/dir/")


def test_channel_selection_sorts_dedupes_and_bounds():
    param = ChannelSelectionParameter(label="channels", channel_count=4)
    assert param.coerce([2, 0, 2]) == [0, 2]
    with pytest.raises(ParameterError):
        param.coerce([0, 9])


def test_coerce_values_applies_defaults_and_rejects_unknown():
    groups = {
        "scan": [NumberParameter(label="pixels", default=512, number_type=int)],
        "save": [CheckboxParameter(label="enabled")],
    }
    coerced = coerce_parameter_values(groups, {"enabled": "yes"})
    assert coerced == {"pixels": 512, "enabled": True}
    with pytest.raises(ParameterError):
        coerce_parameter_values(groups, {"bogus": 1})


def test_coerce_values_aggregates_field_errors():
    groups = {"scan": [NumberParameter(label="pixels", minimum=1, number_type=int)]}
    with pytest.raises(ParameterError) as info:
        coerce_parameter_values(groups, {"pixels": -1})
    assert "pixels" in info.value.errors


def test_validate_groups_rejects_duplicate_labels():
    groups = {
        "a": [NumberParameter(label="dupe")],
        "b": [NumberParameter(label="dupe")],
    }
    with pytest.raises(ParameterError):
        validate_parameter_groups(groups)
