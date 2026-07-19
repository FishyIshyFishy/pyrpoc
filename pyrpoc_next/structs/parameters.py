"""Declarative parameters: data-only specs of configurable values.

These describe a value (type, bounds, default) and know how to coerce a raw input.
Widget rendering lives in ``gui`` — nothing here touches Qt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attrs import define, field


class ParameterError(Exception):
    """Raised when parameter values fail coercion or validation."""

    def __init__(self, message: str, errors: dict[str, str] | None = None):
        super().__init__(message)
        self.errors = errors or {}


@define
class ParameterValue:
    """A configured value tagged with the label of the parameter it fills."""

    label: str
    value: Any


@define(kw_only=True)
class Parameter:
    """Base declaration for one configurable value."""

    label: str
    default: Any = None
    required: bool = True
    tooltip: str = ""
    display_label: str = ""

    def summary_label(self) -> str:
        return self.display_label or self.label

    def coerce(self, value: Any) -> Any:
        """Convert a raw value to this parameter's type, raising ParameterError if invalid."""
        raise NotImplementedError


@define(kw_only=True)
class TextParameter(Parameter):
    """Free text."""

    def coerce(self, value):
        text = "" if value is None else str(value)
        if self.required and not text:
            raise ParameterError(f"{self.summary_label()} is required")
        return text


@define(kw_only=True)
class PathParameter(TextParameter):
    """A filesystem path; expands ~ and rejects a directory-only path."""

    def coerce(self, value):
        text = super().coerce(value)
        if not text:
            return text
        if text.endswith(("/", "\\")):
            raise ParameterError(f"{self.summary_label()} must be a file, not a directory")
        return Path(text).expanduser()


@define(kw_only=True)
class NumberParameter(Parameter):
    """An integer or float within optional bounds."""

    minimum: float | None = None
    maximum: float | None = None
    step: float = 1
    number_type: type = float

    def coerce(self, value):
        try:
            number = self.number_type(value)
        except (TypeError, ValueError):
            raise ParameterError(f"{self.summary_label()} must be a number")
        if self.minimum is not None and number < self.minimum:
            raise ParameterError(f"{self.summary_label()} must be >= {self.minimum}")
        if self.maximum is not None and number > self.maximum:
            raise ParameterError(f"{self.summary_label()} must be <= {self.maximum}")
        return number


@define(kw_only=True)
class CheckboxParameter(Parameter):
    """A boolean toggle."""

    default: bool = False

    def coerce(self, value):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        raise ParameterError(f"{self.summary_label()} must be true or false")


@define(kw_only=True)
class ChoiceParameter(Parameter):
    """A value chosen from a fixed set of strings."""

    choices: list[str] = field(factory=list)

    def coerce(self, value):
        text = str(value)
        if text not in self.choices:
            raise ParameterError(f"{self.summary_label()} must be one of {self.choices}")
        return text


@define(kw_only=True)
class ChannelSelectionParameter(Parameter):
    """A set of active channel indices chosen from 0..channel_count-1."""

    channel_count: int = 9

    def coerce(self, value):
        indices = sorted({int(index) for index in value})
        for index in indices:
            if not 0 <= index < self.channel_count:
                raise ParameterError(f"{self.summary_label()} has out-of-range channel {index}")
        return indices


ParameterGroups = dict[str, list[Parameter]]


def flatten_parameters(groups: ParameterGroups) -> dict[str, Parameter]:
    """Return a {label: parameter} map across all groups."""
    return {parameter.label: parameter for group in groups.values() for parameter in group}


def coerce_parameter_values(groups: ParameterGroups, values: dict[str, Any]) -> dict[str, Any]:
    """Coerce a {label: raw} mapping against declarations, applying defaults for missing keys.

    Returns {label: coerced}. Raises ParameterError aggregating every field error.
    """
    declarations = flatten_parameters(groups)
    unknown = set(values) - set(declarations)
    if unknown:
        raise ParameterError(f"unknown parameters: {sorted(unknown)}")

    coerced: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for label, parameter in declarations.items():
        raw = values.get(label, parameter.default)
        try:
            coerced[label] = parameter.coerce(raw)
        except ParameterError as error:
            errors[label] = str(error)
    if errors:
        raise ParameterError("invalid parameters", errors)
    return coerced


def validate_parameter_groups(groups: ParameterGroups) -> None:
    """Check that parameter labels are unique across all groups."""
    seen: set[str] = set()
    for group in groups.values():
        for parameter in group:
            if parameter.label in seen:
                raise ParameterError(f"duplicate parameter label: {parameter.label}")
            seen.add(parameter.label)
