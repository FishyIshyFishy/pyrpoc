from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Action, Parameter, ParameterGroups


@dataclass
class ParameterValidationError(Exception):
    message: str
    errors: dict[str, str]

    def __str__(self) -> str:
        if not self.errors:
            return self.message
        return f"{self.message}: {self.errors}"


def _validate_single_parameter(param: Parameter) -> None:
    if not isinstance(param.label, str) or not param.label.strip():
        raise TypeError("parameter label must be a non-empty string")
    if not isinstance(param.param_type, type):
        raise TypeError(f"parameter '{param.label}' param_type must be a type")

    allowed_types = {int, float, bool, str, Path}
    if param.param_type not in allowed_types:
        raise TypeError(f"parameter '{param.label}' has unsupported type {param.param_type!r}")

    if param.minimum is not None or param.maximum is not None or param.step is not None:
        if param.param_type not in {int, float}:
            raise TypeError(
                f"parameter '{param.label}' uses numeric limits/step but type is not int or float"
            )

    if param.minimum is not None and param.maximum is not None and param.minimum > param.maximum:
        raise ValueError(f"parameter '{param.label}' has minimum > maximum")

    if param.choices is not None:
        if param.param_type is not str:
            raise TypeError(f"parameter '{param.label}' choices are only supported for str parameters")
        if len(param.choices) == 0:
            raise ValueError(f"parameter '{param.label}' choices cannot be empty")


def validate_parameter_groups(groups: ParameterGroups) -> None:
    if not isinstance(groups, dict):
        raise TypeError("parameter groups must be a dictionary")

    seen_labels: set[str] = set()
    for group_name, params in groups.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise TypeError("parameter group names must be non-empty strings")
        if not isinstance(params, list):
            raise TypeError(f"group '{group_name}' must be a list of Parameter objects")

        for param in params:
            if not isinstance(param, Parameter):
                raise TypeError(f"group '{group_name}' contains a non-Parameter entry")
            _validate_single_parameter(param)
            if param.label in seen_labels:
                raise ValueError(f"duplicate parameter label '{param.label}' across groups")
            seen_labels.add(param.label)


def validate_action_list(actions: list[Action]) -> None:
    if not isinstance(actions, list):
        raise TypeError("actions must be a list")

    seen_action_labels: set[str] = set()
    seen_method_names: set[str] = set()
    for action in actions:
        if not isinstance(action, Action):
            raise TypeError("actions must contain Action objects")
        if not isinstance(action.label, str) or not action.label.strip():
            raise TypeError("action label must be a non-empty string")
        if not isinstance(action.method_name, str) or not action.method_name.strip():
            raise TypeError(f"action '{action.label}' method_name must be a non-empty string")

        if action.label in seen_action_labels:
            raise ValueError(f"duplicate action label '{action.label}'")
        if action.method_name in seen_method_names:
            raise ValueError(f"duplicate action method_name '{action.method_name}'")
        seen_action_labels.add(action.label)
        seen_method_names.add(action.method_name)

        param_labels: set[str] = set()
        for param in action.parameters:
            if not isinstance(param, Parameter):
                raise TypeError(f"action '{action.label}' contains a non-Parameter entry")
            _validate_single_parameter(param)
            if param.label in param_labels:
                raise ValueError(f"duplicate parameter label '{param.label}' in action '{action.label}'")
            param_labels.add(param.label)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"cannot convert {value!r} to bool")


def _coerce_parameter_value(param: Parameter, value: Any) -> Any:
    if param.param_type is int:
        coerced: Any = int(value)
    elif param.param_type is float:
        coerced = float(value)
    elif param.param_type is bool:
        coerced = _to_bool(value)
    elif param.param_type is str:
        coerced = str(value)
    elif param.param_type is Path:
        coerced = Path(value)
    else:
        raise ValueError(f"unsupported parameter type for '{param.label}'")

    if param.choices is not None and str(coerced) not in param.choices:
        raise ValueError(f"must be one of {param.choices}")

    if isinstance(coerced, (int, float)):
        if param.minimum is not None and coerced < param.minimum:
            raise ValueError(f"must be >= {param.minimum}")
        if param.maximum is not None and coerced > param.maximum:
            raise ValueError(f"must be <= {param.maximum}")

    return coerced


def coerce_parameter_values(groups: ParameterGroups, raw: dict[str, Any] | None) -> dict[str, Any]:
    validate_parameter_groups(groups)
    raw_values = raw or {}

    params_by_label: dict[str, Parameter] = {}
    for params in groups.values():
        for param in params:
            params_by_label[param.label] = param

    unknown = [key for key in raw_values.keys() if key not in params_by_label]
    if unknown:
        raise ParameterValidationError(
            "unknown parameters provided",
            {label: "unknown parameter" for label in unknown},
        )

    result: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for label, param in params_by_label.items():
        candidate = raw_values[label] if label in raw_values else param.default
        if candidate is None and param.required:
            errors[label] = "value is required"
            continue

        if candidate is None:
            result[label] = None
            continue

        try:
            result[label] = _coerce_parameter_value(param, candidate)
        except Exception as exc:
            errors[label] = str(exc)

    if errors:
        raise ParameterValidationError("parameter validation failed", errors)
    return result


def coerce_action_values(action: Action, raw: dict[str, Any] | None) -> dict[str, Any]:
    validate_action_list([action])
    raw_values = raw or {}

    params_by_label = {param.label: param for param in action.parameters}
    unknown = [key for key in raw_values.keys() if key not in params_by_label]
    if unknown:
        raise ParameterValidationError(
            f"unknown parameters for action '{action.label}'",
            {label: "unknown parameter" for label in unknown},
        )

    result: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for label, param in params_by_label.items():
        candidate = raw_values[label] if label in raw_values else param.default
        if candidate is None and param.required:
            errors[label] = "value is required"
            continue

        if candidate is None:
            result[label] = None
            continue

        try:
            result[label] = _coerce_parameter_value(param, candidate)
        except Exception as exc:
            errors[label] = str(exc)

    if errors:
        raise ParameterValidationError(
            f"action parameter validation failed for '{action.label}'",
            errors,
        )

    return result
