from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParameterValue:
    label: str
    value: Any


@dataclass(frozen=True)
class AcquisitionParameters:
    """Base for the frozen, typed per-preset parameter dataclasses.

    Holds the acquisition fields every scan preset shares; subclasses add their
    own scan/DAQ-specific fields.
    """

    save_enabled: bool
    save_path: str
    num_frames: int


@dataclass
class Action:
    label: str
    method_name: str
    parameters: list["BaseParameter"] = field(default_factory=list)
    tooltip: str = ""
    dangerous: bool = False
    confirm_text: str | None = None


ParameterGroups = dict[str, list["BaseParameter"]]


@dataclass
class ParameterValidationError(Exception):
    message: str
    errors: dict[str, str]

    def __str__(self) -> str:
        if not self.errors:
            return self.message
        return f"{self.message}: {self.errors}"


NumberType = int | float


@dataclass
class BaseParameter:
    """Backend-neutral parameter specification and validation.

    Holds the parameter's identity, default, and constraints, and knows how to
    coerce a raw value into its validated type. Widget construction lives in
    ``gui`` (keyed by parameter type); this layer stays Qt-free.
    """

    label: str
    default: Any = None
    required: bool = True
    tooltip: str = ""
    display_label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise TypeError("parameter label must be a non-empty string")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")
        if self.tooltip is not None and not isinstance(self.tooltip, str):
            raise TypeError("tooltip must be a string")
        if not isinstance(self.display_label, str):
            raise TypeError("display_label must be a string")
        if not self.display_label.strip():
            self.display_label = self.label

    def coerce(self, value: Any) -> Any:
        raise NotImplementedError

    def validate_default(self, value: Any) -> None:
        if value is None:
            return
        self.coerce(value)


class TextParameter(BaseParameter):
    def coerce(self, value: Any) -> str:
        return str(value)


class PathParameter(TextParameter):
    def coerce(self, value: Any) -> Path:
        if value is None:
            raise ValueError("path cannot be empty")
        if isinstance(value, Path):
            text = str(value)
        elif isinstance(value, str):
            text = value
        else:
            raise TypeError("path must be text")

        if not text.strip():
            raise ValueError("path cannot be empty")
        if text.rstrip().endswith(("\\", "/")):
            raise ValueError("path must include a filename")

        return Path(str(value)).expanduser()


@dataclass
class NumberParameter(BaseParameter):
    minimum: NumberType | None = None
    maximum: NumberType | None = None
    step: NumberType | None = None
    number_type: type[NumberType] = float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.number_type not in {int, float}:
            raise TypeError("number_type must be int or float")
        if self.minimum is not None and not isinstance(self.minimum, (int, float)):
            raise TypeError("minimum must be a number")
        if self.maximum is not None and not isinstance(self.maximum, (int, float)):
            raise TypeError("maximum must be a number")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        if self.step is not None and not isinstance(self.step, (int, float)):
            raise TypeError("step must be a number")
        if self.default is not None and not isinstance(self.default, self.number_type):
            # allow float defaults for int and int defaults for float via coercion
            if self.number_type is int and isinstance(self.default, bool):
                raise TypeError("int default cannot be bool")
            if self.number_type is float and not isinstance(self.default, (int, float)):
                raise TypeError("float default must be numeric")

    @property
    def is_integer(self) -> bool:
        return self.number_type is int

    def coerce(self, value: Any) -> int | float:
        if self.number_type is int:
            coerced: int | float = int(value)
        else:
            coerced = float(value)

        if self.minimum is not None and coerced < self.minimum:
            raise ValueError(f"must be >= {self.minimum}")
        if self.maximum is not None and coerced > self.maximum:
            raise ValueError(f"must be <= {self.maximum}")
        return coerced


@dataclass
class CheckboxParameter(BaseParameter):
    default: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.default is not None and not isinstance(self.default, bool):
            raise TypeError("CheckboxParameter default must be bool")

    def coerce(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise ValueError("cannot convert value to bool")


@dataclass
class ChoiceParameter(BaseParameter):
    choices: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.choices, list) or len(self.choices) == 0:
            raise ValueError("choices must be a non-empty list")
        if not all(isinstance(choice, str) for choice in self.choices):
            raise TypeError("all choices must be strings")

    def coerce(self, value: Any) -> str:
        text = str(value)
        if text not in self.choices:
            raise ValueError(f"must be one of {self.choices}")
        return text


class ChannelSelectionParameter(BaseParameter):
    """A parameter selecting a set of active AI channels.

    The stored value is a sorted ``list[int]`` of *active* channel indices.
    """

    def __init__(
        self,
        label: str,
        num_channels: int = 9,
        default: list[int] | None = None,
        required: bool = False,
        tooltip: str = "",
        display_label: str = "",
    ) -> None:
        if default is None:
            default = list(range(num_channels))
        super().__init__(
            label=label,
            default=default,
            required=required,
            tooltip=tooltip,
            display_label=display_label,
        )
        self.num_channels = num_channels

    def coerce(self, value: Any) -> list[int]:
        if isinstance(value, list):
            return sorted(set(int(v) for v in value))
        raise TypeError("channel selection must be a list of integer channel indices")


def validate_single_parameter(param: BaseParameter) -> None:
    if not isinstance(param, BaseParameter):
        raise TypeError(f"parameter '{getattr(param, 'label', '<unknown>')}' is not a BaseParameter")
    param.validate_default(param.default)


def validate_parameter_groups(groups: ParameterGroups) -> None:
    if not isinstance(groups, dict):
        raise TypeError("parameter groups must be a dictionary")

    seen_labels: set[str] = set()
    for group_name, params in groups.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise TypeError("parameter group names must be non-empty strings")
        if not isinstance(params, list):
            raise TypeError(f"group '{group_name}' must be a list of BaseParameter objects")

        for param in params:
            validate_single_parameter(param)
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

        action_param_labels: set[str] = set()
        for param in action.parameters:
            validate_single_parameter(param)

            if param.label in action_param_labels:
                raise ValueError(f"duplicate parameter label '{param.label}' in action '{action.label}'")
            action_param_labels.add(param.label)


def coerce_parameter_values(groups: ParameterGroups, raw: dict[str, Any] | None) -> dict[str, Any]:
    validate_parameter_groups(groups)
    raw_values = raw or {}

    params_by_label: dict[str, BaseParameter] = {}
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
            result[label] = param.coerce(candidate)
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
            result[label] = param.coerce(candidate)
        except Exception as exc:
            errors[label] = str(exc)

    if errors:
        raise ParameterValidationError(f"action parameter validation failed for '{action.label}'", errors)

    return result
