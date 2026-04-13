from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from .contracts import Action, ParameterGroups


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

    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        raise NotImplementedError

    def coerce(self, value: Any) -> Any:
        raise NotImplementedError

    def get_value(self, widget: QWidget) -> Any:
        raise NotImplementedError

    def set_value(self, widget: QWidget, value: Any) -> None:
        raise NotImplementedError

    def connect_changed(self, widget: QWidget, callback) -> None:
        if callback is None:
            return
        if not callable(callback):
            raise TypeError("callback must be callable")

    def format_summary(self, widget: QWidget) -> str:
        """Return a short human-readable summary of the current widget value."""
        try:
            return str(self.get_value(widget))
        except Exception:
            return "—"

    def validate_default(self, value: Any) -> None:
        if value is None:
            return
        self.coerce(value)


class TextParameter(BaseParameter):
    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        widget = QLineEdit(parent)
        widget.setText("" if self.default is None else str(self.default))
        if self.tooltip:
            widget.setToolTip(self.tooltip)
        return widget

    def coerce(self, value: Any) -> str:
        return str(value)

    def get_value(self, widget: QWidget) -> Any:
        if not isinstance(widget, QLineEdit):
            raise TypeError("TextParameter expects a QLineEdit")
        return widget.text()

    def set_value(self, widget: QWidget, value: Any) -> None:
        if not isinstance(widget, QLineEdit):
            raise TypeError("TextParameter expects a QLineEdit")
        widget.blockSignals(True)
        widget.setText("" if value is None else str(value))
        widget.blockSignals(False)

    def connect_changed(self, widget: QWidget, callback) -> None:
        super().connect_changed(widget, callback)
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(lambda *_: callback())


class PathParameter(TextParameter):
    browse_button_label: str = "Browse..."
    file_dialog_title: str = "Select output path"
    file_dialog_filter: str = "All Files (*)"

    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        root = QWidget(parent)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)

        line_edit = QLineEdit(root)
        line_edit.setPlaceholderText("Path")
        if self.default is not None:
            line_edit.setText(str(self.default))
        if self.tooltip:
            line_edit.setToolTip(self.tooltip)

        browse_btn = QPushButton(self.browse_button_label, root)
        if self.tooltip:
            browse_btn.setToolTip(self.tooltip)

        def _pick_file() -> None:
            current = line_edit.text().strip()
            if current:
                start = str(Path(current).expanduser())
                initial_dir = str(Path(start).parent)
                if Path(start).suffix:
                    suggested_name = str(Path(start).name)
                else:
                    suggested_name = "acquisition"
                initial_path = str(Path(initial_dir) / suggested_name)
            else:
                initial_path = str(Path.cwd())

            selected, _ = QFileDialog.getSaveFileName(
                root,
                self.file_dialog_title,
                initial_path,
                self.file_dialog_filter,
            )
            if selected:
                line_edit.setText(selected)

        browse_btn.clicked.connect(_pick_file)

        layout.addWidget(line_edit, 1)
        layout.addWidget(browse_btn)
        return root

    def _get_line_edit(self, widget: QWidget) -> QLineEdit:
        if not isinstance(widget, QWidget):
            raise TypeError("PathParameter expects a QWidget container")
        line_edit = widget.findChild(QLineEdit)
        if not isinstance(line_edit, QLineEdit):
            raise TypeError("PathParameter widget is missing its QLineEdit")
        return line_edit

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

    def get_value(self, widget: QWidget) -> Any:
        return self._get_line_edit(widget).text()

    def set_value(self, widget: QWidget, value: Any) -> None:
        line_edit = self._get_line_edit(widget)
        line_edit.blockSignals(True)
        line_edit.setText("" if value is None else str(value))
        line_edit.blockSignals(False)

    def connect_changed(self, widget: QWidget, callback) -> None:
        super().connect_changed(widget, callback)
        if callback is None:
            return
        line_edit = self._get_line_edit(widget)
        line_edit.textChanged.connect(lambda *_: callback())


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

    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        if self.is_integer:
            widget = QSpinBox(parent)
            widget.setMinimum(int(self.minimum if self.minimum is not None else -1_000_000))
            widget.setMaximum(int(self.maximum if self.maximum is not None else 1_000_000))
            widget.setSingleStep(int(self.step if self.step is not None else 1))
            widget.setValue(int(self.default) if self.default is not None else 0)
        else:
            widget = QDoubleSpinBox(parent)
            widget.setDecimals(6)
            widget.setMinimum(float(self.minimum if self.minimum is not None else -1e12))
            widget.setMaximum(float(self.maximum if self.maximum is not None else 1e12))
            widget.setSingleStep(float(self.step if self.step is not None else 0.1))
            widget.setValue(float(self.default) if self.default is not None else 0.0)
        if self.tooltip:
            widget.setToolTip(self.tooltip)
        return widget

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

    def get_value(self, widget: QWidget) -> Any:
        if self.is_integer:
            if not isinstance(widget, QSpinBox):
                raise TypeError("NumberParameter expects a QSpinBox for integer values")
            return int(widget.value())
        if not isinstance(widget, QDoubleSpinBox):
            raise TypeError("NumberParameter expects a QDoubleSpinBox for floating values")
        return float(widget.value())

    def set_value(self, widget: QWidget, value: Any) -> None:
        if self.is_integer:
            if not isinstance(widget, QSpinBox):
                raise TypeError("NumberParameter expects a QSpinBox for integer values")
            widget.blockSignals(True)
            widget.setValue(int(value) if value is not None else 0)
            widget.blockSignals(False)
            return
        if not isinstance(widget, QDoubleSpinBox):
            raise TypeError("NumberParameter expects a QDoubleSpinBox for floating values")
        widget.blockSignals(True)
        widget.setValue(float(value) if value is not None else 0.0)
        widget.blockSignals(False)

    def connect_changed(self, widget: QWidget, callback) -> None:
        super().connect_changed(widget, callback)
        if self.is_integer and isinstance(widget, QSpinBox):
            widget.valueChanged.connect(lambda *_: callback())
        elif not self.is_integer and isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(lambda *_: callback())


class CheckboxParameter(BaseParameter):
    default: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.default is not None and not isinstance(self.default, bool):
            raise TypeError("CheckboxParameter default must be bool")

    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        widget = QCheckBox(parent)
        widget.setChecked(bool(self.default))
        if self.tooltip:
            widget.setToolTip(self.tooltip)
        return widget

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

    def get_value(self, widget: QWidget) -> Any:
        if not isinstance(widget, QCheckBox):
            raise TypeError("CheckboxParameter expects a QCheckBox")
        return widget.isChecked()

    def set_value(self, widget: QWidget, value: Any) -> None:
        if not isinstance(widget, QCheckBox):
            raise TypeError("CheckboxParameter expects a QCheckBox")
        widget.blockSignals(True)
        widget.setChecked(bool(value))
        widget.blockSignals(False)

    def connect_changed(self, widget: QWidget, callback) -> None:
        super().connect_changed(widget, callback)
        if isinstance(widget, QCheckBox):
            widget.toggled.connect(lambda *_: callback())


class ChoiceParameter(BaseParameter):
    choices: list[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.choices, list) or len(self.choices) == 0:
            raise ValueError("choices must be a non-empty list")
        if not all(isinstance(choice, str) for choice in self.choices):
            raise TypeError("all choices must be strings")

    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        widget = QComboBox(parent)
        widget.addItems(self.choices)
        if self.default is not None:
            widget.setCurrentText(str(self.default))
        if self.tooltip:
            widget.setToolTip(self.tooltip)
        return widget

    def coerce(self, value: Any) -> str:
        text = str(value)
        if text not in self.choices:
            raise ValueError(f"must be one of {self.choices}")
        return text

    def get_value(self, widget: QWidget) -> Any:
        if not isinstance(widget, QComboBox):
            raise TypeError("ChoiceParameter expects a QComboBox")
        return widget.currentText()

    def set_value(self, widget: QWidget, value: Any) -> None:
        if not isinstance(widget, QComboBox):
            raise TypeError("ChoiceParameter expects a QComboBox")
        widget.blockSignals(True)
        widget.setCurrentText(str(value))
        widget.blockSignals(False)

    def connect_changed(self, widget: QWidget, callback) -> None:
        super().connect_changed(widget, callback)
        if isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(lambda *_: callback())


class ChannelSelectionParameter(BaseParameter):
    """A parameter that renders a row of toggleable AI channel buttons.

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

    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        active_set = set(self.default) if self.default else set()
        buttons: list[QToolButton] = []
        for i in range(self.num_channels):
            btn = QToolButton(container)
            btn.setCheckable(True)
            btn.setChecked(i in active_set)
            btn.setText(f"AI{i}")
            btn.setToolTip(f"Toggle AI{i}")
            btn.setStyleSheet(
                "QToolButton {"
                "padding: 4px 8px;"
                "border: 1px solid palette(mid);"
                "border-radius: 10px;"
                "background: palette(base);"
                "}"
                "QToolButton:checked {"
                "background: palette(highlight);"
                "color: palette(highlighted-text);"
                "border: 1px solid palette(highlight);"
                "font-weight: 700;"
                "}"
            )
            layout.addWidget(btn)
            buttons.append(btn)
        container._channel_buttons = buttons  # type: ignore[attr-defined]
        if self.tooltip:
            container.setToolTip(self.tooltip)
        return container

    def _get_buttons(self, widget: QWidget) -> list[QToolButton]:
        return getattr(widget, "_channel_buttons", [])

    def get_value(self, widget: QWidget) -> list[int]:
        return [i for i, btn in enumerate(self._get_buttons(widget)) if btn.isChecked()]

    def set_value(self, widget: QWidget, value: Any) -> None:
        active = set(value) if value is not None else set()
        for i, btn in enumerate(self._get_buttons(widget)):
            btn.blockSignals(True)
            btn.setChecked(i in active)
            btn.blockSignals(False)

    def coerce(self, value: Any) -> list[int]:
        if isinstance(value, list):
            return sorted(set(int(v) for v in value))
        raise TypeError("channel selection must be a list of integer channel indices")

    def connect_changed(self, widget: QWidget, callback) -> None:
        super().connect_changed(widget, callback)
        if callback is None:
            return
        for btn in self._get_buttons(widget):
            btn.toggled.connect(lambda *_: callback())

    def format_summary(self, widget: QWidget) -> str:
        active = self.get_value(widget)
        if not active:
            return "none"
        return ", ".join(f"AI{i}" for i in active)


def _validate_single_parameter(param: BaseParameter) -> None:
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

        action_param_labels: set[str] = set()
        for param in action.parameters:
            _validate_single_parameter(param)

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
