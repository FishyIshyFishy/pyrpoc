from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from pyrpoc.structs.parameters import (
    BaseParameter,
    ChannelSelectionParameter,
    CheckboxParameter,
    ChoiceParameter,
    NumberParameter,
    PathParameter,
    TextParameter,
)


# ---------------------------------------------------------------------------
# Widget construction, dispatched by parameter type
# ---------------------------------------------------------------------------

def create_widget(param: BaseParameter, parent: QWidget | None = None) -> QWidget:
    if isinstance(param, PathParameter):
        return build_path_widget(param, parent)
    if isinstance(param, TextParameter):
        widget = QLineEdit(parent)
        widget.setText("" if param.default is None else str(param.default))
        if param.tooltip:
            widget.setToolTip(param.tooltip)
        return widget
    if isinstance(param, NumberParameter):
        return build_number_widget(param, parent)
    if isinstance(param, CheckboxParameter):
        widget = QCheckBox(parent)
        widget.setChecked(bool(param.default))
        if param.tooltip:
            widget.setToolTip(param.tooltip)
        return widget
    if isinstance(param, ChoiceParameter):
        combo = QComboBox(parent)
        combo.addItems(param.choices)
        if param.default is not None:
            combo.setCurrentText(str(param.default))
        if param.tooltip:
            combo.setToolTip(param.tooltip)
        return combo
    if isinstance(param, ChannelSelectionParameter):
        return build_channel_widget(param, parent)
    raise TypeError(f"no widget builder for parameter type {type(param).__name__}")


def build_path_widget(param: PathParameter, parent: QWidget | None) -> QWidget:
    root = QWidget(parent)
    layout = QHBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)

    line_edit = QLineEdit(root)
    line_edit.setPlaceholderText("Path")
    if param.default is not None:
        line_edit.setText(str(param.default))
    if param.tooltip:
        line_edit.setToolTip(param.tooltip)

    browse_btn = QPushButton("Browse...", root)
    if param.tooltip:
        browse_btn.setToolTip(param.tooltip)

    def pick_file() -> None:
        current = line_edit.text().strip()
        if current:
            start = str(Path(current).expanduser())
            initial_dir = str(Path(start).parent)
            suggested = str(Path(start).name) if Path(start).suffix else "acquisition"
            initial_path = str(Path(initial_dir) / suggested)
        else:
            initial_path = str(Path.cwd())
        selected, _ = QFileDialog.getSaveFileName(root, "Select output path", initial_path, "All Files (*)")
        if selected:
            line_edit.setText(selected)

    browse_btn.clicked.connect(pick_file)
    layout.addWidget(line_edit, 1)
    layout.addWidget(browse_btn)
    return root


def build_number_widget(param: NumberParameter, parent: QWidget | None) -> QWidget:
    if param.is_integer:
        spin = QSpinBox(parent)
        spin.setMinimum(int(param.minimum if param.minimum is not None else -1_000_000))
        spin.setMaximum(int(param.maximum if param.maximum is not None else 1_000_000))
        spin.setSingleStep(int(param.step if param.step is not None else 1))
        spin.setValue(int(param.default) if param.default is not None else 0)
        if param.tooltip:
            spin.setToolTip(param.tooltip)
        return spin
    dspin = QDoubleSpinBox(parent)
    dspin.setDecimals(6)
    dspin.setMinimum(float(param.minimum if param.minimum is not None else -1e12))
    dspin.setMaximum(float(param.maximum if param.maximum is not None else 1e12))
    dspin.setSingleStep(float(param.step if param.step is not None else 0.1))
    dspin.setValue(float(param.default) if param.default is not None else 0.0)
    if param.tooltip:
        dspin.setToolTip(param.tooltip)
    return dspin


def build_channel_widget(param: ChannelSelectionParameter, parent: QWidget | None) -> QWidget:
    container = QWidget(parent)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
    active = set(param.default) if param.default else set()
    buttons: list[QToolButton] = []
    for i in range(param.num_channels):
        btn = QToolButton(container)
        btn.setCheckable(True)
        btn.setChecked(i in active)
        btn.setText(f"AI{i}")
        btn.setToolTip(f"Toggle AI{i}")
        btn.setStyleSheet(
            "QToolButton { padding: 4px 8px; border: 1px solid palette(mid);"
            " border-radius: 10px; background: palette(base); }"
            "QToolButton:checked { background: palette(highlight);"
            " color: palette(highlighted-text); border: 1px solid palette(highlight); font-weight: 700; }"
        )
        layout.addWidget(btn)
        buttons.append(btn)
    container._channel_buttons = buttons  # type: ignore[attr-defined]
    if param.tooltip:
        container.setToolTip(param.tooltip)
    return container


# ---------------------------------------------------------------------------
# Value read/write/connect, dispatched by parameter type
# ---------------------------------------------------------------------------

def channel_buttons(widget: QWidget) -> list[QToolButton]:
    return getattr(widget, "_channel_buttons", [])


def path_line_edit(widget: QWidget) -> QLineEdit:
    line_edit = widget.findChild(QLineEdit)
    if not isinstance(line_edit, QLineEdit):
        raise TypeError("PathParameter widget is missing its QLineEdit")
    return line_edit


def get_value(param: BaseParameter, widget: QWidget) -> Any:
    if isinstance(param, PathParameter):
        return path_line_edit(widget).text()
    if isinstance(param, TextParameter):
        assert isinstance(widget, QLineEdit)
        return widget.text()
    if isinstance(param, NumberParameter):
        if param.is_integer:
            assert isinstance(widget, QSpinBox)
            return int(widget.value())
        assert isinstance(widget, QDoubleSpinBox)
        return float(widget.value())
    if isinstance(param, CheckboxParameter):
        assert isinstance(widget, QCheckBox)
        return widget.isChecked()
    if isinstance(param, ChoiceParameter):
        assert isinstance(widget, QComboBox)
        return widget.currentText()
    if isinstance(param, ChannelSelectionParameter):
        return [i for i, btn in enumerate(channel_buttons(widget)) if btn.isChecked()]
    raise TypeError(f"no value getter for parameter type {type(param).__name__}")


def set_value(param: BaseParameter, widget: QWidget, value: Any) -> None:
    if isinstance(param, PathParameter):
        line_edit = path_line_edit(widget)
        line_edit.blockSignals(True)
        line_edit.setText("" if value is None else str(value))
        line_edit.blockSignals(False)
        return
    if isinstance(param, TextParameter):
        assert isinstance(widget, QLineEdit)
        widget.blockSignals(True)
        widget.setText("" if value is None else str(value))
        widget.blockSignals(False)
        return
    if isinstance(param, NumberParameter):
        if param.is_integer:
            assert isinstance(widget, QSpinBox)
            widget.blockSignals(True)
            widget.setValue(int(value) if value is not None else 0)
            widget.blockSignals(False)
        else:
            assert isinstance(widget, QDoubleSpinBox)
            widget.blockSignals(True)
            widget.setValue(float(value) if value is not None else 0.0)
            widget.blockSignals(False)
        return
    if isinstance(param, CheckboxParameter):
        assert isinstance(widget, QCheckBox)
        widget.blockSignals(True)
        widget.setChecked(bool(value))
        widget.blockSignals(False)
        return
    if isinstance(param, ChoiceParameter):
        assert isinstance(widget, QComboBox)
        widget.blockSignals(True)
        widget.setCurrentText(str(value))
        widget.blockSignals(False)
        return
    if isinstance(param, ChannelSelectionParameter):
        active = set(value) if value is not None else set()
        for i, btn in enumerate(channel_buttons(widget)):
            btn.blockSignals(True)
            btn.setChecked(i in active)
            btn.blockSignals(False)
        return
    raise TypeError(f"no value setter for parameter type {type(param).__name__}")


def connect_changed(param: BaseParameter, widget: QWidget, callback) -> None:
    if callback is None:
        return
    if isinstance(param, PathParameter):
        path_line_edit(widget).textChanged.connect(lambda *_: callback())
    elif isinstance(param, TextParameter) and isinstance(widget, QLineEdit):
        widget.textChanged.connect(lambda *_: callback())
    elif isinstance(param, NumberParameter):
        if param.is_integer and isinstance(widget, QSpinBox):
            widget.valueChanged.connect(lambda *_: callback())
        elif not param.is_integer and isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(lambda *_: callback())
    elif isinstance(param, CheckboxParameter) and isinstance(widget, QCheckBox):
        widget.toggled.connect(lambda *_: callback())
    elif isinstance(param, ChoiceParameter) and isinstance(widget, QComboBox):
        widget.currentTextChanged.connect(lambda *_: callback())
    elif isinstance(param, ChannelSelectionParameter):
        for btn in channel_buttons(widget):
            btn.toggled.connect(lambda *_: callback())


def format_summary(param: BaseParameter, widget: QWidget) -> str:
    if isinstance(param, ChannelSelectionParameter):
        active = get_value(param, widget)
        if not active:
            return "none"
        return ", ".join(f"AI{i}" for i in active)
    try:
        return str(get_value(param, widget))
    except Exception:
        return "-"
