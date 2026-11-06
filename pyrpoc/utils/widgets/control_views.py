"""``ControlView``: one runner control's widget, the views of the common controls, and the
lookup that finds the view for any ``Control``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from PyQt6.QtWidgets import QPushButton, QWidget

from pyrpoc.core.runner import Control
from pyrpoc.utils.controls import Button, Toggle

from .icons import asset_icon

ControlT = TypeVar("ControlT", bound=Control)


@dataclass
class ControlView:
    """One control's widget, what it says on hover, and how to stop it following its
    control."""

    widget: QWidget
    tooltip: str
    close: Callable[[], None]


# The view of each type of ``Control``, found for a control by its type or the nearest it
# derives from.
control_view_builders: dict[type[Control], Callable[[Any, QWidget], ControlView]] = {}


def control_view_for(
    control_type: type[ControlT],
) -> Callable[
    [Callable[[ControlT, QWidget], ControlView]], Callable[[ControlT, QWidget], ControlView]
]:
    """Register the decorated function as the view of ``control_type``."""

    def register(
        build: Callable[[ControlT, QWidget], ControlView],
    ) -> Callable[[ControlT, QWidget], ControlView]:
        if control_type in control_view_builders:
            raise ValueError(f"{control_type.__name__} already has a view")
        control_view_builders[control_type] = build
        return build

    return register


def build_control_view(control: Control, parent: QWidget) -> ControlView:
    """The view of ``control``'s type."""
    build = next(
        (
            control_view_builders[cls]
            for cls in type(control).__mro__
            if cls in control_view_builders
        ),
        None,
    )
    if build is None:
        raise TypeError(f"no view is registered for {type(control).__name__}")
    return build(control, parent)


def push_button(label: str, icon: str | None, parent: QWidget) -> QPushButton:
    """A button showing ``icon``, or ``label`` when it has none."""
    button = QPushButton(parent)
    if icon is not None:
        button.setIcon(asset_icon(icon))
    else:
        button.setText(label)
    return button


@control_view_for(Button)
def button_view(control: Button, parent: QWidget) -> ControlView:
    button = push_button(control.label, control.icon, parent)
    button.clicked.connect(control.press)
    return ControlView(button, control.tooltip or control.label, lambda: None)


@control_view_for(Toggle)
def toggle_view(control: Toggle, parent: QWidget) -> ControlView:
    button = push_button(control.label, control.icon, parent)
    button.setCheckable(True)
    button.setChecked(control.checked)
    # ``clicked`` rather than ``toggled``, so the runner moving its own toggle is shown without
    # being mistaken for the user.
    button.clicked.connect(control.flip)

    def follow() -> None:
        if button.isChecked() != control.checked:
            button.setChecked(control.checked)

    control.watch(follow)
    return ControlView(button, control.tooltip or control.label, lambda: control.unwatch(follow))
