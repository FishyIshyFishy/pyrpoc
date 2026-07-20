"""Collapsible parameter-group cards — the old acquisition-tab look.

One card per parameter section (scan / daq / ...), with a form inside and a live
one-line summary shown while collapsed.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QWidget

from pyrpoc_next.gui.widgets.cards import BaseCardWidget
from pyrpoc_next.gui.widgets.parameter_widgets import (
    connect_changed,
    create_widget,
    read_value,
    set_value,
)
from pyrpoc_next.structs.parameters import ParameterGroups, ParameterValue


class ParameterGroupCard(BaseCardWidget):
    """A collapsible card holding one parameter group's editors."""

    def __init__(self, section: str, params, values):
        super().__init__(None, section.capitalize())
        self.set_toggle_visible(False)
        self.expand_requested.connect(lambda _: self.set_expanded(not self.is_expanded()))
        self.widgets: dict = {}

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        current = {value.label: value.value for value in (values or [])}
        for param in params:
            widget = create_widget(param)
            if param.tooltip:
                widget.setToolTip(param.tooltip)
            value = current.get(param.label, param.default)
            if value is not None:
                set_value(param, widget, value)
            self.widgets[param.label] = (param, widget)
            form.addRow(param.summary_label(), widget)
            connect_changed(widget, self.update_summary)
        self.set_body_widget(form_widget)
        self.update_summary()

    def update_summary(self) -> None:
        parts = [f"{param.summary_label()}: {read_value(param, widget)}"
                 for _, (param, widget) in self.widgets.items()]
        self.set_description("   |   ".join(parts))

    def values(self) -> list[ParameterValue]:
        return [ParameterValue(label, read_value(param, widget))
                for label, (param, widget) in self.widgets.items()]


def group_cards(groups: ParameterGroups, values=None) -> list[ParameterGroupCard]:
    """Build one card per parameter group."""
    return [ParameterGroupCard(section, params, values) for section, params in groups.items()]


def collect_values(cards: list[ParameterGroupCard]) -> list[ParameterValue]:
    """Read all values back out of a list of group cards."""
    result: list[ParameterValue] = []
    for card in cards:
        result.extend(card.values())
    return result
