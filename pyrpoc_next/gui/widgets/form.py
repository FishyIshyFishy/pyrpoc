"""A grouped form of parameter editors, built from a parameter_groups declaration."""

from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout, QWidget

from pyrpoc_next.gui.widgets.parameter_widgets import create_widget, read_value, set_value
from pyrpoc_next.structs.parameters import ParameterGroups, ParameterValue


class ParameterForm(QWidget):
    """Renders a parameter_groups dict and reads the values back out."""

    def __init__(self, groups: ParameterGroups, values: list[ParameterValue] | None = None):
        super().__init__()
        self.entries: dict[str, tuple] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        current = {value.label: value.value for value in (values or [])}
        for section, params in groups.items():
            box = QGroupBox(section.capitalize())
            form = QFormLayout(box)
            for parameter in params:
                widget = create_widget(parameter)
                value = current.get(parameter.label, parameter.default)
                if value is not None:
                    set_value(parameter, widget, value)
                form.addRow(parameter.summary_label(), widget)
                self.entries[parameter.label] = (parameter, widget)
            layout.addWidget(box)

    def values(self) -> list[ParameterValue]:
        return [ParameterValue(label, read_value(parameter, widget))
                for label, (parameter, widget) in self.entries.items()]
