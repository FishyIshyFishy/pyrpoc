"""The dockable panels: instruments, displays, and the routine editor / settings menu."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc_next.acquisition import modality_registry, modifier_registry
from pyrpoc_next.gui.displays import display_registry
from pyrpoc_next.gui.widgets.form import ParameterForm
from pyrpoc_next.instruments import instrument_registry
from pyrpoc_next.structs.routine import ModifierSlot, Routine, RoutineBlock


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)


class InstrumentsPanel(QWidget):
    """Add/remove instruments and test their connection."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        for key in instrument_registry.available():
            self.combo.addItem(key.value, key)
        add = QPushButton("Add Instrument")
        add.clicked.connect(self.add)
        layout.addWidget(self.combo)
        layout.addWidget(add)
        self.rows = QVBoxLayout()
        layout.addLayout(self.rows)
        layout.addStretch()

    def add(self) -> None:
        instrument = instrument_registry.create(self.combo.currentData())
        self.state.instruments.append(instrument)
        self.add_row(instrument)

    def add_row(self, instrument) -> None:
        row = QWidget()
        layout = QHBoxLayout(row)
        label = QLabel(instrument.summary())
        test = QPushButton("Test")
        remove = QPushButton("Remove")
        test.clicked.connect(lambda: (instrument.test_connection(), label.setText(instrument.summary())))
        remove.clicked.connect(lambda: (self.state.instruments.remove(instrument), row.setParent(None)))
        layout.addWidget(label)
        layout.addWidget(test)
        layout.addWidget(remove)
        self.rows.addWidget(row)


class DisplaysPanel(QWidget):
    """Add displays, which appear as their own docks."""

    def __init__(self, state, add_dock: Callable[[QWidget, str], None]):
        super().__init__()
        self.state = state
        self.add_dock = add_dock
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        for key in display_registry.available():
            self.combo.addItem(key.value, key)
        add = QPushButton("Add Display")
        add.clicked.connect(self.add)
        layout.addWidget(self.combo)
        layout.addWidget(add)
        layout.addStretch()

    def add(self) -> None:
        key = self.combo.currentData()
        display = display_registry.create(key)
        self.state.displays.append(display)
        self.add_dock(display, display.manifest.display_name)


class BlockEditor(QGroupBox):
    """One routine block: a modality, its parameters, and its available modifiers."""

    def __init__(self):
        super().__init__("Block")
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.modality_combo = QComboBox()
        for key in modality_registry.available():
            self.modality_combo.addItem(modality_registry.manifest(key).display_name, key)
        self.active_radio = QRadioButton("Active")
        top.addWidget(QLabel("Modality"))
        top.addWidget(self.modality_combo)
        top.addWidget(self.active_radio)
        layout.addLayout(top)

        self.form_area = QVBoxLayout()
        layout.addLayout(self.form_area)
        self.modifier_area = QVBoxLayout()
        layout.addLayout(self.modifier_area)

        self.form: ParameterForm | None = None
        self.modifier_widgets: dict = {}
        self.modality_combo.currentIndexChanged.connect(self.rebuild)
        self.rebuild()

    def modality_key(self):
        return self.modality_combo.currentData()

    def rebuild(self) -> None:
        clear_layout(self.form_area)
        clear_layout(self.modifier_area)
        self.modifier_widgets = {}

        manifest = modality_registry.manifest(self.modality_key())
        self.form = ParameterForm(manifest.parameter_groups)
        self.form_area.addWidget(self.form)

        for modifier_key in manifest.realizable_modifiers:
            modifier_manifest = modifier_registry.manifest(modifier_key)
            box = QGroupBox(f"Modifier: {modifier_manifest.display_name}")
            box_layout = QVBoxLayout(box)
            enable = QCheckBox("Enabled")
            form = ParameterForm(modifier_manifest.parameter_groups)
            box_layout.addWidget(enable)
            box_layout.addWidget(form)
            self.modifier_area.addWidget(box)
            self.modifier_widgets[modifier_key] = (enable, form)

    def to_block(self) -> RoutineBlock:
        modifiers = [
            ModifierSlot(key=key, available=True, enabled=enable.isChecked(), values=form.values())
            for key, (enable, form) in self.modifier_widgets.items()
        ]
        return RoutineBlock(modality=self.modality_key(), values=self.form.values(), modifiers=modifiers)


class RoutinePanel(QWidget):
    """The routine editor + settings menu: edit blocks, pick the active one, play/stop."""

    def __init__(self, controller, bridge):
        super().__init__()
        self.controller = controller
        self.state = controller.state
        layout = QVBoxLayout(self)

        self.blocks_layout = QVBoxLayout()
        self.active_group = QButtonGroup(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder.setLayout(self.blocks_layout)
        scroll.setWidget(holder)
        layout.addWidget(scroll)

        add_block = QPushButton("Add Block")
        add_block.clicked.connect(self.add_block)
        layout.addWidget(add_block)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.play_button.clicked.connect(self.play)
        self.stop_button.clicked.connect(self.controller.stop)
        controls.addWidget(self.play_button)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)

        self.status = QLabel("idle")
        layout.addWidget(self.status)

        self.editors: list[BlockEditor] = []
        self.add_block()

        bridge.started.connect(self.on_started)
        bridge.stopped.connect(self.on_stopped)
        bridge.errored.connect(self.on_error)

    def add_block(self) -> None:
        editor = BlockEditor()
        self.active_group.addButton(editor.active_radio)
        if not self.editors:
            editor.active_radio.setChecked(True)
        self.editors.append(editor)
        self.blocks_layout.addWidget(editor)

    def sync_routine(self) -> None:
        blocks = [editor.to_block() for editor in self.editors]
        active = next((index for index, editor in enumerate(self.editors)
                       if editor.active_radio.isChecked()), 0)
        self.state.routine = Routine(name="session", blocks=blocks, active_index=active)

    def play(self) -> None:
        self.sync_routine()
        report = self.controller.play()
        if report.blocked:
            QMessageBox.warning(self, "Cannot start acquisition",
                                "\n".join(issue.message for issue in report.issues))

    def on_started(self) -> None:
        self.status.setText("acquiring")
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def on_stopped(self) -> None:
        self.status.setText("idle")
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def on_error(self, message: str) -> None:
        self.status.setText(f"error: {message}")
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
