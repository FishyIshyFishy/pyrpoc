"""``ShortcutsDialog``: every command with the key that runs it, to rebind, clear or reset."""

from __future__ import annotations

from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.utils.widgets.empty_state import EmptyState
from pyrpoc.utils.widgets.table import ListTable, horizontal_header

from .commands import Command, ShortcutController


class ShortcutsDialog(QDialog):
    """Edits the controller's keys, applying them only on OK and only while no two commands share
    a key."""

    def __init__(self, controller: ShortcutController, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(520, 420)
        self.controller = controller
        self.edits: dict[Command, QKeySequenceEdit] = {}
        # Built first so conflict checks can reach OK; laid out last.
        self.buttons = self.build_buttons()
        root = QVBoxLayout(self)
        hint = QLabel("Click a shortcut and press the keys to use. Changes apply on OK.", self)
        hint.setStyleSheet("color: palette(mid);")
        root.addWidget(hint)
        commands = controller.commands()
        if commands:
            root.addWidget(self.build_table(commands), 1)
        else:
            root.addWidget(
                EmptyState("No shortcuts yet", "Commands with shortcuts will be listed here.", self)
            )
            root.addStretch(1)
        self.conflict = QLabel("", self)
        self.conflict.setStyleSheet("color: #e06c6c;")
        self.conflict.setWordWrap(True)
        root.addWidget(self.conflict)
        root.addWidget(self.buttons)
        self.standard_button(QDialogButtonBox.StandardButton.RestoreDefaults).setEnabled(
            bool(self.edits)
        )
        self.check_conflicts()

    def build_table(self, commands: list[Command]) -> ListTable:
        table = ListTable(["Command", "Shortcut", ""], self)
        table.setRowCount(len(commands))
        header = horizontal_header(table)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(1, 180)
        for row, command in enumerate(commands):
            table.set_cell(row, 0, command.label)
            edit = QKeySequenceEdit(self.controller.sequence(command), table)
            edit.setMaximumSequenceLength(1)
            edit.setClearButtonEnabled(True)
            edit.keySequenceChanged.connect(lambda _sequence: self.check_conflicts())
            table.setCellWidget(row, 1, edit)
            reset = QPushButton("Default", table)
            reset.setToolTip(f"Reset to {command.default or 'none'}")
            reset.clicked.connect(lambda _checked=False, c=command: self.reset(c))
            table.setCellWidget(row, 2, reset)
            # A cell widget does not size its column, so the column is fitted to the button.
            table.setColumnWidth(2, reset.sizeHint().width() + 8)
            self.edits[command] = edit
        return table

    def build_buttons(self) -> QDialogButtonBox:
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        restore = buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        if restore is None:
            raise RuntimeError("the button box has no Restore Defaults button")
        restore.clicked.connect(lambda _checked=False: self.reset_all())
        return buttons

    def standard_button(self, which: QDialogButtonBox.StandardButton) -> QPushButton:
        button = self.buttons.button(which)
        if button is None:
            raise RuntimeError(f"the button box has no {which.name} button")
        return button

    def reset(self, command: Command) -> None:
        self.edits[command].setKeySequence(QKeySequence(command.default))

    def reset_all(self) -> None:
        for command in self.edits:
            self.reset(command)

    def check_conflicts(self) -> None:
        """Say which commands share a key, and allow OK only when none do."""
        owners: dict[str, list[str]] = {}
        for command, edit in self.edits.items():
            text = edit.keySequence().toString(QKeySequence.SequenceFormat.NativeText)
            if text:
                owners.setdefault(text, []).append(command.label)
        clashes = [
            f"{key}: {', '.join(labels)}" for key, labels in owners.items() if len(labels) > 1
        ]
        self.conflict.setText("Used more than once — " + "; ".join(clashes) if clashes else "")
        self.standard_button(QDialogButtonBox.StandardButton.Ok).setEnabled(not clashes)

    def accept(self) -> None:
        self.controller.apply({command: edit.keySequence() for command, edit in self.edits.items()})
        super().accept()
