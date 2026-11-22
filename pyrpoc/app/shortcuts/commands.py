"""``Command``: something the window can do from a key, and ``ShortcutController``, which binds
each command to its key and remembers the keys the user chose."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QWidget

from pyrpoc.app.settings import app_settings

# Keys are saved under this group, one entry per command key.
SETTINGS_GROUP = "shortcuts"


@dataclass(frozen=True)
class Command:
    """A thing a shortcut can do: ``key`` names it in saved preferences, ``label`` in the
    Shortcuts dialog, and ``default`` is its key sequence before the user changes it, empty for
    none."""

    key: str
    label: str
    default: str


class ShortcutController:
    """Each registered command's action on ``window``, with the key the user saved for it or else
    its default."""

    def __init__(self, window: QWidget):
        self.window = window
        self.settings = app_settings()
        self.actions: dict[Command, QAction] = {}

    def register(self, command: Command, run: Callable[[], None]) -> None:
        action = QAction(command.label, self.window)
        # Application-wide, so a key works whichever dock, floating or not, has focus.
        action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        action.triggered.connect(lambda _checked=False: run())
        action.setShortcut(self.saved_sequence(command))
        self.window.addAction(action)
        self.actions[command] = action

    def commands(self) -> list[Command]:
        return list(self.actions)

    def sequence(self, command: Command) -> QKeySequence:
        return self.actions[command].shortcut()

    def saved_sequence(self, command: Command) -> QKeySequence:
        """The key the user saved for ``command``, or its default if they never changed it."""
        # Preferences are a boundary; a cleared shortcut is saved as "", unlike one never set.
        raw = self.settings.value(f"{SETTINGS_GROUP}/{command.key}", command.default)
        return QKeySequence(str(raw), QKeySequence.SequenceFormat.PortableText)

    def apply(self, sequences: dict[Command, QKeySequence]) -> None:
        """Bind and save the given keys."""
        for command, sequence in sequences.items():
            self.actions[command].setShortcut(sequence)
            text = sequence.toString(QKeySequence.SequenceFormat.PortableText)
            self.settings.setValue(f"{SETTINGS_GROUP}/{command.key}", text)
