"""``NoteDialog``: write the note the next run is saved with."""

from __future__ import annotations

from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout, QWidget


class NoteDialog(QDialog):
    """A text box for the next run's note; ``text`` holds it once accepted."""

    def __init__(self, note: str, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle("Note")
        self.resize(420, 240)
        root = QVBoxLayout(self)
        hint = QLabel("Saved with the next run. You can edit it later in the Data Library.", self)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        root.addWidget(hint)
        self.edit = QPlainTextEdit(note, self)
        self.edit.setTabChangesFocus(True)
        root.addWidget(self.edit, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        # Enter makes new lines, so Ctrl+Enter is the way to finish without the mouse.
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.accept)
        self.edit.setFocus()
        cursor = self.edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.edit.setTextCursor(cursor)

    @property
    def text(self) -> str:
        return self.edit.toPlainText()
