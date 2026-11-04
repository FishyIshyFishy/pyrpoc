"""``PathField``: one field reading as a path, a folder you click to choose and a name you
type inside it."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QWidget,
)

# Scoped by object name so the frame's border does not reach its parts. The parts are flat so
# the whole reads as one field, ``folder / name``.
_STYLESHEET = (
    "#pathField { background: palette(base); border: 1px solid palette(mid);"
    " border-radius: 3px; }"
    '#pathField[focused="true"] { border-color: palette(highlight); }'
    "#pathField QLineEdit { background: transparent; border: none; padding: 0 4px 0 0; }"
    "#pathField QToolButton { background: transparent; border: none; padding: 0 0 0 6px; }"
    "#pathField QToolButton:hover { text-decoration: underline; }"
    "#pathField QLabel { background: transparent; color: palette(mid); padding: 0 4px; }"
)
# The folder's text: chosen, missing (a prompt to choose one), or not in use.
_FOLDER_SET = "color: palette(text);"
_FOLDER_MISSING = "color: palette(highlight);"
_FOLDER_UNUSED = "color: palette(mid);"


class PathField(QFrame):
    """``folder / name``: the folder is a button that asks to choose another, the name a line
    edit."""

    folder_clicked = pyqtSignal()
    name_edited = pyqtSignal(str)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("pathField")
        self.setStyleSheet(_STYLESHEET)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(1, 1, 1, 1)
        row.setSpacing(0)

        self.folder_btn = QToolButton(self)
        self.folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.folder_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.folder_btn.clicked.connect(self.folder_clicked)
        row.addWidget(self.folder_btn)
        row.addWidget(QLabel("/", self))
        self.name_edit = QLineEdit(self)
        self.name_edit.setMinimumWidth(60)
        self.name_edit.textEdited.connect(self.name_edited)
        row.addWidget(self.name_edit, 1)
        # As tall as a plain line edit, so it lines up with the fields around it.
        self.setFixedHeight(QLineEdit().sizeHint().height())
        # The frame shows focus for its line edit, whose own border is gone.
        self.name_edit.installEventFilter(self)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        del a0
        if a1 is not None and a1.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            self.setProperty("focused", a1.type() == QEvent.Type.FocusIn)
            # A dynamic property only restyles once the style is reapplied.
            style = self.style()
            if style is not None:
                style.unpolish(self)
                style.polish(self)
        return False

    def set_folder(self, name: str | None, tooltip: str, *, in_use: bool) -> None:
        """Show the chosen folder's ``name``, or a prompt to choose one when there is none.
        A folder not ``in_use`` is dimmed."""
        self.folder_btn.setText(name if name is not None else "Choose folder")
        self.folder_btn.setToolTip(tooltip)
        if not in_use:
            color = _FOLDER_UNUSED
        else:
            color = _FOLDER_SET if name is not None else _FOLDER_MISSING
        self.folder_btn.setStyleSheet(color)

    def set_name(self, text: str) -> None:
        """Show ``text`` unless it is already shown, so typing keeps its cursor."""
        if self.name_edit.text() != text:
            self.name_edit.setText(text)
