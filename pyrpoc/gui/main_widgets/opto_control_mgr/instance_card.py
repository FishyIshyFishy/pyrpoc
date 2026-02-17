from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class InstanceCardWidget(QFrame):
    expand_requested = pyqtSignal(str)
    enable_toggled = pyqtSignal(str, bool)
    remove_requested = pyqtSignal(str)

    def __init__(self, instance_id: str, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.instance_id = instance_id
        self._expanded = False
        self._enable_guard = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        self.expand_btn = QToolButton(self)
        self.expand_btn.setText("Expand")
        self.expand_btn.clicked.connect(self._on_expand_clicked)
        header_row.addWidget(self.expand_btn)

        self.title_label = QLabel(title, self)
        header_row.addWidget(self.title_label, 1)

        self.marker_label = QLabel("", self)
        header_row.addWidget(self.marker_label)

        self.enable_checkbox = QCheckBox("Enable", self)
        self.enable_checkbox.toggled.connect(self._on_enable_toggled)
        header_row.addWidget(self.enable_checkbox)

        self.remove_btn = QPushButton("Remove", self)
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.instance_id))
        header_row.addWidget(self.remove_btn)

        root.addLayout(header_row)

        self.local_status_label = QLabel("Status: idle", self)
        root.addWidget(self.local_status_label)

        self.body_container = QWidget(self)
        self.body_layout = QVBoxLayout(self.body_container)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(6)
        self.body_container.setVisible(False)
        root.addWidget(self.body_container)

    def _on_expand_clicked(self) -> None:
        self.expand_requested.emit(self.instance_id)

    def _on_enable_toggled(self, checked: bool) -> None:
        if self._enable_guard:
            return
        self.enable_toggled.emit(self.instance_id, checked)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.body_container.setVisible(expanded)
        self.expand_btn.setText("Collapse" if expanded else "Expand")

    def set_marker_text(self, text: str) -> None:
        self.marker_label.setText(text)

    def set_local_status(self, text: str) -> None:
        self.local_status_label.setText(text)

    def set_enable_checked(self, checked: bool, guarded: bool = True) -> None:
        if guarded:
            self._enable_guard = True
        self.enable_checkbox.setChecked(checked)
        if guarded:
            self._enable_guard = False

    def set_enable_visible(self, visible: bool) -> None:
        self.enable_checkbox.setVisible(visible)

    def set_body_widget(self, body: QWidget | None) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
                child.deleteLater()
        if body is not None:
            self.body_layout.addWidget(body)

