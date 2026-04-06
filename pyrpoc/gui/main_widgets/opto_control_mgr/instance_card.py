from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class InstanceCardWidget(QFrame):
    expand_requested = pyqtSignal(object)
    enable_toggled = pyqtSignal(object, bool)
    remove_requested = pyqtSignal(object)

    def __init__(self, state_obj: object, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_obj = state_obj
        self._expanded = False
        self._enable_guard = False

        self.setObjectName("optoControlCard")
        self.setProperty("enabledCard", False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(self._compose_stylesheet(enabled=False))

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(2)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(4)

        self.expand_btn = QToolButton(self)
        self.expand_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.expand_btn.setAutoRaise(True)
        self.expand_btn.setToolTip("Expand")
        self.expand_btn.clicked.connect(self._on_expand_clicked)
        header_row.addWidget(self.expand_btn)

        self.enable_checkbox = QCheckBox("", self)
        self.enable_checkbox.setToolTip("Enable")
        self.enable_checkbox.toggled.connect(self._on_enable_toggled)
        header_row.addWidget(self.enable_checkbox)

        self.title_label = QLabel(title, self)
        header_row.addWidget(self.title_label, 1)

        self.remove_btn = QToolButton(self)
        self.remove_btn.setAutoRaise(True)
        self.remove_btn.setText("X")
        self.remove_btn.setStyleSheet("QToolButton { color: palette(text); font-weight: 700; }")
        self.remove_btn.setToolTip("Remove")
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.state_obj))
        header_row.addWidget(self.remove_btn)

        root.addLayout(header_row)

        self._description_label = QLabel("", self)
        self._description_label.setStyleSheet(
            "color: palette(mid); font-size: 9pt; padding-left: 22px;"
        )
        self._description_label.setWordWrap(True)
        self._description_label.setVisible(False)
        root.addWidget(self._description_label)

        self.body_container = QWidget(self)
        self.body_layout = QVBoxLayout(self.body_container)
        self.body_layout.setContentsMargins(0, 2, 0, 0)
        self.body_layout.setSpacing(4)
        self.body_container.setVisible(False)
        root.addWidget(self.body_container)

    def _on_expand_clicked(self) -> None:
        self.expand_requested.emit(self.state_obj)

    def _on_enable_toggled(self, checked: bool) -> None:
        self._apply_enabled_visual(bool(checked))
        if self._enable_guard:
            return
        self.enable_toggled.emit(self.state_obj, checked)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self.body_container.setVisible(expanded)
        self.expand_btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.expand_btn.setToolTip("Collapse" if expanded else "Expand")
        if self._description_label.text():
            self._description_label.setVisible(not expanded)

    def is_expanded(self) -> bool:
        '''Report current expanded state for expand toggle handlers in manager logic.'''
        return self._expanded

    def set_description(self, text: str) -> None:
        self._description_label.setText(text)
        self._description_label.setStyleSheet(
        "color: white;"
        )
        self._description_label.setVisible(bool(text) and not self._expanded)

    def set_marker_text(self, text: str) -> None:
        del text

    def set_local_status(self, text: str) -> None:
        self.setToolTip(text)

    def set_enable_checked(self, checked: bool, guarded: bool = True) -> None:
        if guarded:
            self._enable_guard = True
        self.enable_checkbox.setChecked(checked)
        self._apply_enabled_visual(bool(checked))
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

    def _apply_enabled_visual(self, enabled: bool) -> None:
        self.setProperty("enabledCard", bool(enabled))
        self.setStyleSheet(self._compose_stylesheet(enabled=bool(enabled)))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _compose_stylesheet(self, enabled: bool) -> str:
        highlight = QColor(self.palette().color(self.palette().ColorRole.Highlight))
        muted_bg = QColor(highlight)
        muted_bg.setAlpha(46)
        muted_border = QColor(highlight)
        muted_border.setAlpha(150)
        enabled_block = ""
        if enabled:
            enabled_block = (
                "#optoControlCard {"
                f"background: rgba({muted_bg.red()}, {muted_bg.green()}, {muted_bg.blue()}, {muted_bg.alpha()});"
                f"border: 1px solid rgba({muted_border.red()}, {muted_border.green()}, {muted_border.blue()}, {muted_border.alpha()});"
                "}"
            )
        return (
            "#optoControlCard {"
            "background: palette(base);"
            "border: 1px solid palette(midlight);"
            "border-radius: 6px;"
            "}"
            "#optoControlCard > QWidget, #optoControlCard > QWidget QWidget {"
            "background: transparent;"
            "}"
            "#optoControlCard QLabel, #optoControlCard QCheckBox, #optoControlCard QToolButton {"
            "background: transparent;"
            "}"
            "#optoControlCard QCheckBox::indicator:checked {"
            "background: palette(highlight);"
            "border: 1px solid palette(highlight);"
            "color: palette(highlighted-text);"
            "}"
            "#optoControlCard QCheckBox::indicator:unchecked {"
            "background: transparent;"
            "border: 1px solid palette(mid);"
            "}"
            f"{enabled_block}"
        )
