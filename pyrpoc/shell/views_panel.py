"""The views panel: which renderers are open.

Replaces gui/main_widgets/display_mgr/. The dropdown is filtered by shape
contract rather than by a program's hardcoded ``allowed_displays`` list, which
was connection logic hiding inside a modality class.

In phase 6 the entries are still the v3.0 display widgets fed by
display_bridge.py; phase 8 swaps them for views that read a bound dataset.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.displays.display_registry import display_registry

from .app import Application
from .cards import RemovableCardWidget


class ViewsPanel(QWidget):
    def __init__(self, app: Application, parent: QWidget | None = None):
        super().__init__(parent)
        self.app = app

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top = QHBoxLayout()
        self.type_combo = QComboBox(self)
        for key in display_registry.list_keys():
            cls = display_registry.get_class(key)
            self.type_combo.addItem(getattr(cls, "display_name", key), key)
        self.add_btn = QPushButton("Add", self)
        top.addWidget(self.type_combo, 1)
        top.addWidget(self.add_btn)
        root.addLayout(top)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.content = QWidget(self.scroll)
        self.instances_layout = QVBoxLayout(self.content)
        self.instances_layout.setContentsMargins(0, 0, 0, 0)
        self.instances_layout.setSpacing(6)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)

        self.add_btn.clicked.connect(self.on_add_clicked)
        self.app.views_changed.connect(self.refresh)
        self.refresh()

    def on_add_clicked(self) -> None:
        key = self.type_combo.currentData()
        if not isinstance(key, str):
            return
        cls = display_registry.get_class(key)
        widget = cls()
        widget.configure({})
        self.app.add_view(widget)

    def refresh(self) -> None:
        while self.instances_layout.count():
            item = self.instances_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        for view in self.app.views:
            title = getattr(view, "user_label", None) or getattr(view, "display_name", "View")
            card = RemovableCardWidget(view, title, self.content)
            card.set_toggle_visible(False)
            card.remove_requested.connect(lambda obj: self.app.remove_view(obj))
            self.instances_layout.addWidget(card)
        self.instances_layout.addStretch(1)
