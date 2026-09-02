"""The views panel: which renderers are open.

Replaces gui/main_widgets/display_mgr/. The dropdown is filtered by shape
contract rather than by a program's hardcoded ``allowed_displays`` list, which
was connection logic hiding inside a modality class.

Each view carries its own source picker, so closing one no longer destroys its
data and two views can show the same run.
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

from pyrpoc.views.registry import view_registry

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
        for key in view_registry.keys():
            self.type_combo.addItem(view_registry.get(key).display_name, key)
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
        view = view_registry.get(key)()
        view.attach_library(self.app.library)
        self.app.add_view(view)

    def refresh(self) -> None:
        while self.instances_layout.count():
            item = self.instances_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        for view in self.app.views:
            title = view.title
            card = RemovableCardWidget(view, title, self.content)
            card.set_toggle_visible(False)
            card.remove_requested.connect(lambda obj: self.app.remove_view(obj))
            self.instances_layout.addWidget(card)
        self.instances_layout.addStretch(1)
