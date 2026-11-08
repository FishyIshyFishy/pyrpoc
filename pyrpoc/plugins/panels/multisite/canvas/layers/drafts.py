"""Units drawn and on their way to the loop, dashed, with a tag saying so."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsScene

from ... import look
from ...state.drafts import Draft
from .badge import Badge
from .unit_outlines import path_of

DRAFT_Z = 12.0


class DraftOutlines:
    def __init__(self, scene: QGraphicsScene):
        self.scene = scene
        self.shown: dict[Draft, tuple[QGraphicsPathItem, Badge]] = {}

    def show(self, drafts: list[Draft]) -> None:
        for draft in list(self.shown):
            if draft not in drafts:
                for item in self.shown.pop(draft):
                    self.scene.removeItem(item)
        for draft in drafts:
            if draft not in self.shown:
                self.shown[draft] = self.add(draft)

    def add(self, draft: Draft) -> tuple[QGraphicsPathItem, Badge]:
        path = QGraphicsPathItem(path_of(draft.unit.outline))
        line = QPen(look.ACCENT, 1.8, Qt.PenStyle.DashLine)
        line.setCosmetic(True)
        path.setPen(line)
        path.setBrush(look.faded(look.ACCENT, 30))
        path.setZValue(DRAFT_Z)
        badge = Badge("joining…", look.ACCENT)
        badge.setZValue(DRAFT_Z)
        left = min(x for x, _ in draft.unit.outline)
        top = min(y for _, y in draft.unit.outline)
        badge.setPos(left, top)
        self.scene.addItem(path)
        self.scene.addItem(badge)
        return path, badge
