"""``MultisiteView``: the dish with the overlay over it, fed by one ``LoopFeed``. The panel will
wrap this one widget."""

from __future__ import annotations

import time

from PyQt6.QtCore import QEvent, QObject, QTimer
from PyQt6.QtGui import QKeyEvent, QResizeEvent
from PyQt6.QtWidgets import QApplication, QWidget

from pyrpoc.plugins.program_io.outputs import Unit

from . import keys
from .canvas.dish_canvas import DishCanvas
from .canvas.indicators import Indicator
from .feed import LoopFeed
from .overlay.overlay import Overlay
from .overlay.queue_card import QueueCard
from .state.peek import Peek

# Countdowns and the reticle move between events, so the view redraws on a clock, not on news.
FRAME_MS = 33


class SummonKey(QObject):
    """Takes the overlay's key from anything inside ``view`` before Qt moves focus with it."""

    def __init__(self, view: MultisiteView):
        super().__init__(view)
        self.view = view

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:
        if event is None or event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False
        # Every key event in the app passes here; only the overlay's key, inside this view, stops.
        if not isinstance(event, QKeyEvent) or event.key() != keys.OVERLAY:
            return False
        inside = isinstance(watched, QWidget) and (
            watched is self.view or self.view.isAncestorOf(watched)
        )
        if not inside:
            return False
        if not event.isAutoRepeat():
            self.view.summon(event.type() == QEvent.Type.KeyPress)
        return True


class MultisiteView(QWidget):
    def __init__(self, parent: QWidget | None, feed: LoopFeed):
        super().__init__(parent)
        self.feed = feed
        self.indicators = Indicator.ALL
        self.peek = Peek()
        self.canvas = DishCanvas(self, feed.field_um())
        self.overlay = Overlay(self)
        self.queue = QueueCard(self.overlay, feed)
        self.overlay.add(self.queue)
        self.connect_parts()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(SummonKey(self))
        self.clock = QTimer(self)
        self.clock.setInterval(FRAME_MS)
        self.clock.timeout.connect(self.tick)
        self.clock.start()

    def connect_parts(self) -> None:
        self.canvas.picked.connect(self.pick)
        self.canvas.drawn.connect(self.add_unit)
        self.canvas.remove_asked.connect(self.ask_remove)
        self.queue.chosen_unit.connect(self.canvas.select)
        self.queue.locate.connect(self.locate)
        self.overlay.dismissed.connect(self.close_overlay)

    def set_indicators(self, indicators: Indicator) -> None:
        self.indicators = indicators

    def tick(self) -> None:
        now_s = time.monotonic()
        snapshot = self.feed.snapshot()
        self.canvas.show_loop(
            snapshot, self.feed.backdrop(), self.feed.frames(), self.indicators, now_s
        )
        self.canvas.tick(now_s)
        if self.overlay.up:
            self.overlay.show_snapshot(snapshot)

    def summon(self, pressed: bool) -> None:
        """The overlay's key went down or up."""
        now_s = time.monotonic()
        if pressed:
            self.peek.press(now_s)
        else:
            self.peek.release(now_s)
        self.match_peek()

    def match_peek(self) -> None:
        if self.peek.shown and not self.overlay.up:
            self.overlay.show_snapshot(self.feed.snapshot())
            self.queue.choose_unit(self.canvas.selected)
            self.overlay.appear()
        elif not self.peek.shown and self.overlay.up:
            self.overlay.vanish()
            self.canvas.setFocus()

    def close_overlay(self) -> None:
        self.peek.close()
        self.match_peek()

    def pick(self, unit: int | None) -> None:
        self.canvas.select(unit)
        self.queue.choose_unit(unit)

    def locate(self) -> None:
        covered = self.overlay.covered_left() if self.overlay.up else 0
        self.canvas.locate(self.feed.snapshot(), covered)

    def add_unit(self, unit: Unit) -> None:
        self.feed.add_unit(unit)

    def ask_remove(self, unit: int) -> None:
        """Removing from the dish asks in the queue card, the one place a removal is asked."""
        self.peek.pin()
        self.match_peek()
        self.queue.ask_remove(unit)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.canvas.setGeometry(self.rect())
        self.overlay.setGeometry(self.rect())
