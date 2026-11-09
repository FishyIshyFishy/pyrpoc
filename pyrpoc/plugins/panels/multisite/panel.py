"""The Multisite Loop panel: a survey's dish with the units on it, and the queue and units of the
newest loop run over it, held up over the dish on glass cards with Tab. Right-click a unit to
steer the loop."""

from __future__ import annotations

import time

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QLabel, QMenu, QMessageBox, QPushButton, QVBoxLayout, QWidget

from pyrpoc.core.library import Library
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.plugins.program_io.outputs import Outline, SurveyOutput, UnitState
from pyrpoc.utils.widgets.data_panel import DataPanel, data_panel_registry
from pyrpoc.utils.widgets.empty_state import EmptyState

from .dish import DishCanvas
from .feed import LoopFeed
from .look import stroke_of
from .outlines import UnitLook
from .overlay import Overlay
from .state.pending import Change
from .state.standing import QueueRow, Snapshot, UnitRow, queue_rows, unit_rows, working_at
from .state.wording import doing_text
from .unit_menu import action, unit_menu

# Countdowns run between the loop's events, so the panel reads the loop on a clock.
TICK_MS = 250
# How long a note about a drawn shape stays in place of the status line, in seconds.
NOTICE_S = 6.0


@data_panel_registry.register("multisite")
class MultisitePanel(DataPanel[SurveyOutput]):
    display_name = "Multisite Loop"
    group = "Multisite"
    order = 40
    renders = SurveyOutput

    def __init__(self, library: Library, inputs: ProgramInputs):
        super().__init__(library, inputs)
        self.feed: LoopFeed | None = None
        self.dish: DishCanvas | None = None
        self.snapshot: Snapshot | None = None
        self.queue: tuple[QueueRow, ...] = ()
        self.units: tuple[UnitRow, ...] = ()
        self.chosen: str | None = None
        self.notice: tuple[str, float] | None = None
        self.build_buttons()
        self.build_body()
        toggle = QShortcut(QKeySequence(Qt.Key.Key_Tab), self)
        toggle.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        toggle.activated.connect(self.queue_button.toggle)
        # Esc closes the overlay only while it is up, so it still stops a drawing otherwise.
        self.close_key = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.close_key.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.close_key.setEnabled(False)
        self.close_key.activated.connect(lambda: self.queue_button.setChecked(False))
        self.clock = QTimer(self)
        self.clock.setInterval(TICK_MS)
        self.clock.timeout.connect(self.tick)
        self.clock.start()
        self.connect_source()

    def build_buttons(self) -> None:
        row = self.source.row
        self.draw_button = QPushButton("Draw unit", self)
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip("Drag on the dish round a region to make it a unit; Esc stops")
        self.draw_button.toggled.connect(self.set_drawing)
        self.fit_button = QPushButton("Whole dish", self)
        self.fit_button.clicked.connect(self.fit_dish)
        self.queue_button = QPushButton("Queue", self)
        self.queue_button.setCheckable(True)
        self.queue_button.setToolTip("Show or hide the queue and the units (Tab)")
        for button in (self.draw_button, self.fit_button, self.queue_button):
            row.addWidget(button)

    def build_body(self) -> None:
        root = QVBoxLayout(self.body)
        root.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("", self.body)
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        holder = QWidget(self.body)
        self.dish_holder = holder
        self.dish_box = QVBoxLayout(holder)
        self.dish_box.setContentsMargins(0, 0, 0, 0)
        self.placeholder = EmptyState(
            "No dish yet", "Choose a survey above. Its dish shows once it has finished.", holder
        )
        self.dish_box.addWidget(self.placeholder)
        root.addWidget(holder, 1)
        self.overlay = Overlay(holder)
        self.overlay.dismissed.connect(lambda: self.queue_button.setChecked(False))
        self.queue_button.toggled.connect(self.show_overlay)
        for card in self.overlay.cards:
            card.table.chose.connect(self.choose)
            card.table.locate.connect(self.locate)
            card.table.menu_asked.connect(lambda key, at: self.show_unit_menu(key, at, False))

    def draw(self, program_output: SurveyOutput) -> None:
        if self.feed is None or self.feed.survey is not program_output:
            self.feed = LoopFeed(program_output, self.library, self.inputs)
            self.drop_dish()
        if self.dish is None:
            self.make_dish(self.feed)
        self.tick()

    def clear(self) -> None:
        self.feed = None
        self.drop_dish()
        self.tick()

    def make_dish(self, feed: LoopFeed) -> None:
        backdrop = feed.backdrop()
        if backdrop is None:
            self.placeholder.show()
            return
        dish = DishCanvas(self.dish_holder, backdrop, feed.field_um())
        dish.chose.connect(self.choose)
        dish.menu_asked.connect(self.show_dish_menu)
        dish.drawn.connect(self.add_unit)
        dish.refused.connect(self.note)
        dish.drawing_changed.connect(self.draw_button.setChecked)
        self.placeholder.hide()
        self.dish_box.addWidget(dish, 1)
        self.overlay.raise_()
        self.dish = dish

    def drop_dish(self) -> None:
        if self.dish is not None:
            self.dish.deleteLater()
            self.dish = None
        self.placeholder.show()

    def tick(self) -> None:
        feed = self.feed
        if feed is None:
            self.status.setText("")
            return
        snapshot = feed.update(time.monotonic())
        self.snapshot = snapshot
        self.queue = queue_rows(snapshot)
        self.units = unit_rows(snapshot, self.queue)
        self.status.setText(self.status_text(snapshot))
        if self.overlay.up:
            self.show_cards(snapshot, feed)
        if self.dish is not None:
            self.show_dish(self.dish, snapshot, feed)

    def status_text(self, snapshot: Snapshot) -> str:
        if self.notice is not None and time.monotonic() - self.notice[1] < NOTICE_S:
            return self.notice[0]
        return doing_text(snapshot, self.queue)

    def show_dish(self, dish: DishCanvas, snapshot: Snapshot, feed: LoopFeed) -> None:
        looks = [
            UnitLook(row.key, row.outline, row.name, stroke_of(row.standing, dish.accent))
            for row in self.units
        ]
        units = () if snapshot.view is None else snapshot.view.units
        present = {unit.number for unit in units if unit.state is not UnitState.REMOVED}
        dish.show_units(looks, feed.frames, present, working_at(snapshot))

    def show_cards(self, snapshot: Snapshot, feed: LoopFeed) -> None:
        self.overlay.queue.show_loop(snapshot, self.queue)
        self.overlay.units.show_loop(snapshot, self.units, feed.pending)

    def show_overlay(self, shown: bool) -> None:
        self.close_key.setEnabled(shown)
        if not shown:
            self.overlay.vanish()
            if self.dish is not None:
                self.dish.setFocus()
            return
        if self.snapshot is not None and self.feed is not None:
            self.show_cards(self.snapshot, self.feed)
        for card in self.overlay.cards:
            card.table.choose(self.chosen)
        self.overlay.appear()

    def choose(self, key: str | None) -> None:
        self.chosen = key
        for card in self.overlay.cards:
            card.table.choose(key)
        if self.dish is not None:
            self.dish.select(key)

    def note(self, text: str) -> None:
        self.notice = (text, time.monotonic())
        self.tick()

    def set_drawing(self, on: bool) -> None:
        if self.dish is not None:
            self.dish.set_drawing(on)
        elif on:
            self.draw_button.setChecked(False)

    def fit_dish(self) -> None:
        if self.dish is not None:
            self.dish.fit_dish()

    def add_unit(self, outline: Outline) -> None:
        if self.feed is not None:
            self.feed.add(outline)
            self.tick()

    # What a right-click on a unit can ask, as ``unit_menu`` calls it.
    def run_next(self, number: int) -> None:
        self.steer(number, Change.RUN_NEXT)

    def pause(self, number: int, paused: bool) -> None:
        self.steer(number, Change.PAUSE if paused else Change.RESUME)

    def steer(self, number: int, change: Change) -> None:
        if self.feed is not None:
            self.feed.edit(number, change, time.monotonic())
            self.tick()

    def remove(self, number: int) -> None:
        if self.feed is None:
            return
        answer = QMessageBox.question(
            self,
            "Remove Unit",
            f"Remove unit {number} from the loop? Every one of its routines stops, and it "
            "cannot be added back: a unit drawn again is a new unit.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.feed.remove(number, time.monotonic())
            self.tick()

    def drop_drawn(self, key: str) -> None:
        if self.feed is not None:
            self.feed.drop(key)
            self.tick()

    def locate(self, key: str) -> None:
        if self.dish is not None:
            free = self.overlay.uncovered() if self.overlay.up else self.dish.view_rect()
            self.dish.locate(key, free)

    def show_unit_menu(self, key: str, at: QPoint, from_dish: bool) -> None:
        row = next((row for row in self.units if row.key == key), None)
        if row is None or self.feed is None or self.snapshot is None:
            return
        self.choose(key)
        pending = None if row.number is None else self.feed.pending.of(row.number)
        menu = unit_menu(self, row, self.snapshot.running, pending, self, from_dish)
        menu.exec(at)

    def show_dish_menu(self, key: str | None, at: QPoint) -> None:
        if key is not None:
            self.show_unit_menu(key, at, True)
            return
        menu = QMenu(self)
        action(menu, "Draw a unit", True, lambda: self.draw_button.setChecked(True))
        action(menu, "Show whole dish", True, self.fit_dish)
        menu.exec(at)
