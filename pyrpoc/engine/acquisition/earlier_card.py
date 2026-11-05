"""``EarlierCard``: the Acquisition panel's card choosing which earlier output each name a
program reads comes from."""

from __future__ import annotations

from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.engine.data_library.earlier import EarlierOutput
from pyrpoc.engine.data_library.format import META_SUFFIX
from pyrpoc.engine.data_library.panel import taken
from pyrpoc.utils.widgets.cards import BaseCardWidget

from .runs import Runs


def option_text(each: EarlierOutput) -> str:
    return f"{each.name} · {each.output} · {taken(each.started_at)}"


class EarlierCard(BaseCardWidget):
    """One dropdown per name read, listing the outputs of its type newest first, with the newest
    chosen until one is picked."""

    def __init__(self, runs: Runs, reads: dict[str, type[ProgramOutput]], parent: QWidget):
        super().__init__("Earlier recordings", parent)
        self.runs = runs
        self.reads = reads
        self.combos: dict[str, QComboBox] = {}
        self.models: dict[str, QStandardItemModel] = {}
        # What each row of each dropdown reads. Rows past these cannot be chosen.
        self.choices: dict[str, list[EarlierOutput]] = {}
        body = QWidget(self)
        column = QVBoxLayout(body)
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(2)
        for name in reads:
            combo = QComboBox(body)
            self.models[name] = QStandardItemModel(combo)
            combo.setModel(self.models[name])
            combo.setToolTip("Which recording in the save folder this run reads.")
            combo.activated.connect(lambda index, n=name: self.on_picked(n, index))
            self.combos[name] = combo
            # Above the dropdown, as input cards show their labels, so a narrow panel never cuts it.
            column.addWidget(QLabel(name, body))
            column.addWidget(combo)
        self.set_body_widget(body)
        runs.earlier_changed.connect(self.reload)
        self.reload()

    def reload(self) -> None:
        """Show what the save folder holds now, and what is picked."""
        for name, combo in self.combos.items():
            self.fill(combo, name)
        self.set_summary([(name, self.summary(name)) for name in self.combos])

    def fill(self, combo: QComboBox, name: str) -> None:
        earlier = self.runs.earlier
        options = earlier.options(self.reads[name])
        combo.blockSignals(True)
        combo.clear()
        if not options:
            combo.addItem("None saved in this folder")
        for each in options:
            combo.addItem(option_text(each))
        for each in earlier.unreadable:
            self.add_unreadable(name, each.meta_path.name.removesuffix(META_SUFFIX), each.reason)
        chosen = earlier.chosen(name, self.reads[name])
        if chosen is not None:
            combo.setCurrentIndex(options.index(chosen))
        combo.setEnabled(bool(options))
        combo.blockSignals(False)
        self.choices[name] = options

    def add_unreadable(self, name: str, stem: str, reason: str) -> None:
        """A row naming a recording that cannot be read, which cannot be chosen."""
        item = QStandardItem(f"{stem} (unreadable)")
        item.setToolTip(reason)
        item.setEnabled(False)
        self.models[name].appendRow(item)

    def summary(self, name: str) -> str:
        chosen = self.runs.earlier.chosen(name, self.reads[name])
        return "none saved" if chosen is None else chosen.name

    def on_picked(self, name: str, index: int) -> None:
        self.runs.pick_earlier(name, self.choices[name][index])

    def detach(self) -> None:
        """Stop following the runs, before this card is thrown away."""
        self.runs.earlier_changed.disconnect(self.reload)
