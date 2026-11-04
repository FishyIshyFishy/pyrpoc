"""``EarlierCard``: the Acquisition panel's card choosing which recording each earlier run a
program reads comes from."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from pyrpoc.core.program import EarlierRun
from pyrpoc.engine.data_library.earlier import EarlierRecording
from pyrpoc.engine.data_library.grouping import program_display_name
from pyrpoc.engine.data_library.panel import taken
from pyrpoc.utils.widgets.cards import BaseCardWidget

from .runs import Runs


def option_text(recording: EarlierRecording) -> str:
    return f"{recording.name} · {taken(recording.started_at)}"


class EarlierCard(BaseCardWidget):
    """One dropdown per earlier run: the latest recording by default, or one picked."""

    def __init__(self, runs: Runs, reads: list[EarlierRun], parent: QWidget):
        super().__init__("Earlier runs", parent)
        self.runs = runs
        self.combos: dict[str, QComboBox] = {}
        body = QWidget(self)
        column = QVBoxLayout(body)
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(2)
        for each in reads:
            combo = QComboBox(body)
            combo.setToolTip("Which recording in the save folder this run reads.")
            combo.activated.connect(lambda _index, key=each.program_key: self.on_picked(key))
            self.combos[each.program_key] = combo
            # Above the dropdown, as input cards show their labels, so a narrow panel never cuts it.
            column.addWidget(QLabel(program_display_name(each.program_key), body))
            column.addWidget(combo)
        self.set_body_widget(body)
        runs.earlier_changed.connect(self.reload)
        self.reload()

    def reload(self) -> None:
        """Show what the save folder holds now, and what is picked."""
        for key, combo in self.combos.items():
            self.fill(combo, key)
        self.set_summary([(program_display_name(key), self.summary(key)) for key in self.combos])

    def fill(self, combo: QComboBox, program_key: str) -> None:
        options = self.runs.earlier.options(program_key)
        combo.blockSignals(True)
        combo.clear()
        if not options:
            combo.addItem("None saved in this folder", None)
        else:
            # The first entry follows whatever is newest, so a new recording is read at once.
            combo.addItem(f"Latest: {option_text(options[0])}", None)
            for recording in options:
                combo.addItem(option_text(recording), str(recording.meta_path))
            picked = self.runs.earlier.picked_path(program_key)
            if picked is not None:
                combo.setCurrentIndex(combo.findData(str(picked)))
        combo.setEnabled(bool(options))
        combo.blockSignals(False)

    def summary(self, program_key: str) -> str:
        options = self.runs.earlier.options(program_key)
        if not options:
            return "none saved"
        picked = self.runs.earlier.picked_path(program_key)
        if picked is None:
            return f"latest ({options[0].name})"
        return next(each.name for each in options if each.meta_path == picked)

    def on_picked(self, program_key: str) -> None:
        data = self.combos[program_key].currentData()
        self.runs.pick_earlier(program_key, None if data is None else Path(data))

    def detach(self) -> None:
        """Stop following the runs, before this card is thrown away."""
        self.runs.earlier_changed.disconnect(self.reload)
