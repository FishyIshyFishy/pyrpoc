"""A mask, and the port and line it drives -- one row per binding.

Pulled out of the generic form-builder engine because it is not a generic
widget: it knows about ``Mask``/``MasksField`` (an acquisition parameter type)
and queries a ``DatasetLibrary`` for open masks, neither of which belongs in
``qt_widgets/``. Registers ``build_masks`` into that engine's ``BUILDERS`` on
import, which is why ``panels/acquisition/__init__.py`` imports this module
before building any form that might contain a ``MasksField``.

Replaces BaseOptoControl, BaseOptoControlWidget, the optocontrol registry and
manager panel, prepare_for_acquisition, get_context, MaskContext,
extract_mask_contexts, allowed_optocontrols and the optocontrols list in
AppState. Masks stop being globally toggleable objects and become run
parameters, so a run's saved metadata records exactly which masks were
applied on which lines -- which v3.0 did not, because the enabled flag lived
outside the modality.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.core.streams import Mask2D
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.programs.components import Mask, MasksField
from pyrpoc.qt_widgets.param_form import BUILDERS, FieldWidget


class MaskSourceCombo(QComboBox):
    """Picks a ``Mask2D`` entry out of the open data.

    Repopulated in ``showPopup`` rather than from a library subscription, and
    that is deliberate. The form is rebuilt on every modality change, and the
    library notifies its subscribers from inside ``Runner.start``; a stale
    callback into a deleted row would raise there, taking out the run that was
    starting. A list built when it is opened cannot go stale.
    """

    def __init__(self, table: "MaskTable", parent: QWidget | None = None):
        super().__init__(parent)
        self._table = table
        self.setToolTip("A mask drawn in the Mask Editor. Add one there to see it here.")

    def showPopup(self) -> None:
        chosen = self.currentData()
        label = self.currentText()
        self.blockSignals(True)
        self.clear()
        found = False
        for dataset in self._table.sources():
            self.addItem(dataset.label, dataset.id)
            if dataset.id == chosen:
                self.setCurrentIndex(self.count() - 1)
                found = True
        if isinstance(chosen, str) and chosen and not found:
            # The entry was closed in the data panel. Keep naming it rather than
            # silently rebinding the row to an unrelated mask.
            self.addItem(f"{label} (closed)", chosen)
            self.setCurrentIndex(self.count() - 1)
        self.blockSignals(False)
        super().showPopup()


class MaskTable(QWidget):
    """A mask, and the port and line it drives -- one row per binding.

    A row names a dataset rather than a file. The mask editor files what it draws
    in the library and this asks the library what it holds, so neither knows the
    other exists and drawing a mask no longer means saving a PNG and browsing
    back to it.
    """

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._library: DatasetLibrary | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Mask", "Port", "Line"])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(90)
        self.table.itemChanged.connect(lambda *_: self.changed.emit())
        root.addWidget(self.table)

        buttons = QHBoxLayout()
        self.add_btn = QPushButton("Add mask", self)
        self.remove_btn = QPushButton("Remove", self)
        self.add_btn.clicked.connect(self.on_add_clicked)
        self.remove_btn.clicked.connect(self.on_remove_clicked)
        buttons.addWidget(self.add_btn)
        buttons.addWidget(self.remove_btn)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.hint = QLabel("Draw a mask in the Mask Editor to bind one here.", self)
        self.hint.setStyleSheet("color: palette(mid);")
        root.addWidget(self.hint)

    # -- the open data ----------------------------------------------------- #

    def attach_library(self, library: DatasetLibrary) -> None:
        self._library = library

    def sources(self) -> list:
        """The mask entries currently open, newest first."""
        if self._library is None:
            return []
        return self._library.matching(Mask2D)

    # -- rows -------------------------------------------------------------- #

    def add_row(self, binding: Mask) -> None:
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)

        combo = MaskSourceCombo(self, self.table)
        if binding.source_id:
            combo.addItem(binding.source_label or binding.source_id, binding.source_id)
        else:
            for dataset in self.sources():
                combo.addItem(dataset.label, dataset.id)
        combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.table.setCellWidget(row, 0, combo)

        for column, value in ((1, binding.port), (2, binding.line)):
            spin = QSpinBox(self.table)
            spin.setRange(0, 1024)
            spin.setValue(int(value))
            spin.valueChanged.connect(lambda *_: self.changed.emit())
            self.table.setCellWidget(row, column, spin)
        self.table.blockSignals(False)
        self.refresh_hint()

    def used_lines(self) -> set[int]:
        lines: set[int] = set()
        for row in range(self.table.rowCount()):
            spin = self.table.cellWidget(row, 2)
            if isinstance(spin, QSpinBox):
                lines.add(int(spin.value()))
        return lines

    def next_free_line(self) -> int:
        """The lowest line no row is already driving.

        Defaulting every row to line 0 would leave two masks fighting over one
        output with nothing on screen saying so.
        """
        used = self.used_lines()
        line = 0
        while line in used:
            line += 1
        return line

    def used_sources(self) -> set[str]:
        taken: set[str] = set()
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            chosen = combo.currentData() if isinstance(combo, QComboBox) else None
            if isinstance(chosen, str) and chosen:
                taken.add(chosen)
        return taken

    def next_source(self):
        """The newest mask no row has taken yet, or the newest if all are taken.

        The same courtesy as ``next_free_line`` and needed for the same reason:
        a second row defaulting to the mask the first row already has looks like
        two bindings and behaves like one.
        """
        taken = self.used_sources()
        available = self.sources()
        for dataset in available:
            if dataset.id not in taken:
                return dataset
        return available[0] if available else None

    def on_add_clicked(self) -> None:
        newest = self.next_source()
        self.add_row(
            Mask(
                source_id=newest.id if newest is not None else "",
                source_label=newest.label if newest is not None else "",
                port=0,
                line=self.next_free_line(),
            )
        )
        self.changed.emit()

    def on_remove_clicked(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        self.table.removeRow(row)
        self.refresh_hint()
        self.changed.emit()

    def refresh_hint(self) -> None:
        self.hint.setVisible(self.table.rowCount() == 0)

    # -- value ------------------------------------------------------------- #

    def value(self) -> tuple[Mask, ...]:
        """The rows that still name open data, resolved to their pixels.

        A row whose entry has been closed is skipped, the way a row with no file
        used to be: the mask cannot be applied. The array is read here because a
        program has no library to read it from later.
        """
        out: list[Mask] = []
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 0)
            dataset_id = combo.currentData() if isinstance(combo, QComboBox) else None
            if not isinstance(dataset_id, str) or not dataset_id:
                continue
            dataset = self._library.by_id(dataset_id) if self._library is not None else None
            array = dataset.latest() if dataset is not None else None
            if array is None:
                continue
            port = self.table.cellWidget(row, 1)
            line = self.table.cellWidget(row, 2)
            out.append(
                Mask(
                    source_id=dataset_id,
                    source_label=dataset.label,
                    array=array,
                    port=port.value() if isinstance(port, QSpinBox) else 0,
                    line=line.value() if isinstance(line, QSpinBox) else 0,
                )
            )
        return tuple(out)

    def set_value(self, bindings) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.blockSignals(False)
        for binding in bindings or ():
            self.add_row(binding)
        self.refresh_hint()

    def summary(self) -> str:
        """How many masks will be applied, and how many rows will not be.

        A row whose entry was closed in the data panel stops applying, and
        ``value`` drops it. Counting only the rows would then show a mask that
        is not going to run, so the two numbers are reported separately.
        """
        rows = self.table.rowCount()
        if rows == 0:
            return "none"
        live = len(self.value())
        text = f"{live} mask" + ("" if live == 1 else "s")
        closed = rows - live
        return text + (f", {closed} closed" if closed else "")


def build_masks(spec: MasksField, parent) -> FieldWidget:
    del spec
    table = MaskTable(parent)
    return FieldWidget(
        table,
        get=table.value,
        set=table.set_value,
        connect=lambda cb: table.changed.connect(lambda *_: cb()),
        summary=table.summary,
        attach_library=table.attach_library,
    )


BUILDERS[MasksField] = build_masks
