"""A parameter whose value is an open library entry, named by reference.

The stored value keeps only the entry's id and label, so the shared block and
the session never hold data; a field's ``resolve`` fills the data into a run's
copy, since a program has no library. Masks, and the series an analysis reads,
are both named this way, and pick their entry with the combos here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from PyQt6.QtWidgets import QComboBox, QWidget

from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.library import Library
from pyrpoc.structs.plugins.params import ParameterError

R = TypeVar("R", bound="LibraryRef")


@dataclass(frozen=True)
class LibraryRef:
    """One library entry. ``source_label`` sits beside the id because an id
    means nothing in metadata read months later."""

    source_id: str = ""
    source_label: str = ""

    def describe(self) -> str:
        """Which library entry this is, for a row that has lost it."""
        return self.source_label or self.source_id or "nothing"

    def to_dict(self) -> dict[str, Any]:
        """Provenance only: the entry's data is data, and this is a parameter."""
        return {"source_id": self.source_id, "source_label": self.source_label}

    @classmethod
    def from_value(cls: type[R], raw: Any) -> R:
        """A reference from the form (already one) or from JSON (a dict)."""
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise ParameterError("a library reference must be an object with a source_id")
        return cls(
            source_id=str(raw.get("source_id", "")), source_label=str(raw.get("source_label", ""))
        )


def open_entry(ref: LibraryRef, library: Library) -> Dataset:
    """The entry ``ref`` names. One that is not open refuses the run."""
    dataset = library.by_id(ref.source_id)
    if dataset is None:
        raise ParameterError(f"'{ref.describe()}' is not open")
    return dataset


CLOSED_SUFFIX = " (closed)"


class SourceCombo(QComboBox):
    """Picks one of ``sources``, after a ``blank`` entry with no id when
    nothing is itself a choice.

    Filled when opened rather than from a library subscription: the form is
    rebuilt on every program change, and a stale callback into a deleted row
    would raise inside the library's notify, which runs at run start.
    """

    def __init__(
        self,
        sources: Callable[[], list[Dataset]],
        blank: str | None,
        tooltip: str,
        parent: QWidget,
    ):
        super().__init__(parent)
        self._sources = sources
        self._blank = blank
        self.setToolTip(tooltip)

    def showPopup(self) -> None:
        chosen = self.currentData()
        label = self.currentText()
        self.blockSignals(True)
        self.clear()
        if self._blank is not None:
            self.addItem(self._blank, "")
        found = False
        for dataset in self._sources():
            self.addItem(dataset.label, dataset.id)
            if dataset.id == chosen:
                self.setCurrentIndex(self.count() - 1)
                found = True
        if chosen and not found:
            # Closed in the data panel: keep naming it rather than silently
            # rebinding the row to an unrelated entry.
            self.addItem(f"{label}{CLOSED_SUFFIX}", chosen)
            self.setCurrentIndex(self.count() - 1)
        self.blockSignals(False)
        super().showPopup()


class RefCombo(SourceCombo, Generic[R]):
    """One of ``sources`` as a ``make`` reference, or None for ``blank``."""

    def __init__(
        self,
        library: Library,
        sources: Callable[[], list[Dataset]],
        make: type[R],
        blank: str,
        tooltip: str,
        parent: QWidget,
    ):
        super().__init__(sources, blank, tooltip, parent)
        self.library = library
        self.make = make
        self.blank = blank

    def value(self) -> R | None:
        source_id = self.currentData()
        if not source_id:
            return None
        dataset = self.library.by_id(source_id)
        label = dataset.label if dataset is not None else self.currentText()
        return self.make(source_id=source_id, source_label=label.removesuffix(CLOSED_SUFFIX))

    def set_value(self, ref: R | None) -> None:
        self.clear()
        self.addItem(self.blank, "")
        if ref is not None:
            self.addItem(ref.describe(), ref.source_id)
            self.setCurrentIndex(1)
