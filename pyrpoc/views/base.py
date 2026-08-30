"""What a view is: something that renders datasets and emits interaction events.

A view holds no arrays. ``self._dataset`` plus ``dataset.latest()`` replaces
``self._data_chw``, which in v3.0 *was* the data -- closing a display destroyed
it, and two displays over one run held two drifting copies.

Every view gets a source picker, because with data outliving its renderer there
is a real question of which run is being shown. "Latest" follows the newest
matching dataset, which reproduces v3.0's implicit behaviour of pushing the
current run at whatever was open.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from pyrpoc.core.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.data.dataset import Dataset
    from pyrpoc.data.library import DatasetLibrary


def make_instance_id(prefix: str) -> str:
    token = (prefix or "view").strip().lower()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in token)
    return f"{safe or 'view'}-{uuid4().hex[:12]}"


class View(QWidget):
    """Renders one or more streams from datasets. Never references a program."""

    display_name: str = "View"
    registry_key: str = "view"

    #: Shape contracts this view can render. A dataset whose spec is not here
    #: cannot be bound to it.
    renders: list[type[Stream]] = []

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.instance_id = make_instance_id(self.registry_key)
        self.user_label: str | None = None
        self.docked_visible: bool = True
        self.last_error: str | None = None

        self._dataset: "Dataset | None" = None
        self._library: "DatasetLibrary | None" = None
        self._follow_latest = True

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(4, 4, 4, 4)
        self.root.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QLabel("Source:", self))
        self.source_combo = QComboBox(self)
        self.source_combo.currentIndexChanged.connect(self.on_source_chosen)
        header.addWidget(self.source_combo, 1)
        self.root.addLayout(header)

        self.body = QWidget(self)
        self.root.addWidget(self.body, 1)

    @property
    def type_key(self) -> str:
        return self.registry_key

    @property
    def title(self) -> str:
        return self.user_label or self.display_name

    # -- binding ------------------------------------------------------------ #

    def attach_library(self, library: "DatasetLibrary") -> None:
        self._library = library
        library.subscribe(self.refresh_sources)
        self.refresh_sources()

    def candidates(self) -> list["Dataset"]:
        if self._library is None:
            return []
        return self._library.matching(*self.renders)

    def refresh_sources(self) -> None:
        """Rebuild the picker, keeping the current choice where possible."""
        current = self.source_combo.currentData()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("Latest", None)
        for dataset in self.candidates():
            self.source_combo.addItem(dataset.label, dataset.id)
        if not self._follow_latest and isinstance(current, str):
            index = self.source_combo.findData(current)
            self.source_combo.setCurrentIndex(max(index, 0))
        self.source_combo.blockSignals(False)
        self.apply_source()

    def on_source_chosen(self, _index: int) -> None:
        self._follow_latest = self.source_combo.currentData() is None
        self.apply_source()

    def apply_source(self) -> None:
        chosen = self.source_combo.currentData()
        if chosen is None:
            candidates = self.candidates()
            self.bind(candidates[0] if candidates else None)
        elif self._library is not None:
            self.bind(self._library.by_id(chosen))

    def bind(self, dataset: "Dataset | None") -> None:
        if dataset is not None and dataset.spec not in self.renders:
            raise TypeError(
                f"{type(self).__name__} renders {[s.name for s in self.renders]}, "
                f"not {dataset.spec.name}"
            )
        self._dataset = dataset
        self.refresh()

    def dataset(self) -> "Dataset | None":
        return self._dataset

    def renders_dataset(self, dataset: "Dataset") -> bool:
        return dataset.spec in self.renders

    # -- drawing ------------------------------------------------------------- #

    def refresh(self) -> None:
        """Re-read the bound dataset and redraw. Subclasses implement this."""
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def configure(self, params: dict[str, Any]) -> None:
        del params

    # -- persistence ---------------------------------------------------------- #

    def export_persistence_state(self) -> dict[str, Any]:
        return {}

    def import_persistence_state(self, state: dict[str, Any]) -> None:
        del state
