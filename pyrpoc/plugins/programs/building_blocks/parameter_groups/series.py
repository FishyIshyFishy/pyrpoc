"""Series: an open ``Image2D`` time series an analysis reads, by reference to
its library entry. Its frames reach the run with its parameters, so an
analysis program needs no library and no devices."""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from typing import Any, ClassVar

import numpy as np

from pyrpoc.structs.data_library.data import Image2D
from pyrpoc.structs.data_library.dataset import Dataset
from pyrpoc.structs.data_library.library import Library
from pyrpoc.structs.plugins.params import (
    Editor,
    Field,
    FieldContext,
    Group,
    ParameterError,
    block,
    spec_field,
)

from .library_ref import LibraryRef, RefCombo, open_entry

NO_SERIES = "Choose a series"
SERIES_TOOLTIP = "An Image2D recording in the Data Library, one frame per time point."


def series_in(library: Library) -> list[Dataset]:
    """The open ``Image2D`` entries with more than one frame. One frame is no
    series, and leaving them out keeps an analysis's own image outputs from
    crowding the list."""
    return [dataset for dataset in library.matching(Image2D) if len(dataset) > 1]


@dataclass(frozen=True)
class SeriesRef(LibraryRef):
    """``frames`` and ``channels`` are filled only in a run's copy, by
    ``resolve_series``. ``compare=False`` because comparing arrays in a frozen
    dataclass's ``__eq__`` raises."""

    # (T, C, H, W): every frame so far, oldest first.
    frames: np.ndarray | None = dc_field(default=None, compare=False)
    channels: tuple[str, ...] = dc_field(default=(), compare=False)

    def stack(self) -> np.ndarray:
        # Narrows for the type checker: a run's series is resolved at start.
        if self.frames is None:
            raise ValueError(f"series '{self.describe()}' was not resolved")
        return self.frames


def resolve_series(series: SeriesRef, library: Library) -> SeriesRef:
    """``series`` with its frames so far, so a series still being acquired can
    be analysed as it stands."""
    dataset = open_entry(series, library)
    frames = dataset.frames()
    if not frames:
        raise ParameterError(f"series '{series.describe()}' has no frames yet")
    stack = np.stack(frames)
    channels = tuple(dataset.resolved_channel_labels(stack.shape[1]))
    return replace(series, frames=stack, channels=channels)


@dataclass(frozen=True)
class SeriesField(Field):
    """One ``Image2D`` entry. Unset refuses the run: there is nothing to analyse."""

    def coerce(self, value: Any) -> SeriesRef | None:
        if value is None:
            return None
        return SeriesRef.from_value(value)

    def encode(self, value: SeriesRef | None) -> Any:
        return None if value is None else value.to_dict()

    def resolve(self, value: SeriesRef | None, library: Library) -> SeriesRef:
        if value is None:
            raise ParameterError(f"{self.label}: choose a series to analyse")
        return resolve_series(value, library)

    def editor(self, parent: Any, context: FieldContext) -> Editor:
        # Narrows for the type checker: every program form offers the library.
        if context.library is None:
            raise ValueError("a series field needs the open data")
        library = context.library
        combo = RefCombo(
            library, lambda: series_in(library), SeriesRef, NO_SERIES, SERIES_TOOLTIP, parent
        )
        return Editor(
            combo,
            get=combo.value,
            set=combo.set_value,
            connect=lambda cb: combo.currentIndexChanged.connect(lambda *_: cb()),
            summary=combo.currentText,
            spec=self,
        )


@block
@dataclass
class SeriesGroup(Group):
    """Shared by every analysis that reads one series, so choosing it once
    carries over between them."""

    label: ClassVar[str] = "Series"

    series: SeriesRef | None = spec_field(
        None, SeriesField("Series", "The recording to analyse. " + SERIES_TOOLTIP)
    )
