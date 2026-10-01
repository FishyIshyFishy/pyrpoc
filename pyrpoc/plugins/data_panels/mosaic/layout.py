"""Reading a mosaic's layout from an output's metadata.

The program that moved the stage wrote it under ``"mosaic"``, in the shape
``docs/data-format.md`` documents; that document is the only thing the program
and this panel share. Only what placing tiles needs is read: each tile's frame
index and grid cell. Stage positions are kept on disk for people, not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

METADATA_KEY = "mosaic"


@dataclass(frozen=True)
class GridTile:
    """Frame ``index`` of the output sits in grid cell (``row``, ``col``)."""

    index: int
    row: int
    col: int


@dataclass(frozen=True)
class MosaicLayout:
    tiles: tuple[GridTile, ...]


def read_layout(metadata: dict[str, Any]) -> MosaicLayout | None:
    """The layout an output's metadata carries, or None when it is not a
    mosaic. Metadata may come from a file, so a malformed layout raises."""
    raw = metadata.get(METADATA_KEY)
    if raw is None:
        return None
    try:
        tiles = tuple(
            GridTile(index=int(tile["index"]), row=int(tile["row"]), col=int(tile["col"]))
            for tile in raw["tiles"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed mosaic layout: {exc!r}") from exc
    return MosaicLayout(tiles)
