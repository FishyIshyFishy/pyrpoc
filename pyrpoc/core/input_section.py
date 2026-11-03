"""A labelled list of a program's parameter fields: how a form groups them on screen."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .program_inputs import Field, Group, block_fields, block_name


@dataclass(frozen=True)
class Section:
    """A heading and the ``(dotted path, field)`` rows under it."""

    label: str
    entries: tuple[tuple[str, Field], ...]


def sections(blocks: Sequence[Group]) -> list[Section]:
    """One section per block, in the order the program declared them."""
    out: list[Section] = []
    for block in blocks:
        entries = tuple((f"{block_name(block)}.{name}", spec) for name, spec in block_fields(block))
        out.append(Section(type(block).label or block_name(block), entries))
    return out
