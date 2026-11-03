"""A labelled list of a program's parameter fields: how a form groups them on screen."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .program_inputs import Field, Group, block_fields


@dataclass(frozen=True)
class Section:
    """A heading and the ``(dotted path, field)`` rows under it."""

    label: str
    entries: tuple[tuple[str, Field], ...]


def sections(classes: Sequence[type[Group]]) -> list[Section]:
    """One section per block class, in the order given."""
    out: list[Section] = []
    for cls in classes:
        entries = tuple((f"{cls.__name__}.{name}", spec) for name, spec in block_fields(cls))
        out.append(Section(cls.label or cls.__name__, entries))
    return out
