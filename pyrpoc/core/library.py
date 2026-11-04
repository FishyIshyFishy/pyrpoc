"""The ``Library`` protocol: access to the open program outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pyrpoc.core.program_output import Data

from .program_output import ProgramOutput

D = TypeVar("D", bound=Data)


class Library(Protocol):
    """The open program outputs: add one, look one up, list them, name the group each is
    listed under, and subscribe to changes."""

    def add(self, program_output: ProgramOutput[D]) -> ProgramOutput[D]: ...

    def by_id(self, output_id: str, kind: type[D]) -> ProgramOutput[D] | None:
        """The open program output with this id, if it holds ``kind``."""
        ...

    def matching(self, kind: type[D]) -> list[ProgramOutput[D]]:
        """Program outputs of this kind, newest first."""
        ...

    def group_label(self, program_output: ProgramOutput[Any]) -> str:
        """The name of the group ``program_output`` is listed under."""
        ...

    def subscribe(self, callback: Callable[[], None]) -> None: ...

    def unsubscribe(self, callback: Callable[[], None]) -> None: ...
