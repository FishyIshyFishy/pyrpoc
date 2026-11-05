"""The ``Library`` protocol: access to the open program outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from .program_output import ProgramOutput

OutputT = TypeVar("OutputT", bound=ProgramOutput)


class Library(Protocol):
    """The open program outputs: add one, look one up, list them, name the group each is
    listed under, and subscribe to changes."""

    def add(self, program_output: OutputT) -> OutputT: ...

    def by_id(self, output_id: str, output_type: type[OutputT]) -> OutputT | None:
        """The open program output with this id, if it is an ``output_type``."""
        ...

    def matching(self, output_type: type[OutputT]) -> list[OutputT]:
        """Program outputs of this type, newest first."""
        ...

    def group_label(self, program_output: ProgramOutput) -> str:
        """The name of the group ``program_output`` is listed under."""
        ...

    def subscribe(self, callback: Callable[[], None]) -> None: ...

    def unsubscribe(self, callback: Callable[[], None]) -> None: ...
