"""Input requests: a panel or runner asking for one value that another panel supplies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from .data import Image
from .program_output import ProgramOutput

T = TypeVar("T")


@dataclass(frozen=True)
class PixelPick:
    """The value a click on an image supplies: the program output clicked on, and
    the pixel's ``x`` along the fast axis and ``y`` along the slow."""

    output: ProgramOutput[Image]
    x: int
    y: int


class InputRequest(Generic[T]):
    """One question for a value of type ``kind``. ``on_answer`` gets the value, or
    None if the request is cancelled; ``open`` is False once either happened."""

    def __init__(self, kind: type[T], prompt: str, on_answer: Callable[[T | None], None]):
        self.kind = kind
        self.prompt = prompt
        self.on_answer = on_answer
        self.open = True


class InputRequests(Protocol):
    """The open input requests, at most one for each kind of value."""

    def request(
        self, kind: type[T], prompt: str, on_answer: Callable[[T | None], None]
    ) -> InputRequest[T]:
        """Open a request, cancelling any open one for the same kind."""
        ...

    def cancel(self, request: InputRequest[Any]) -> None:
        """Close ``request`` unanswered, if it is still open."""
        ...

    def is_open(self, kind: type) -> bool:
        """Whether a request for ``kind`` is waiting for an answer."""
        ...

    def answer(self, value: object) -> None:
        """Answer the open request for the type of ``value``, if there is one."""
        ...

    def subscribe(self, callback: Callable[[], None]) -> None: ...

    def unsubscribe(self, callback: Callable[[], None]) -> None: ...
