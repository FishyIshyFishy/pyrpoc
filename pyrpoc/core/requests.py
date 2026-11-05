"""Requests: ``Request`` declarations of the commands running programs accept, and the
``Requests`` collection panels post them to."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from .program_input import Field

T = TypeVar("T")


# Every declared request, by key.
request_registry: dict[str, Request[Any]] = {}


@dataclass(frozen=True, eq=False)
class Request(Generic[T]):
    """The declaration of one command a running program may accept: its key, and the rules
    the value posted with it follows (which carry its label)."""

    key: str
    rule: Field

    def __post_init__(self) -> None:
        # Declared once, so a key in a recording's log can never mean two commands.
        if self.key in request_registry:
            raise ValueError(f"request {self.key!r} is already declared")
        request_registry[self.key] = self


class Requests(Protocol):
    """Commands posted for running programs: each is taken once by every run that accepts
    it."""

    def accepted(self) -> frozenset[Request[Any]]:
        """The requests some running program accepts now."""
        ...

    def post(self, request: Request[T], value: T) -> bool:
        """Check ``value`` against the request's rules and give it to every running program
        that accepts it; returns whether one did. Raises ``ParameterError`` for a value the
        rules refuse."""
        ...

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` after what is accepted changes."""
        ...

    def unsubscribe(self, callback: Callable[[], None]) -> None: ...
