"""``RequestsStore``: where panels post requests, each given to the inbox of every running run
that accepts it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pyrpoc.core.requests import Request

from .inbox import RunInbox

T = TypeVar("T")


class RequestsStore:
    """The inbox of every run still going, and who to tell when what they accept changes."""

    def __init__(self) -> None:
        self.inboxes: list[RunInbox] = []
        self._subscribers: list[Callable[[], None]] = []

    def accepted(self) -> frozenset[Request[Any]]:
        """The requests some running program accepts now."""
        return frozenset(each for inbox in self.inboxes for each in inbox.accepts())

    def post(self, request: Request[T], value: T) -> bool:
        """Check ``value`` against the request's rules and give it to every running program
        that accepts it; returns whether one did."""
        coerced = request.rule.coerce(value)
        targets = [inbox for inbox in self.inboxes if request in inbox.accepts()]
        for inbox in targets:
            inbox.give(request, coerced)
        return bool(targets)

    def open(self, inbox: RunInbox) -> None:
        """Give a new run's inbox the requests posted until it is closed."""
        self.inboxes.append(inbox)
        self.notify()

    def close(self, inbox: RunInbox) -> None:
        """Stop giving a run requests, on the thread that posts them."""
        self.inboxes.remove(inbox)
        self.notify()

    def subscribe(self, callback: Callable[[], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def notify(self) -> None:
        for callback in list(self._subscribers):
            callback()
