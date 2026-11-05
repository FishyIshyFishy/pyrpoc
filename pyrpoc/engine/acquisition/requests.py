"""``RequestsStore``: where panels post requests, and ``RequestInbox``, the requests posted to
one run."""

# Panels post on the GUI thread and a run takes on its own, so each inbox is guarded by a lock.

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pyrpoc.core.recording import utc_now
from pyrpoc.core.requests import Request
from pyrpoc.engine.data_library.format import TakenRequest

T = TypeVar("T")


class RequestInbox:
    """The requests posted to one run and not yet taken, each with when it was posted."""

    def __init__(
        self,
        accepts: Sequence[Request[Any]],
        on_taken: Callable[[TakenRequest], None] | None,
    ):
        self.on_taken = on_taken
        self.lock = threading.Lock()
        self.pending: dict[Request[Any], list[tuple[str, Any]]] = {each: [] for each in accepts}

    def accepts(self) -> frozenset[Request[Any]]:
        return frozenset(self.pending)

    def give(self, request: Request[Any], value: Any) -> None:
        with self.lock:
            self.pending[request].append((utc_now(), value))

    def take(self, request: Request[Any]) -> list[Any]:
        """The values posted with ``request`` since it was last taken, oldest first, each
        logged as taken."""
        with self.lock:
            taken, self.pending[request] = self.pending[request], []
        # None for a run that is not saved: there is no recording to log into.
        if self.on_taken is not None:
            taken_at = utc_now()
            for posted_at, value in taken:
                encoded = request.rule.encode(value)
                self.on_taken(TakenRequest(posted_at, taken_at, request.key, encoded))
        return [value for _, value in taken]


class RequestsStore:
    """The inbox of every run still going, and who to tell when what they accept changes."""

    def __init__(self) -> None:
        self.inboxes: list[RequestInbox] = []
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

    def open(
        self, accepts: Sequence[Request[Any]], on_taken: Callable[[TakenRequest], None] | None
    ) -> RequestInbox:
        """A new run's inbox, taking posts until it is closed."""
        inbox = RequestInbox(accepts, on_taken)
        self.inboxes.append(inbox)
        self.notify()
        return inbox

    def close(self, inbox: RequestInbox) -> None:
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
