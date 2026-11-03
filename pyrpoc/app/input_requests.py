"""``OpenInputRequests``: the app's one set of open input requests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pyrpoc.core.input_requests import InputRequest

T = TypeVar("T")


class OpenInputRequests:
    """The open input requests, at most one for each kind of value, and who is
    told when one opens or closes."""

    def __init__(self) -> None:
        self._open: dict[type, InputRequest[Any]] = {}
        self._subscribers: list[Callable[[], None]] = []

    def request(
        self, kind: type[T], prompt: str, on_answer: Callable[[T | None], None]
    ) -> InputRequest[T]:
        """Open a request, cancelling any open one for the same kind."""
        previous = self._open.get(kind)
        if previous is not None:
            self.cancel(previous)
        request = InputRequest(kind, prompt, on_answer)
        self._open[kind] = request
        self.notify()
        return request

    def cancel(self, request: InputRequest[Any]) -> None:
        """Close ``request`` unanswered, if it is still open."""
        if self.close(request):
            request.on_answer(None)

    def is_open(self, kind: type) -> bool:
        return kind in self._open

    def answer(self, value: object) -> None:
        """Answer the open request for the type of ``value``, if there is one."""
        request = self._open.get(type(value))
        if request is not None and self.close(request):
            request.on_answer(value)

    def close(self, request: InputRequest[Any]) -> bool:
        """Take ``request`` off the open list, returning whether it was open."""
        # Closed and announced before the answer is handed on, so no path out of
        # a request leaves a panel waiting for it.
        if self._open.get(request.kind) is not request:
            return False
        del self._open[request.kind]
        request.open = False
        self.notify()
        return True

    def subscribe(self, callback: Callable[[], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def notify(self) -> None:
        for callback in list(self._subscribers):
            callback()
