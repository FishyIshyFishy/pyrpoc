"""``Peek``: whether the overlay is up, from presses and releases of its key. Held, it shows until
released; tapped, it stays until the key is pressed again."""

from __future__ import annotations

from enum import Enum

# A press released sooner than this is a tap, which pins the overlay.
TAP_S = 0.25


class PeekState(Enum):
    CLOSED = "closed"
    PEEKING = "peeking"
    PINNED = "pinned"


class Peek:
    def __init__(self) -> None:
        self.state = PeekState.CLOSED
        self.pressed_s = 0.0
        # The press that closed a pinned overlay; its release must not open it again.
        self.closing = False

    @property
    def shown(self) -> bool:
        return self.state is not PeekState.CLOSED

    def press(self, now_s: float) -> PeekState:
        if self.state is PeekState.CLOSED:
            self.state, self.pressed_s = PeekState.PEEKING, now_s
        elif self.state is PeekState.PINNED:
            self.state, self.closing = PeekState.CLOSED, True
        return self.state

    def release(self, now_s: float) -> PeekState:
        if self.closing:
            self.closing = False
        elif self.state is PeekState.PEEKING:
            held = now_s - self.pressed_s
            self.state = PeekState.PINNED if held < TAP_S else PeekState.CLOSED
        return self.state

    def pin(self) -> None:
        """Open and keep open, whatever the key is doing."""
        self.state, self.closing = PeekState.PINNED, False

    def close(self) -> None:
        self.state, self.closing = PeekState.CLOSED, False
