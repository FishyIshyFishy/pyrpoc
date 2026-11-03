"""``ProgramInputsStore``: the current value of every program input, by key."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from pyrpoc.core import program_inputs as P
from pyrpoc.core.library import Library

B = TypeVar("B", bound=P.Group)


class ProgramInputsStore:
    """Every program input's current value, keyed by ``<Block>.<field>``. One
    value per key, shared by every program that declares it."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._subscribers: list[Callable[[str], None]] = []

    @staticmethod
    def spec(key: str) -> P.Field:
        """The rules for the input at ``key``."""
        name, attr = P.split_path(key)
        return dict(P.block_fields(P.block_registry.get(name)))[attr]

    def get(self, key: str) -> Any:
        """The current value, which is the input's default until one is set."""
        if key not in self._values:
            name, attr = P.split_path(key)
            self._values[key] = getattr(P.block_registry.get(name)(), attr)
        return self._values[key]

    def set(self, key: str, value: Any) -> None:
        """Check ``value`` against the input's rules, store it and tell every
        subscriber. Raises ``ParameterError`` for a value the rules refuse."""
        self._values[key] = self.spec(key).coerce(value)
        for callback in list(self._subscribers):
            callback(key)

    def subscribe(self, callback: Callable[[str], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def block(self, cls: type[B]) -> B:
        """A new ``cls`` holding the current values of its inputs."""
        return cls(**{name: self.get(f"{cls.__name__}.{name}") for name, _ in P.block_fields(cls)})

    def validate(self, classes: Sequence[type[P.Group]]) -> None:
        """Re-check the current values of every input of ``classes``."""
        for cls in classes:
            P.validate_block(self.block(cls))

    def blocks_for(self, classes: Sequence[type[P.Group]], library: Library) -> P.BlockMap:
        """Copies of ``classes`` as a run sees them, with values that name open data
        filled in."""
        return P.BlockMap({cls: P.resolve_block(self.block(cls), library) for cls in classes})

    def encode_blocks(self, classes: Sequence[type[P.Group]]) -> dict[str, Any]:
        """The values of ``classes`` as JSON, one object per block."""
        return {cls.__name__: P.encode_block(self.block(cls)) for cls in classes}

    def to_dict(self) -> dict[str, Any]:
        """Every value set or read so far, as JSON, by key."""
        return {key: self.spec(key).encode(value) for key, value in self._values.items()}

    def load_dict(self, raw: Mapping[str, Any]) -> None:
        """Take values from saved JSON, skipping any whose input no longer
        exists or whose value its rules refuse."""
        for key, value in raw.items():
            try:
                self._values[key] = self.spec(key).decode(value)
            except (KeyError, P.ParameterError):
                continue

    def clear(self) -> None:
        """Forget every value, so each input is back at its default."""
        self._values.clear()
