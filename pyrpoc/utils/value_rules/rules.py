"""The common rules a value follows: whole numbers, decimals, text, paths, yes/no, a choice
from a list, and a set of channels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pyrpoc.core.program_input import Field, ParameterError


@dataclass(frozen=True)
class IntField(Field):
    minimum: int | None = None
    maximum: int | None = None
    step: int = 1

    def coerce(self, value: Any) -> int:
        if isinstance(value, bool):
            raise ParameterError(f"{self.label}: expected an integer, got a boolean")
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"{self.label}: expected an integer") from exc
        if self.minimum is not None and out < self.minimum:
            raise ParameterError(f"{self.label}: must be >= {self.minimum}")
        if self.maximum is not None and out > self.maximum:
            raise ParameterError(f"{self.label}: must be <= {self.maximum}")
        return out


@dataclass(frozen=True)
class FloatField(Field):
    minimum: float | None = None
    maximum: float | None = None
    step: float = 0.1
    decimals: int = 6

    def coerce(self, value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"{self.label}: expected a number") from exc
        if self.minimum is not None and out < self.minimum:
            raise ParameterError(f"{self.label}: must be >= {self.minimum}")
        if self.maximum is not None and out > self.maximum:
            raise ParameterError(f"{self.label}: must be <= {self.maximum}")
        return out


@dataclass(frozen=True)
class TextField(Field):
    def coerce(self, value: Any) -> str:
        return "" if value is None else str(value)


@dataclass(frozen=True)
class PathField(Field):
    dialog_filter: str = "All Files (*)"

    def coerce(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value)
        if not text.strip():
            return ""
        if text.rstrip().endswith(("\\", "/")):
            raise ParameterError(f"{self.label}: path must include a filename")
        return text


@dataclass(frozen=True)
class BoolField(Field):
    def coerce(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        raise ParameterError(f"{self.label}: expected true or false")


@dataclass(frozen=True)
class ChoiceField(Field):
    choices: tuple[str, ...] = ()

    def coerce(self, value: Any) -> str:
        text = str(value)
        if text not in self.choices:
            raise ParameterError(f"{self.label}: must be one of {list(self.choices)}")
        return text


@dataclass(frozen=True)
class ChannelsField(Field):
    """A row of toggleable channel buttons. The value is a sorted int tuple."""

    num_channels: int = 9

    def coerce(self, value: Any) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise ParameterError(f"{self.label}: expected a list of channel indices")
        try:
            out = sorted({int(item) for item in value})
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"{self.label}: channel indices must be integers") from exc
        for index in out:
            if index < 0 or index >= self.num_channels:
                raise ParameterError(
                    f"{self.label}: channel {index} is outside 0..{self.num_channels - 1}"
                )
        return tuple(out)

    def encode(self, value: Any) -> Any:
        return list(value)
