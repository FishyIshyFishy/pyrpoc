"""``Roster``: every unit a loop has had, known by key and numbered as each joined."""

from __future__ import annotations

from collections.abc import Iterable

from pyrpoc.plugins.program_io.outputs import Unit, UnitRecord


class Roster:
    """Every unit the loop has had, those that left included, so a number is never given twice
    and a key never joins twice."""

    def __init__(self, records: Iterable[UnitRecord]):
        self.by_key = {record.unit.key: record for record in records}

    def records(self) -> tuple[UnitRecord, ...]:
        return tuple(sorted(self.by_key.values(), key=lambda record: record.number))

    def has(self, key: str) -> bool:
        """Whether the unit with ``key`` ever joined, even if it has left since."""
        return key in self.by_key

    def present(self, key: str) -> UnitRecord | None:
        """The unit with ``key``, if it joined and has not left."""
        record = self.by_key.get(key)
        return record if record is not None and record.left_s is None else None

    def join(self, unit: Unit, now_s: float) -> UnitRecord:
        record = UnitRecord(len(self.by_key) + 1, unit, now_s, None, False)
        self.by_key[unit.key] = record
        return record

    def update(self, record: UnitRecord) -> None:
        self.by_key[record.unit.key] = record
