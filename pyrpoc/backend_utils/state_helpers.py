from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


def make_instance_id(prefix: str) -> str:
    token = (prefix or "item").strip().lower()
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in token)
    if not safe:
        safe = "item"
    return f"{safe}-{uuid4().hex[:12]}"


def _to_persistable(value: Any, depth: int = 6) -> tuple[bool, Any]:
    if depth <= 0:
        return False, None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True, value
    if isinstance(value, Path):
        return True, value
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            ok, converted = _to_persistable(item, depth=depth - 1)
            if not ok:
                return False, None
            out.append(converted)
        return True, out
    if isinstance(value, tuple):
        out_t: list[Any] = []
        for item in value:
            ok, converted = _to_persistable(item, depth=depth - 1)
            if not ok:
                return False, None
            out_t.append(converted)
        return True, out_t
    if isinstance(value, dict):
        out_d: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                return False, None
            ok, converted = _to_persistable(item, depth=depth - 1)
            if not ok:
                return False, None
            out_d[str(key)] = converted
        return True, out_d
    return False, None


def export_object_state(
    obj: object,
    *,
    include_fields: Iterable[str] | None = None,
    exclude_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    include = set(include_fields) if include_fields is not None else None
    exclude = set(exclude_fields or ())

    out: dict[str, Any] = {}
    for key, value in vars(obj).items():
        if include is not None and key not in include:
            continue
        if key in exclude:
            continue
        if key.startswith("_"):
            continue
        if callable(value):
            continue
        ok, persisted = _to_persistable(value)
        if ok:
            out[key] = persisted
    return out


def import_object_state(
    obj: object,
    state: dict[str, Any],
    *,
    include_fields: Iterable[str] | None = None,
    exclude_fields: Iterable[str] | None = None,
) -> None:
    include = set(include_fields) if include_fields is not None else None
    exclude = set(exclude_fields or ())
    for key, value in state.items():
        if include is not None and key not in include:
            continue
        if key in exclude:
            continue
        if include is None and not hasattr(obj, key):
            continue
        setattr(obj, key, value)
