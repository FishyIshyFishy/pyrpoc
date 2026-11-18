"""JSON files: one value written whole, or a log of values, one JSON line each."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, content: Any) -> None:
    """``content`` as the whole of ``path``."""
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_json_line(path: Path, content: Any) -> None:
    """``content`` as one more line of ``path``, which is made if it does not exist."""
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(content) + "\n")


def read_json_lines(path: Path) -> list[Any]:
    """Every line of ``path`` as the value it holds, oldest first."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]
