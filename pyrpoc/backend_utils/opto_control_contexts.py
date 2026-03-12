from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaseOptoControlContext:
    optocontrol_key: str
    alias: str


@dataclass(frozen=True)
class MaskContext(BaseOptoControlContext):
    mask: Any
    daq_port: int
    daq_line: int

