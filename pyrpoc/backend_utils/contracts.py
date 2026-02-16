from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Parameter:
    label: str
    param_type: type
    default: Any = None
    required: bool = True
    tooltip: str = ""
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    choices: list[str] | None = None


@dataclass
class Action:
    label: str
    method_name: str
    parameters: list[Parameter] = field(default_factory=list)
    tooltip: str = ""
    dangerous: bool = False
    confirm_text: str | None = None


ParameterGroups = dict[str, list[Parameter]]
