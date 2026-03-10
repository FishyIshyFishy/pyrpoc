from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Action:
    label: str
    method_name: str
    parameters: list["BaseParameter"] = field(default_factory=list)
    tooltip: str = ""
    dangerous: bool = False
    confirm_text: str | None = None


ParameterGroups = dict[str, list["BaseParameter"]]
