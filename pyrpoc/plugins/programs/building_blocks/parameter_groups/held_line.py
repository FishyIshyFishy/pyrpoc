"""Digital lines to hold high outside a scan, such as AOM gates while aligning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyrpoc.structs.plugins.params import Group, ParameterError, block, int_field, text_field


def parse_lines(text: str) -> list[int]:
    """``"5, 6"`` or ``"5-7"`` as sorted line numbers, each once."""
    lines: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        try:
            start = int(first)
            stop = int(last) if last else start
        except ValueError as exc:
            raise ParameterError(f"Lines: '{part}' is not a line or a range like 5-7") from exc
        if start < 0 or stop < start:
            raise ParameterError(f"Lines: '{part}' is not a line or a range like 5-7")
        lines.update(range(start, stop + 1))
    if not lines:
        raise ParameterError("Lines: name at least one line")
    return sorted(lines)


@block
@dataclass
class HeldLineGroup(Group):
    label: ClassVar[str] = "Held Lines"

    port: int = int_field("Port", 0, minimum=0, tooltip="Digital port of the lines to hold high")
    lines: str = text_field(
        "Lines", "5", tooltip="Lines to hold high: a list and/or ranges, e.g. 5, 6 or 4-7"
    )

    def channels(self, device_name: str) -> list[str]:
        """The NI-DAQ channel strings, named as a mask's are."""
        return [f"{device_name}/port{self.port}/line{line}" for line in parse_lines(self.lines)]
