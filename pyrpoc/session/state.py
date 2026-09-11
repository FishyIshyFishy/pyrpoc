"""What a saved session holds.

Configuration only: what exists, how it is configured, and the layout. Small,
JSON, and there so the workbench comes back on relaunch.

Acquired data is deliberately not here. It is large, lives in TIFF, and exists
because it is the experimental result -- a different size, format, lifetime and
reason to exist. Merge them and the session file starts trying to hold arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Bumped from 7. Parameters are no longer stored per program: a block is
#: shared by every modality that declares it, so there is one flat state dict
#: keyed by block class name instead of a nested dict keyed by program. A v7
#: file's ``params_by_program`` has no single answer to map onto -- three
#: programs could each hold a different ScanGroup -- so there is no converter
#: and a v7 session loads as defaults, once.
SCHEMA_VERSION = 8


@dataclass
class DeviceState:
    key: str
    instance_id: str = ""
    user_label: str | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewState:
    key: str
    instance_id: str = ""
    user_label: str | None = None
    visible: bool = True
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class SaveState:
    """What acquisitions are called and where they are written.

    One block, not one per program: saving is not something a program decides.
    """

    name: str = "acquisition"
    directory: str = ""
    enabled: bool = False


@dataclass
class SessionState:
    schema_version: int = SCHEMA_VERSION
    devices: list[DeviceState] = field(default_factory=list)
    views: list[ViewState] = field(default_factory=list)
    selected_program: str | None = None
    #: Every parameter block, keyed by class name. The state dict: one entry
    #: per block, not one per program, because a block is shared.
    param_blocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    save: SaveState = field(default_factory=SaveState)
    ads_layout: str | None = None

    def is_empty(self) -> bool:
        return not self.devices and not self.views and not self.param_blocks
