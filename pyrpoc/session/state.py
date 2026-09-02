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

#: Bumped from 6. Parameters are stored differently -- nested groups keyed by
#: program rather than a flat list keyed by widget label -- and instruments have
#: become devices, so a v6 file has nothing to map onto. There is no converter:
#: a v6 session loads as defaults, once.
SCHEMA_VERSION = 7


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
class SessionState:
    schema_version: int = SCHEMA_VERSION
    theme_mode: str = "system"
    devices: list[DeviceState] = field(default_factory=list)
    views: list[ViewState] = field(default_factory=list)
    selected_program: str | None = None
    params_by_program: dict[str, dict[str, Any]] = field(default_factory=dict)
    ads_layout: str | None = None

    def is_empty(self) -> bool:
        return not self.devices and not self.views and not self.params_by_program
