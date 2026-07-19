"""The instrument base: a controllable piece of hardware. Qt-free.

Minimal by design — identity plus connect/test. Each device adds its own methods,
and a modality that uses an instrument just calls those methods directly. Editor
widgets live in gui and are matched to an instrument by its key.
"""

from __future__ import annotations

from pyrpoc_next.structs.keys import InstrumentKey
from pyrpoc_next.structs.status import ConnectionStatus


class Instrument:
    """Base for one instrument instance."""

    key: InstrumentKey
    display_name: str = "Instrument"

    def __init__(self):
        self.status = ConnectionStatus.untested

    def connect(self) -> bool:
        """Open the device connection. Override for real hardware."""
        self.status = ConnectionStatus.ok
        return True

    def test_connection(self) -> bool:
        """Probe the device and record the outcome in status."""
        ok = self.connect()
        self.status = ConnectionStatus.ok if ok else ConnectionStatus.failed
        return ok

    def summary(self) -> str:
        """Short text for the inventory card."""
        return f"{self.display_name}: {self.status.value}"
