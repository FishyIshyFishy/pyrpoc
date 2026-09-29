"""One exception tree for the whole application.

Replaces the three separate ``DaqUnavailableError`` classes that lived one per
``modalities/*/acquisition_core.py``, where ``except DaqUnavailableError``
imported from one module did not catch the other two.
"""

from __future__ import annotations


class PyrpocError(Exception):
    """Base for every error this application raises deliberately."""


class ParameterError(PyrpocError):
    """A parameter value is missing, out of range, or the wrong type."""


class Cancelled(PyrpocError):
    """Raised inside a running program when the run has been stopped.

    Propagates out through ``Program.run``, which is what makes a program's
    ``finally`` blocks the teardown mechanism for a cancelled run.
    """


class DeviceError(PyrpocError):
    """A device could not be reached or configured."""


class MissingDevice(DeviceError):
    """A program needs devices that are not in the inventory."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("missing required devices: " + ", ".join(self.missing))


class DaqError(DeviceError):
    """An NI-DAQ operation failed."""


class TaggerError(DeviceError):
    """A TimeTagger operation failed."""
