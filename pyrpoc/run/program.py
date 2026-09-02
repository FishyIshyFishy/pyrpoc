"""What a program is, and the service surface it gets while running.

A program is something with a ``run()`` that drives hardware and emits data over
time. Its three attributes are not a declaration format; they are the three
things the runner must know in order to start it. Nothing about labels, menus or
how it was launched -- a program should not know it is in a dropdown.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np

from pyrpoc.core.errors import Cancelled
from pyrpoc.core.streams import Stream

if TYPE_CHECKING:  # pragma: no cover
    from pyrpoc.data.dataset import Dataset
    from pyrpoc.devices.base import Device


class Program:
    """Subclasses define exactly ``uses``, ``params``, ``emits`` and ``run``."""

    #: Device classes to claim. Claims propagate along ``backed_by``.
    uses: list[type["Device"]] = []

    #: The parameter dataclass this program is configured with.
    params: type | None = None

    #: Named output streams and their shape contracts.
    emits: dict[str, type[Stream]] = {}

    def run(self, ctx: "RunContext") -> None:
        raise NotImplementedError


class RunContext:
    """The service surface handed to a running program.

    One concrete class, written once and never subclassed. Programs call it;
    they never implement it. It is the same idea as v3.0's five loose
    ``acquire_continuous`` arguments, given a name and one place to live.
    """

    def __init__(
        self,
        *,
        params: Any,
        devices: dict[type["Device"], "Device"],
        datasets: dict[str, "Dataset"],
        cancel: threading.Event,
        continuous: bool = False,
        on_status: Callable[[str], None] | None = None,
    ):
        self.params = params
        self.devices = devices
        self.datasets = datasets
        self.continuous = continuous
        self._cancel = cancel
        self._on_status = on_status

    # -- output ------------------------------------------------------------ #

    def publish(self, stream: str, data: np.ndarray, *, channels=None, **coords: Any) -> int:
        """Write one array into one of this run's datasets.

        The stream name is declared in ``emits``, so a view binding exists
        before the run starts rather than being inferred from a tag mid-flight.
        """
        dataset = self.datasets.get(stream)
        if dataset is None:
            raise KeyError(
                f"{stream!r} is not declared in emits; this program declares "
                f"{sorted(self.datasets)}"
            )
        if channels and not dataset.channel_labels:
            dataset.channel_labels = list(channels)
        return dataset.append(data, **coords)

    def describe(self, stream: str, **metadata: Any) -> None:
        """Record metadata on one of this run's datasets.

        FLIM uses it for laser_period_ps / binwidth_ps / n_bins, which v3.0
        attached to every AcquiredData it emitted. Per stream and set once, not
        per frame.
        """
        dataset = self.datasets.get(stream)
        if dataset is None:
            raise KeyError(f"{stream!r} is not declared in emits")
        dataset.metadata.update(metadata)

    def status(self, text: str) -> None:
        if self._on_status is not None:
            self._on_status(text)

    # -- control flow ------------------------------------------------------- #

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled("run stopped")

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def frames(self, count: int | None = None) -> Iterator[int]:
        """Iterate frame indices, checking for cancellation before each one.

        ``count`` is ignored when the run was started in continuous mode, which
        is how the Continuous button survives without the program's body
        changing and without overwriting the user's stored ``num_frames``.
        """
        index = 0
        limit = None if self.continuous else count
        while limit is None or index < limit:
            self.check_cancel()
            yield index
            index += 1

    def sleep(self, seconds: float) -> None:
        """Wait, but return early if the run is stopped."""
        if seconds <= 0:
            self.check_cancel()
            return
        if self._cancel.wait(seconds):
            raise Cancelled("run stopped")
