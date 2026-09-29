"""What a program is, and the service surface it gets while running.

``uses``, ``params`` and ``emits`` are what the executor must know to start a
program; ``runners`` are the ways it can be started, which depend on the
program. ``display_name`` is its one word of presentation, as a device has.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar

import numpy as np

from .data import Data, Dataset
from .device import Device
from .runner import Runner

D = TypeVar("D", bound=Device)


class Cancelled(Exception):
    """Raised inside a running program when the run has been stopped, so a
    program's ``finally`` blocks are its teardown."""


class DeviceMap(Mapping[type[Device], Device]):
    """The devices resolved for one run, keyed by class, so ``ctx.devices[DAQ]``
    is typed as a ``DAQ``."""

    def __init__(self, devices: Mapping[type[Device], Device]):
        self._devices: dict[type[Device], Device] = dict(devices)

    def __getitem__(self, key: type[D]) -> D:
        device = self._devices[key]
        # Narrows for the type checker: devices are keyed by their own class.
        if not isinstance(device, key):
            raise TypeError(f"{device!r} is stored under {key.__name__} but is not one")
        return device

    def __iter__(self) -> Iterator[type[Device]]:
        return iter(self._devices)

    def __len__(self) -> int:
        return len(self._devices)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DeviceMap({self._devices!r})"


class Program:
    """Subclasses define ``display_name``, ``uses``, ``params``, ``emits``,
    ``runners`` and ``run``."""

    display_name: str = "Program"

    # Device classes to claim. Claims propagate along ``backed_by``.
    uses: list[type[Device]] = []

    # Parameter blocks, in form order. Declaring a block shares it with every
    # other program that declares it.
    params: list[type] = []

    # Named outputs, and the kind of ``Data`` each one carries.
    emits: dict[str, type[Data]] = {}

    # The ways this program can be started, in the order their controls are drawn.
    runners: list[Runner] = []

    def run(self, ctx: RunContext) -> None:
        raise NotImplementedError


class RunContext:
    """The service surface handed to a running program. Programs call it;
    they never implement it."""

    def __init__(
        self,
        *,
        params: Any,
        devices: Mapping[type[Device], Device],
        datasets: dict[str, Dataset],
        cancel: threading.Event,
        on_status: Callable[[str], None],
    ):
        self.params = params
        self.devices = DeviceMap(devices)
        self.datasets = datasets
        self._cancel = cancel
        self._on_status = on_status

    def dataset_for(self, output: str) -> Dataset:
        if output not in self.datasets:
            raise KeyError(
                f"{output!r} is not declared in emits; this program declares "
                f"{sorted(self.datasets)}"
            )
        return self.datasets[output]

    def publish(self, output: str, data: np.ndarray, *, channels=None) -> None:
        """Write one array into one of this run's datasets."""
        dataset = self.dataset_for(output)
        if channels and not dataset.channel_labels:
            dataset.channel_labels = list(channels)
        dataset.append(data)

    def describe(self, output: str, **metadata: Any) -> None:
        """Record per-output metadata, set once rather than per frame."""
        self.dataset_for(output).metadata.update(metadata)

    def status(self, text: str) -> None:
        self._on_status(text)

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled("run stopped")

    def sleep(self, seconds: float) -> None:
        """Wait, but return early if the run is stopped."""
        if seconds <= 0:
            self.check_cancel()
            return
        if self._cancel.wait(seconds):
            raise Cancelled("run stopped")
