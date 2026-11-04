"""``Program``: the declaration of one experiment, and ``RunContext``, what it
is handed while running."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar

from pyrpoc.core.device import Device
from pyrpoc.core.program_input import InputSection, ProgramInput, ProgramInputValues
from pyrpoc.core.program_output import Data, ProgramOutput
from pyrpoc.core.registry import Registry

from .runner import Runner

D = TypeVar("D", bound=Device)


class Cancelled(Exception):
    """Raised inside a running program when its run is stopped."""


class DeviceMap(Mapping[type[Device], Device]):
    """The devices resolved for one run, keyed by class, where ``ctx.devices[DAQ]`` is
    typed as a ``DAQ``."""

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
    """One experiment: the devices it uses, the program inputs it reads (laid out
    in input sections), its outputs, the runners that start it, and ``run``,
    which does the work."""

    display_name: str = "Program"

    # The heading it is listed under, and its place in the program list.
    group: str
    order: int

    # Device classes to claim. Claims propagate along ``backed_by``.
    devices: list[type[Device]] = []

    # The program inputs it reads, grouped as a panel shows them.
    sections: list[InputSection] = []

    # Named outputs, and the kind of ``Data`` each one carries.
    emits: dict[str, type[Data]] = {}

    # The ways this program can be started, in the order their controls are drawn.
    runners: list[Runner] = []

    def run(self, ctx: RunContext) -> None:
        raise NotImplementedError

    @classmethod
    def inputs(cls) -> list[ProgramInput[Any]]:
        """Every program input it reads, in section order."""
        return [each for section in cls.sections for each in section.inputs]


class RunContext:
    """What a running program is handed: its input values, its devices, its
    outputs, status and cancel."""

    def __init__(
        self,
        *,
        inputs: ProgramInputValues,
        devices: Mapping[type[Device], Device],
        outputs: dict[str, ProgramOutput[Any]],
        cancel: threading.Event,
        on_status: Callable[[str], None],
    ):
        self.inputs = inputs
        self.devices = DeviceMap(devices)
        self.outputs = outputs
        self._cancel = cancel
        self._on_status = on_status

    def output_for(self, output: str) -> ProgramOutput[Any]:
        if output not in self.outputs:
            raise KeyError(
                f"{output!r} is not declared in emits; this program declares {sorted(self.outputs)}"
            )
        return self.outputs[output]

    def publish(self, output: str, value: Data) -> None:
        """Make ``value``, of the kind ``emits`` declares, the current value of
        one of this run's outputs."""
        program_output = self.output_for(output)
        if type(value) is not program_output.kind:
            raise TypeError(
                f"{output!r} carries {program_output.kind.name}, not {type(value).name}"
            )
        program_output.publish(value)

    def describe(self, output: str, **metadata: Any) -> None:
        """Record metadata about one output, once per run."""
        self.output_for(output).metadata.update(metadata)

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


# Programs keep their key at the registration site, not on the class.
program_registry: Registry[Program] = Registry("ProgramRegistry", Program, stamp=None)
