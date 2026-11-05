"""``Program``: the declaration of one experiment, and ``RunContext``, what it
is handed while running."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pyrpoc.core.device import Device
from pyrpoc.core.program_input import InputSection, ProgramInput, ProgramInputValues
from pyrpoc.core.program_output import Data, ProgramOutput
from pyrpoc.core.registry import Registry

from .runner import Runner

D = TypeVar("D", bound=Device)
T = TypeVar("T")


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


@dataclass(frozen=True)
class EarlierRun:
    """An earlier run in the same Experiment that a program reads, named by the program that
    made it."""

    program_key: str


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

    # Earlier runs it reads; it cannot start until each has one to read.
    reads: list[EarlierRun] = []

    # Its inputs that a run may read again while it runs, to follow edits made after it
    # started. The rest stay as they were when it started.
    live: list[ProgramInput[Any]] = []

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


class LiveValues(Protocol):
    """The current values of one run's live inputs, safe to read from the run's thread."""

    def keys(self) -> frozenset[str]: ...

    def current(self, key: str) -> Any: ...


class RunContext:
    """What a running program is handed: its input values as they were when it started, the
    current values of its live inputs, the earlier runs it reads, its devices, its outputs,
    status and cancel."""

    def __init__(
        self,
        *,
        inputs: ProgramInputValues,
        live: LiveValues,
        earlier: Mapping[str, Mapping[str, ProgramOutput[Any]]],
        devices: Mapping[type[Device], Device],
        outputs: dict[str, ProgramOutput[Any]],
        cancel: threading.Event,
        on_status: Callable[[str], None],
    ):
        self.inputs = inputs
        self._live = live
        self._earlier = dict(earlier)
        self.devices = DeviceMap(devices)
        self.outputs = outputs
        self._cancel = cancel
        self._on_status = on_status

    def current(self, program_input: ProgramInput[T]) -> T:
        """The value of a live input now, which may have changed since the run started."""
        # Read between a program's own steps, so a value never changes under it mid-step.
        if program_input.key not in self._live.keys():
            raise KeyError(
                f"{program_input.key!r} is not declared in live; this program declares "
                f"{sorted(self._live.keys())}"
            )
        return self._live.current(program_input.key)

    def earlier(self, run: EarlierRun) -> Mapping[str, ProgramOutput[Any]]:
        """The outputs of the earlier run chosen for ``run``, by name, each holding its last
        value."""
        if run.program_key not in self._earlier:
            raise KeyError(
                f"{run.program_key!r} is not declared in reads; this program declares "
                f"{sorted(self._earlier)}"
            )
        return self._earlier[run.program_key]

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
