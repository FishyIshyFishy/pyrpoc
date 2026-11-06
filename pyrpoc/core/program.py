"""``Program``: the declaration of one experiment, and ``RunContext``, what it
is handed while running."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol, TypeVar

from pyrpoc.core.check import Check
from pyrpoc.core.device import Device
from pyrpoc.core.program_input import ProgramInput, ProgramInputValues
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.registry import Registry

from .runner import Runner

D = TypeVar("D", bound=Device)
OutputT = TypeVar("OutputT", bound=ProgramOutput)
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


class Program:
    """One experiment: the devices it uses, the program inputs it reads, its outputs, the
    runners that start it, and ``run``, which does the work."""

    display_name: str = "Program"

    # The heading it is listed under, and its place in the program list.
    group: str
    order: int

    # Device classes to claim. Claims propagate along ``backed_by``.
    devices: list[type[Device]] = []

    # Every program input it reads.
    inputs: list[ProgramInput[Any]] = []

    # Screen layout only: headings, each over some of its inputs. An input on no card gets no
    # row, and with no cards every input is shown.
    cards: dict[str, Sequence[ProgramInput[Any]]] = {}

    # Outputs of earlier recordings in the same Experiment it reads, by the name it reads each
    # under, and the type each must be. It cannot start until each has one to read.
    reads: dict[str, type[ProgramOutput]] = {}

    # Its inputs that a run may read again while it runs, to follow edits made after it
    # started. The rest stay as they were when it started.
    live: list[ProgramInput[Any]] = []

    # Named outputs, and the type of each.
    emits: dict[str, type[ProgramOutput]] = {}

    # The ways this program can be started, in the order their controls are drawn.
    runners: list[Runner] = []

    # Conditions of its own for starting, judged with the rest of what a run needs.
    checks: list[Check] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A card showing an input the program does not read would edit a value no run gets.
        unread = [
            each.key for card in cls.cards.values() for each in card if each not in cls.inputs
        ]
        if unread:
            raise TypeError(f"{cls.__name__}'s cards show inputs it does not read: {unread}")

    def run(self, ctx: RunContext) -> None:
        raise NotImplementedError


# note to self: python protocols are useful just for type checkers
class RunInbox(Protocol):
    """What has been posted to one run and not yet taken, safe to take from the run's thread:
    the changes to its live inputs."""

    def live_keys(self) -> frozenset[str]: ...

    def current(self, key: str) -> Any:
        """A live input's value now, taking a change posted since it was last asked for."""
        ...


class RunContext:
    """What a running program is handed for one part of its recording: its input values as
    they were when the part began, its inbox, which gives the current values of its live inputs,
    the earlier outputs it reads, its devices, the part's outputs, status and cancel."""

    def __init__(
        self,
        *,
        inputs: ProgramInputValues,
        inbox: RunInbox,
        earlier: Mapping[str, ProgramOutput],
        devices: Mapping[type[Device], Device],
        outputs: dict[str, ProgramOutput],
        cancel: threading.Event,
        on_status: Callable[[str], None],
    ):
        self.inputs = inputs
        self._inbox = inbox
        self._earlier = dict(earlier)
        self.devices = DeviceMap(devices)
        self.outputs = outputs
        self._cancel = cancel
        self._on_status = on_status

    def current(self, program_input: ProgramInput[T]) -> T:
        """The value of a live input now, which may have changed since the run started."""
        # Read between a program's own steps, so a value never changes under it mid-step.
        if program_input.key not in self._inbox.live_keys():
            raise KeyError(
                f"{program_input.key!r} is not declared in live; this program declares "
                f"{sorted(self._inbox.live_keys())}"
            )
        return self._inbox.current(program_input.key)

    def earlier(self, name: str, output_type: type[OutputT]) -> OutputT:
        """The earlier output chosen for ``name``, which ``reads`` declares as
        ``output_type``. It is read, never added to."""
        if name not in self._earlier:
            raise KeyError(
                f"{name!r} is not declared in reads; this program declares {sorted(self._earlier)}"
            )
        program_output = self._earlier[name]
        # Narrows for the type checker, and catches a program asking for the wrong type.
        if not isinstance(program_output, output_type):
            raise TypeError(
                f"{name!r} is a {type(program_output).__name__}, not {output_type.__name__}"
            )
        return program_output

    def output(self, name: str, output_type: type[OutputT]) -> OutputT:
        """This run's output ``name``, which ``emits`` declares as ``output_type``."""
        if name not in self.outputs:
            raise KeyError(
                f"{name!r} is not declared in emits; this program declares {sorted(self.outputs)}"
            )
        program_output = self.outputs[name]
        # Narrows for the type checker, and catches a program asking for the wrong type.
        if not isinstance(program_output, output_type):
            raise TypeError(
                f"{name!r} is a {type(program_output).__name__}, not {output_type.__name__}"
            )
        return program_output

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
