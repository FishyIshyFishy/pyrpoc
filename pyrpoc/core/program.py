"""``Program``: the declaration of one experiment, and ``RunContext``, what it
is handed while running."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol, TypeVar

from pyrpoc.core.device import Device
from pyrpoc.core.program_input import ProgramInput, ProgramInputValues
from pyrpoc.core.program_output import ProgramOutput
from pyrpoc.core.registry import Registry
from pyrpoc.core.requests import Request

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

    # Devices the user chooses from the inventory, by the name it uses each under, and the class
    # each must be. Claims propagate along ``backed_by``, as for ``devices``.
    chooses: dict[str, type[Device]] = {}

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

    # The requests panels may post to a run of it, which it takes when it chooses.
    requests: list[Request[Any]] = []

    # Named outputs, and the type of each.
    emits: dict[str, type[ProgramOutput]] = {}

    # The ways this program can be started, in the order their controls are drawn.
    runners: list[Runner] = []

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


class LiveValues(Protocol):
    """The current values of one run's live inputs, safe to read from the run's thread."""

    def keys(self) -> frozenset[str]: ...

    def current(self, key: str) -> Any: ...


class PostedRequests(Protocol):
    """The requests posted to one run and not yet taken, safe to take from the run's
    thread."""

    def accepts(self) -> frozenset[Request[Any]]: ...

    def take(self, request: Request[Any]) -> list[Any]:
        """The values posted with ``request`` since it was last taken, oldest first."""
        ...


class RunContext:
    """What a running program is handed: its input values as they were when it started, the
    current values of its live inputs, the requests posted to it, the earlier outputs it reads,
    its devices and the devices chosen for it, its outputs, status and cancel."""

    def __init__(
        self,
        *,
        inputs: ProgramInputValues,
        live: LiveValues,
        requests: PostedRequests,
        earlier: Mapping[str, ProgramOutput],
        devices: Mapping[type[Device], Device],
        chosen: Mapping[str, Device],
        outputs: dict[str, ProgramOutput],
        cancel: threading.Event,
        on_status: Callable[[str], None],
    ):
        self.inputs = inputs
        self._live = live
        self._requests = requests
        self._earlier = dict(earlier)
        self.devices = DeviceMap(devices)
        self._chosen = dict(chosen)
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

    def requests(self, request: Request[T]) -> list[T]:
        """The values posted with ``request`` since the run last asked, oldest first; each is
        given once."""
        if request not in self._requests.accepts():
            raise KeyError(
                f"{request.key!r} is not declared in requests; this program declares "
                f"{sorted(each.key for each in self._requests.accepts())}"
            )
        return self._requests.take(request)

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

    def chosen(self, name: str, device_type: type[D]) -> D:
        """The device chosen for ``name``, which ``chooses`` declares as ``device_type``."""
        if name not in self._chosen:
            raise KeyError(
                f"{name!r} is not declared in chooses; this program declares {sorted(self._chosen)}"
            )
        device = self._chosen[name]
        # Narrows for the type checker, and catches a program asking for the wrong type.
        if not isinstance(device, device_type):
            raise TypeError(f"{name!r} is a {type(device).__name__}, not {device_type.__name__}")
        return device

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
