from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import queue
from typing import Any

from pyrpoc.acquisition.hardware.geometry import pixel_to_voltage
from pyrpoc.structs.acquired_data import AcquiredData
from pyrpoc.structs.commands import Command, PointDwellCommand


class CommandSource:
    """Yields commands one at a time.

    ``next_command`` returns the next command, or None when finished.
    ``observe_result`` lets a source (or decorator) see what a command produced
    — the hook that makes closed-loop behaviour possible.
    """

    def next_command(self) -> Command | None:
        raise NotImplementedError

    def observe_result(self, command: Command, results: list[AcquiredData]) -> None:
        return None


class SourceDecorator(CommandSource):
    """A source that wraps another source, satisfying the same contract.

    Override ``transform`` to modify each outgoing command; ``observe_result``
    fans down the wrapped stack so closed-loop decorators anywhere see results.
    """

    def __init__(self, inner: CommandSource):
        self.inner = inner

    def next_command(self) -> Command | None:
        command = self.inner.next_command()
        return self.transform(command) if command is not None else None

    def transform(self, command: Command) -> Command:
        return command

    def observe_result(self, command: Command, results: list[AcquiredData]) -> None:
        self.inner.observe_result(command, results)


class FiniteScanSource(CommandSource):
    """Yields one command per frame from a factory, up to ``frame_limit``.

    ``frame_limit=None`` runs continuously (until the executor's should_stop).
    """

    def __init__(self, command_factory: Callable[[int], Command], frame_limit: int | None):
        self.command_factory = command_factory
        self.frame_limit = frame_limit
        self.frame_index = 0

    def next_command(self) -> Command | None:
        if self.frame_limit is not None and self.frame_index >= self.frame_limit:
            return None
        command = self.command_factory(self.frame_index)
        self.frame_index += 1
        return command


@dataclass
class ClickPoint:
    px: float
    py: float


class PointClickSource(CommandSource):
    """Click-to-acquire source (stub).

    A GUI-thread producer pushes ClickPoints onto ``click_queue``; this source,
    on the worker thread, blocks on that queue and converts each click's pixel
    to galvo volts via the scan geometry. Pushing None ends the source.
    """

    def __init__(self, *, geometry: dict[str, Any], command_fields: dict[str, Any], poll_timeout_s: float = 0.2):
        self.click_queue: "queue.Queue[ClickPoint | None]" = queue.Queue()
        self.geometry = geometry
        self.command_fields = command_fields
        self.poll_timeout_s = poll_timeout_s
        self.frame_index = 0

    def submit_click(self, px: float, py: float) -> None:
        self.click_queue.put(ClickPoint(px=px, py=py))

    def finish(self) -> None:
        self.click_queue.put(None)

    def next_command(self) -> Command | None:
        while True:
            try:
                point = self.click_queue.get(timeout=self.poll_timeout_s)
            except queue.Empty:
                continue
            if point is None:
                return None
            vx, vy = pixel_to_voltage(point.px, point.py, **self.geometry)
            command = PointDwellCommand(
                x_volts=vx,
                y_volts=vy,
                frame_index=self.frame_index,
                **self.command_fields,
            )
            self.frame_index += 1
            return command
