"""``RunInbox``: everything posted to one run and not yet taken, each taken at its moment and
logged one way."""

# Things are posted on the GUI thread and taken on the run's own, so the inbox is guarded by a
# lock. Nothing here knows who posted: a panel today, perhaps the run's own logic later.

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from pyrpoc.core.device import Device, encode_config
from pyrpoc.core.program import Program
from pyrpoc.core.program_input import ProgramInputValues
from pyrpoc.core.recording import utc_now
from pyrpoc.core.requests import Request
from pyrpoc.engine.data_library.format import Taken

from .program_inputs import ProgramInputsStore


def device_state(devices: dict[type[Device], Device]) -> dict[str, Any]:
    return {cls.__name__: encode_config(device.config) for cls, device in devices.items()}


class RunInbox:
    """What has been posted to one run: requests, edits to its program inputs, and edits to its
    devices. Requests are taken when the program asks for them, a live input's edit when the
    program reads it, and every other edit between passes of a repeating run, where it begins
    a new part of the recording."""

    def __init__(
        self,
        program: Program,
        inputs: ProgramInputsStore,
        devices: dict[type[Device], Device],
        on_taken: Callable[[Taken], None] | None,
    ):
        self.inputs = inputs
        self.devices = devices
        # None for a run that is not saved: there is no recording to log into.
        self.on_taken = on_taken
        self.rules = {each.key: each.rule for each in program.inputs}
        self.live = frozenset(each.key for each in program.live)
        self.lock = threading.Lock()
        # What the run has been given, and what was posted since, with when.
        self.values = {key: inputs.get(key) for key in self.rules}
        self.posted: dict[str, tuple[str, Any]] = {}
        self.device_settings = device_state(devices)
        self.devices_posted_at: str | None = None
        self.requests: dict[Request[Any], list[tuple[str, Any]]] = {
            each: [] for each in program.requests
        }
        inputs.subscribe(self.input_set)

    def input_set(self, key: str) -> None:
        """Post an edit to a program input, on the thread that made it."""
        if key in self.rules:
            with self.lock:
                self.posted[key] = (utc_now(), self.inputs.get(key))

    def devices_edited(self) -> None:
        """Post an edit to the devices, on the thread that made it."""
        with self.lock:
            self.devices_posted_at = utc_now()

    def give(self, request: Request[Any], value: Any) -> None:
        """Post a request."""
        with self.lock:
            self.requests[request].append((utc_now(), value))

    def accepts(self) -> frozenset[Request[Any]]:
        return frozenset(self.requests)

    def live_keys(self) -> frozenset[str]:
        return self.live

    def take(self, request: Request[Any]) -> list[Any]:
        """The values posted with ``request`` since it was last taken, oldest first."""
        with self.lock:
            taken, self.requests[request] = self.requests[request], []
        for posted_at, value in taken:
            self.log(posted_at, "request", request.key, request.rule.encode(value))
        return [value for _, value in taken]

    def current(self, key: str) -> Any:
        """A live input's value, taking an edit posted since it was last read."""
        with self.lock:
            posted = self.posted.pop(key, None)
        if posted is not None:
            self.take_input(key, *posted)
        return self.values[key]

    def take_settings(self) -> bool:
        """Take every edit posted to the inputs that are not live and to the devices, returning
        whether any changed what the run runs with."""
        with self.lock:
            edits = {key: self.posted.pop(key) for key in list(self.posted) if key not in self.live}
            devices_posted_at, self.devices_posted_at = self.devices_posted_at, None
        changed = [self.take_input(key, *posted) for key, posted in edits.items()]
        if devices_posted_at is not None:
            changed.append(self.take_devices(devices_posted_at))
        return any(changed)

    def take_input(self, key: str, posted_at: str, value: Any) -> bool:
        """Give the run an input's posted value, returning whether it differs from the last."""
        rule = self.rules[key]
        encoded = rule.encode(value)
        if encoded == rule.encode(self.values[key]):
            return False
        self.values[key] = value
        self.log(posted_at, "input", key, encoded)
        return True

    def take_devices(self, posted_at: str) -> bool:
        """Take the devices' settings as they are now, returning whether any changed."""
        settings = device_state(self.devices)
        changed = [name for name, each in settings.items() if each != self.device_settings[name]]
        for name in changed:
            self.log(posted_at, "device", name, settings[name])
        self.device_settings = settings
        return bool(changed)

    def log(self, posted_at: str, kind: str, key: str, value: Any) -> None:
        if self.on_taken is not None:
            self.on_taken(Taken(posted_at, utc_now(), kind, key, value))

    def given(self) -> ProgramInputValues:
        """Every program input's value as the run has been given it."""
        return ProgramInputValues(dict(self.values))

    def parameters(self) -> dict[str, Any]:
        """Every program input's value as the run has been given it, as JSON, by key."""
        return {key: rule.encode(self.values[key]) for key, rule in self.rules.items()}

    def close(self) -> None:
        """Stop taking posts, on the thread that posts them."""
        self.inputs.unsubscribe(self.input_set)
