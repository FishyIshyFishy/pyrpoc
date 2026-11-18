"""``RunInbox``: everything posted to one run and not yet applied, each applied at its moment
and logged one way."""

# Things are posted on the GUI thread and applied on the run's own, so the inbox is guarded by
# a lock. Nothing here knows who posted: a panel today, perhaps the run's own logic later.

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from pyrpoc.core.device import Device, encode_config
from pyrpoc.core.program import Program
from pyrpoc.core.program_input import ProgramInput, SidecarProgramInput
from pyrpoc.core.recording import utc_now
from pyrpoc.engine.data_library.format import AppliedEdit, AppliedEditKind
from pyrpoc.engine.data_library.saving import RecordingSaver

from .program_inputs import ProgramInputsStore

T = TypeVar("T")


def device_state(devices: dict[type[Device], Device]) -> dict[str, Any]:
    return {cls.__name__: encode_config(device.config) for cls, device in devices.items()}


class RunInbox:
    """What has been posted to one run: edits to its program inputs, and edits to its devices.
    A live input's edit is applied when the program reads it; a fixed input's, and the devices',
    between passes of a repeating run, where it begins a new part of the recording. A running
    program reads its inputs from here."""

    def __init__(
        self,
        program: Program,
        inputs: ProgramInputsStore,
        devices: dict[type[Device], Device],
        saver: RecordingSaver | None,
    ):
        self.inputs = inputs
        self.devices = devices
        # None for a run that is not saved: there is nothing to log into.
        self.saver = saver
        self.declared = {each.key: each for each in program.inputs}
        self.fixed = frozenset(each.key for each in program.fixed)
        # A sidecar input is kept only in its own file, so only when the program saves it and
        # the run is saved; any other input is kept in the metadata.
        self.save_sidecar: dict[str, Callable[[Any], str]] = (
            {each.key: partial(saver.save_sidecar, each) for each in program.saves}
            if saver is not None
            else {}
        )
        self.kept_keys = frozenset(
            key
            for key, each in self.declared.items()
            if key in self.save_sidecar or not isinstance(each, SidecarProgramInput)
        )
        self.lock = threading.Lock()
        # What the run has been given, and what was posted since, with when.
        self.values = {key: inputs.get(key) for key in self.declared}
        self.posted: dict[str, tuple[str, Any]] = {}
        self.device_settings = device_state(devices)
        self.devices_posted_at: str | None = None
        inputs.subscribe(self.input_set)

    def __getitem__(self, program_input: ProgramInput[T]) -> T:
        """A live input's value now, applying an edit posted since it was last read, or a fixed
        input's value as the part began."""
        key = program_input.key
        if key not in self.declared:
            raise KeyError(
                f"{key!r} is not an input of this program; it declares {sorted(self.declared)}"
            )
        if key not in self.fixed:
            with self.lock:
                posted = self.posted.pop(key, None)
            if posted is not None:
                self.apply_input(key, *posted)
        return self.values[key]

    def input_set(self, key: str) -> None:
        """Post an edit to a program input, on the thread that made it."""
        if key in self.declared:
            with self.lock:
                self.posted[key] = (utc_now(), self.inputs.get(key))

    def devices_edited(self) -> None:
        """Post an edit to the devices, on the thread that made it."""
        with self.lock:
            self.devices_posted_at = utc_now()

    def apply_settings(self) -> bool:
        """Apply every edit posted to the fixed inputs and to the devices, returning whether
        any changed what the run runs with."""
        with self.lock:
            edits = {key: self.posted.pop(key) for key in list(self.posted) if key in self.fixed}
            devices_posted_at, self.devices_posted_at = self.devices_posted_at, None
        changed = [self.apply_input(key, *posted) for key, posted in edits.items()]
        if devices_posted_at is not None:
            changed.append(self.apply_devices(devices_posted_at))
        return any(changed)

    def apply_input(self, key: str, posted_at: str, value: Any) -> bool:
        """Give the run an input's posted value, returning whether it differs from the last."""
        rule = self.declared[key].rule
        if rule.encode(value) == rule.encode(self.values[key]):
            return False
        self.values[key] = value
        self.log(posted_at, "input", key, self.kept(key) if key in self.kept_keys else None)
        return True

    def kept(self, key: str) -> Any:
        """An input's value as the recording keeps it: as JSON, or as the file a saved sidecar
        input was just written to."""
        if key in self.save_sidecar:
            return {"file": self.save_sidecar[key](self.values[key])}
        return self.declared[key].rule.encode(self.values[key])

    def apply_devices(self, posted_at: str) -> bool:
        """Apply the devices' settings as they are now, returning whether any changed."""
        settings = device_state(self.devices)
        changed = [name for name, each in settings.items() if each != self.device_settings[name]]
        for name in changed:
            self.log(posted_at, "device", name, settings[name])
        self.device_settings = settings
        return bool(changed)

    def log(self, posted_at: str, kind: AppliedEditKind, key: str, value: Any) -> None:
        if self.saver is not None:
            self.saver.log_applied(AppliedEdit(posted_at, utc_now(), kind, key, value))

    def parameters(self) -> dict[str, Any]:
        """Every input the recording keeps, as the run has been given it, by key."""
        return {key: self.kept(key) for key in self.declared if key in self.kept_keys}

    def close(self) -> None:
        """Stop taking posts, on the thread that posts them."""
        self.inputs.unsubscribe(self.input_set)
