"""The orchestration hub. Qt-free.

Turns the current routine into a configured modality, checks compatibility, and runs
it — routing emitted parcels to displays and storage. The gui drives it through this
plain API and observes status via the on_* callbacks.
"""

from __future__ import annotations

from collections.abc import Callable

from pyrpoc_next.acquisition import Runner, modality_registry, modifier_registry
from pyrpoc_next.core.app_state import AppState
from pyrpoc_next.core.compatibility import check_routine
from pyrpoc_next.core.router import route_parcel
from pyrpoc_next.core.storage import FrameStorage
from pyrpoc_next.structs.parameters import ParameterValue, coerce_parameter_values
from pyrpoc_next.structs.parcels import Parcel
from pyrpoc_next.structs.status import CompatibilityReport, RunStatus


def values_map(values: list[ParameterValue]) -> dict:
    return {value.label: value.value for value in values}


class Controller:
    """Coordinates instruments, acquisition, displays, and storage for one app."""

    def __init__(self, state: AppState | None = None):
        self.state = state or AppState()
        self.runner = Runner()
        self.storage = FrameStorage()
        self.on_started: Callable[[], None] | None = None
        self.on_stopped: Callable[[], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None
        self.prepare_modifier: Callable[[object, object], None] | None = None

    def check(self) -> CompatibilityReport:
        return check_routine(self.state)

    def play(self) -> CompatibilityReport:
        """Validate and, if clear, start the active block. Returns the report either way."""
        report = self.check()
        if report.blocked:
            return report
        modality, values = self.build_modality()
        if values["Save"]:
            self.storage.begin(values["Save Path"])
        self.state.run_status = RunStatus.running
        if self.on_started:
            self.on_started()
        self.runner.start(modality, self.sink, frame_limit=values["Frames"], on_finished=self.finished)
        return report

    def stop(self) -> None:
        self.state.run_status = RunStatus.stopping
        self.runner.stop()

    def sink(self, parcel: Parcel) -> None:
        route_parcel(parcel, self.state.displays)
        self.storage.save(parcel)

    def finished(self, error: Exception | None) -> None:
        self.storage.finish()
        self.state.run_status = RunStatus.error if error else RunStatus.idle
        if error and self.on_error:
            self.on_error(error)
        if self.on_stopped:
            self.on_stopped()

    def build_modality(self):
        """Build and configure the modality for the routine's active block."""
        block = self.state.routine.active_block
        modality = modality_registry.create(block.modality)
        values = coerce_parameter_values(modality.manifest.parameter_groups, values_map(block.values))
        instruments = {key: self.state.instrument_for(key) for key in modality.manifest.required_instruments}
        modifiers = [self.build_modifier(slot) for slot in block.enabled_modifiers()]
        modality.configure(values, instruments, modifiers)
        return modality, values

    def build_modifier(self, slot):
        """Build a modifier dataclass from a slot; let the gui attach runtime data (e.g. a mask)."""
        groups = modifier_registry.manifest(slot.key).parameter_groups
        modifier = modifier_registry.build(slot.key, coerce_parameter_values(groups, values_map(slot.values)))
        if self.prepare_modifier:
            self.prepare_modifier(modifier, slot)
        return modifier
