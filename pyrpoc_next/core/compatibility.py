"""Compatibility check, run before play.

Reads hand-declared manifests and reports inconsistencies that would make a run
confusing or impossible — a missing instrument, no display that can show the emitted
data, a modifier the modality can't apply. Halting issues block the run; the gui
shows them in a dialog the user must clear.
"""

from __future__ import annotations

from pyrpoc_next.acquisition import modality_registry
from pyrpoc_next.core.app_state import AppState
from pyrpoc_next.structs.status import CompatibilityReport, IssueSeverity


def check_routine(state: AppState) -> CompatibilityReport:
    """Validate the routine's active block against the current inventory."""
    report = CompatibilityReport()
    block = state.routine.active_block
    if block is None:
        report.add(IssueSeverity.halt, "The routine has no active block to run.")
        return report

    manifest = modality_registry.manifest(block.modality)

    present = {instrument.key for instrument in state.instruments}
    for key in manifest.required_instruments:
        if key not in present:
            report.add(IssueSeverity.halt,
                       f"{manifest.display_name} needs the '{key.value}' instrument, which has not been added.")

    if not state.displays:
        report.add(IssueSeverity.halt, "No displays are open to show the acquired data.")
    else:
        for parcel_type in manifest.emitted_parcels:
            if not any(issubclass(parcel_type, display.manifest.accepted_parcels) for display in state.displays):
                report.add(IssueSeverity.halt,
                           f"No open display can show {parcel_type.__name__} produced by {manifest.display_name}.")

    for slot in block.enabled_modifiers():
        if slot.key not in manifest.realizable_modifiers:
            report.add(IssueSeverity.halt,
                       f"{manifest.display_name} cannot apply the '{slot.key.value}' modifier.")

    return report
