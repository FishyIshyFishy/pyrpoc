# Phase 1 API survey & reconciliation

**Branch:** `refactor/routines`
**Companion docs:** [`incremental_migration.md`](incremental_migration.md),
[`refactor_routines.md`](refactor_routines.md), [`refactor_questions.md`](refactor_questions.md).

**Status:** survey only. No code changes. This is the "enumerate the surface before
writing the adapters" step flagged as Phase 1's linchpin risk.

## 0. Purpose

Two questions:

1. **What API does the original `pyrpoc` GUI actually depend on?** (So we know exactly
   what the migration must preserve or replace.)
2. **What is the desired GUI-facing API of the new `pyrpoc_next` backend?** (So we know
   what the GUI should be talking to when we're done.)

New permission from this round: **we may change the old GUI** as long as the *visual
layout* is preserved. That reframes Phase 1 — we are no longer obligated to make the new
core wear the old service costume exactly. We can move whichever side is cleaner to move.

## 1. The two sides at a glance

| Concern | Old `pyrpoc` (services) | New `pyrpoc_next` (core) |
| --- | --- | --- |
| Backend entry | 4 `QObject` services (Instrument/Modality/Display/OptoControl) + `AcquisitionInterpreter` + `SessionCoordinator` | 1 plain `Controller` + `AppState` + 4 registries |
| Eventing | many granular Qt signals (`inventory_changed`, `display_added`, `data_emitted`, …) | 3 callback attrs (`on_started/on_stopped/on_error`) + `prepare_modifier` hook |
| Unit of work | one **selected modality** + its params | the **active `RoutineBlock`** (modality + values + modifier slots) |
| Data currency | `AcquiredData(data, kind: DataKind, labels, metadata)`, routed by `kind` | typed **parcels** (`ImageFrameParcel`/`PartialImageParcel`/`HistogramCubeParcel`), routed by `isinstance` |
| Display contract | `BaseDisplay.render(AcquiredData)`, `accepted_kinds` | `DisplaySink`: `manifest` + `render(Parcel)` |
| Parameters | `BaseParameter` w/ `create_widget` (Qt-coupled) in `backend_utils` | Qt-free `Parameter` in `structs`; GUI builds widgets separately |
| Optocontrols | `OptoControlService` + separate tab + `prepare_for_acquisition()` context | **modifiers-as-data** (`ModifierSlot`, `Modifier` dataclass) + `prepare_modifier` hook |
| Threading | acquisition on `threading.Thread`; results back via Qt signals | acquisition on daemon thread; results back via callbacks (GUI marshals with `RunBridge`) |
| Persistence | `SessionCoordinator` + `SessionRepository`, schema v6, autosave, dock layout | **deferred** (question G2) — not built yet |

## 2. What the old GUI needs (the surface to preserve or replace)

Full method/signal/line-level inventory lives in the exploration output; the load-bearing
shape is:

- **InstrumentService** — `list_available`, `list_instances` (rows keyed by a stable
  `state` object), `create_instrument(key)`, `remove_instrument`, `get_widget(instance,
  on_change=…)`, `mark_instance_changed`; signals `inventory_changed`, `instance_changed`.
- **ModalityService** — `list_available`, `select_modality(key)`, `get_selected_parameters`,
  `get_selected_contract`, `configure(params)`, `start(force_continuous=…)`, `stop`,
  `get/set_parameter_values`; signals `modality_selected(str)`, `modality_params_changed`,
  `requirements_changed(ok, missing)`, `acq_started/stopped/error/warning`,
  `data_emitted(AcquiredData)`.
- **DisplayService** — `list_available`, `list_compatible_with(kinds)`, `create_display`,
  `remove_display`, `attach/detach`, `set_dock_visibility`, `list_instances`,
  `get_display_by_id`, `get_widget`, `get_rpoc_input`; signals `display_added/removed/
  changed`, `display_error`.
- **OptoControlService** — `list_available`, `list_instances`, `create_opto_control`,
  `set_enabled`, `remove_opto_control`, `get_widget(control, display_service=…)`,
  `collect_data_for_acquisition`; signals `inventory_changed`, `control_state_changed`,
  `control_changed`, `opto_control_error`.

**Cross-cutting consumers to watch** (not just the manager panels):
- `AcquisitionInterpreter` self-wires `data_emitted → route` — routing lives *outside* the
  GUI in the old design. In the new design routing is `Controller.sink → route_parcel`.
- `SessionCoordinator` listens to nearly every inventory/param signal for autosave.
- The **mask editor** (`optocontrols/mask.py`) is coupled to `DisplayService`: it calls
  `list_instances()` / `get_display_by_id()` and listens to `display_added/removed/changed`,
  and reads `display.get_normalized_data_3d()`. Any display-side migration must keep those.

## 3. The desired new-core GUI-facing API (the target contract)

This is what the migrated GUI should ultimately talk to. It already largely exists in
`pyrpoc_next` — this section just states it as the contract.

**Run lifecycle — `Controller`:**
- `play(continuous: bool) -> CompatibilityReport` — validates, builds+configures the
  active block's modality, starts the run; returns a blocked report instead of running if
  incompatible.
- `stop() -> None`
- `check() -> CompatibilityReport`
- callbacks: `on_started`, `on_stopped`, `on_error(exc)` (worker-thread; GUI marshals via
  `RunBridge`), and the `prepare_modifier(modifier, slot)` hook for attaching runtime data.

**Shared model — `AppState`:** `instruments`, `displays`, `routine`, `run_status`;
`instrument_for(key)`. Panels mutate these lists directly.

**"What's available" — registries:** `instrument_registry`, `modality_registry`,
`modifier_registry`, `display_registry`, each with `available()`, `create(key)`,
`manifest(key)`.

**Data currency — parcels + `DisplaySink`:** displays satisfy `manifest` + `render(Parcel)`
and register themselves into `state.displays`. Routing is `route_parcel`.

**Metadata/compatibility — manifests:** modality manifest declares `emitted_parcels`,
`required_instruments`, `realizable_modifiers`, `parameter_groups`.

**Concept mapping the GUI must adopt:**
- "selected modality + params" ⇒ a `Routine` with a single `RoutineBlock` whose `modality`
  and `values` you edit. `select_modality(key)` ⇒ set that block's modality.
- `AcquiredData.kind` ⇒ concrete parcel type: `INTENSITY_FRAME→ImageFrameParcel`,
  `PARTIAL_FRAME→PartialImageParcel`, `FLIM_*→HistogramCubeParcel`.
- optocontrols ⇒ modifier slots on the block; the mask array is attached via
  `prepare_modifier`.

## 4. Gaps the new core must grow before the GUI can fully land on it

These are places the new backend is currently thinner than the old GUI assumes. None are
blockers for the survey; they're the concrete backlog for Phases 2–4.

1. **Instrument configuration parameters.** New `Instrument` base exposes only
   `key/display_name/connect/test_connection/summary` — no `parameter_groups`. The old
   instrument cards expand into a parameter editor. New instruments need declared params +
   a GUI editor built from them (same pattern as modalities). *(Phase 2)*
2. **Display config + capabilities.** New `DisplayWidget`/`DisplayManifest` have no
   `parameter_groups` and no `configure()`; old displays carry `display_parameters`
   (LUT/colormap/channel) and `configure(params)`. Also missing: `get_normalized_data_3d()`
   and `export_rpoc_input() -> RPOCImageInput`, which the **mask editor** needs. *(Phase 3/4)*
3. **Per-instance naming.** Old instances carry `user_label` and the cards let you rename;
   new instances don't. Needed for card parity. *(Phase 2/3)*
4. **Optocontrol coverage.** Only `mask` exists as a modifier today. Enumerate the old
   optocontrols — anything beyond mask must map to a modifier (or be consciously dropped).
   The old lifecycle (`enabled` toggle, `connected`, `cleanup`, `get_context`) is richer
   than an inert dataclass; confirm masks are the only stateful one. *(Phase 4)*
5. **Live instrument gating.** Old GUI shows missing instruments live via
   `requirements_changed(ok, missing)`. New equivalent is calling `check()` and reading the
   `CompatibilityReport`. Decide whether to re-check on inventory change (to keep the live
   feel) or move to on-play validation only. *(Phase 2)*
6. **Load-bearing param labels.** `Controller.play` reads the literal labels `"Save"`,
   `"Save Path"`, `"Frames"` off the active block. Whatever supplies the acquisition
   controls (global vs per-block) must produce params with exactly those labels. *(Phase 1/4)*
7. **Persistence/menubar.** Session persistence is deferred (G2), but the menubar's
   new/open/save actions are wired to `SessionCoordinator` today. Phase 1 needs those to be
   safe no-ops (or a minimal stub) so the menu doesn't crash.

## 5. The one strategic fork (for Phase 1)

Given "GUI changes are allowed if visuals are preserved," how does the old GUI reach the
new core?

- **A — Service shims.** Reimplement the four services as `QObject` adapters exposing the
  *exact* old signals/methods over `Controller`+`AppState`. Old GUI (managers, handlers,
  `SessionCoordinator`, mask widget) is untouched. Max fidelity, but we build a large
  translation layer — carrying the old concept model (modality-select, `AcquiredData`, opto
  contexts) — that gets deleted again by Phase 4.
- **B — Handler rewire (recommended).** Keep the widgets, theme, cards, docking untouched;
  edit the thin **`handlers.py`** in each manager to call `Controller`/`AppState`/registries
  directly. Replace granular service signals with either a small local signal bus or direct
  refresh calls. Fewer throwaway artifacts; GUI ends closer to the target. Touches GUI code
  now — which is now permitted.
- **C — Hybrid.** Rewire the straightforward managers (instruments/displays), shim only the
  deeply cross-cutting couplings (the mask widget's `DisplayService` dependency; autosave —
  which we can also just disable while persistence is deferred).

**Recommendation: B, sliding to C only where coupling is deep.** The pure-shim (A) was
justified mainly by "don't touch the GUI"; that constraint is now lifted, and A leaves the
GUI further from the destination. The widgets — the part that regressed last time — stay
untouched under B/C; only the handler glue moves.

## 6. Gotchas to carry forward

- Old acquisition runs on a background thread and marshals results **only** via signals;
  the new design marshals via `RunBridge`. Keep all widget mutation on the GUI thread.
- Manager cards key their maps by the identity of the backend `state` object. New instances
  are stable identities, so this convention survives — don't hand back fresh wrapper objects
  per `list_instances` call.
- `OptoControlService.get_widget(display_service=…)` injects the display service into control
  widgets; the mask editor uses it. Whatever replaces `DisplayService` must still give the
  mask editor a way to enumerate displays and pull their normalized 3D data.
