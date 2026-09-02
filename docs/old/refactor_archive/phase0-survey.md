# Phase 0 survey — what pyrpoc v3.0.2 actually does

Written during phase 0 as the "record what the current code does" half of the
task. Describes the tree **as it stands after the phase 0 removals**, and lists
what was taken out. Companion to `260827-refactor_plan.md` (the destination)
and `260829-implementation_plan.md` (the route).

## 1. How a run happens today

Pressing play walks this path:

```
AcquisitionManagerWidget (Qt form)
  -> handlers.configure_modality
       collect_values(param_widgets)          # the form IS the parameter store
  -> ModalityService.configure(raw_params)
       coerce_parameter_values(...)           # validate/coerce against parameter_groups
       bind required/optional instruments by class
       control.prepare_for_acquisition()      # each enabled, allowed optocontrol
       modality.configure(params, instruments, opto_controls)
  -> ModalityService.start()
       instance.prepare_acquisition_storage(frame_limit)
       instance.acquire_continuous(...)       # spawns the daemon thread
           while not should_stop():
               acquire_once(on_data=handle_acquired)
  -> ModalityService.handle_acquired(AcquiredData)
       if kind.is_persistent: save_acquired_frame(...)
       data_emitted.emit(acquired)
  -> AcquisitionInterpreter.route
       for display in app_state.displays:
           if display.attached and display.docked_visible
              and acquired.kind in display.accepted_kinds:
                  display.render(acquired)
```

Two facts about this worth carrying forward:

- **The loop belongs to `BaseModality`**, not to the experiment. `acquire_once()`
  must be self-contained, which is why `FlimModality` creates and frees the
  TimeTagger once per *frame*.
- **Routing is by `DataKind` tag**, matched against each display's
  `accepted_kinds`, and additionally filtered by the modality's hardcoded
  `allowed_displays` list when the display dropdown is built.

## 2. What exists, by folder

### `modalities/` — three experiments over one base class

| | confocal | split_confocal | flim |
|---|---|---|---|
| required instruments | none | none | `TimeTaggerInstrument` |
| emits | `INTENSITY_FRAME` | `INTENSITY_FRAME` (+ raw stream, smuggled) | `INTENSITY_FRAME`, `FLIM_RAW_FRAME` |
| output shape | `(C, H, W)` | `(C*2, H, W)` alternating t0/t2 | `(1,H,W)` + `(H,W,bins)` |
| channel labels | `ai{n}` | `ai{n}_t0`, `ai{n}_t2` | `intensity` |

`BaseModality` carries the loop (`acquire_continuous` /
`build_continuous_worker`), the frame limit, the storage hooks
(`prepare_acquisition_storage` / `save_acquired_frame` /
`finalize_acquisition_storage`), channel-splitting helpers, and the class-level
declaration block (`parameter_groups`, `required_instruments`,
`allowed_optocontrols`, `emitted_kinds`, `allowed_displays`). It is the class
the refactor exists to dissolve.

**Split confocal's second output has no home.** `acquire_once` stashes the raw
pixel stream in `self._pending_auxiliary["raw_pixel_stream"]`; `storage.py`
picks it up via `append_auxiliary_payload`, buffers every frame in memory, and
writes one `*_raw_pixel_stream.npz` at the end. No display ever sees it. This
is the concrete case the `emits = {...}` design fixes.

**FLIM's histogram cube takes a similar side path**: `_pending_flim_frame` is
read by `flim/storage.py::save_acquired_frame` and accumulated into
`_raw_frames`, then written as one `*_raw.npz` with `dtype=object`.

### `modalities/*/acquisition_core.py` — the arithmetic worth preserving

The healthiest code in the repo, and what phase 1 moves to `operations/`:

- `helpers/daq.py::generate_raster_waveform` — the only genuinely shared piece.
- `resize_mask_nearest`, `preprocess_mask_to_scan_grid`, `extract_kept_samples`
  — **duplicated verbatim** between `confocal/` and `split_confocal/`.
- `generate_mask_ttl_signals` — near-duplicate; split confocal's adds one
  `ttl[:, :, t0_samples:] = False` gating line.
- `run_daq` — **duplicated verbatim**; sets up synchronised AO + AI + DO on the
  AO sample clock. Does one blocking
  `ai_task.read(number_of_samples_per_channel=total_samples)`, which is why
  nothing can emit partial frames today.
- `reshape_to_frame` (mean over pixel samples) vs `reshape_to_split_frame`
  (mean over `[:t0]` and `[t0+t1:]`, returning the raw cube alongside).
- `flim/acquisition_core.py` — `run_flim_scan` drives AO plus a counter-derived
  pixel clock and an exported start trigger; `reshape_flim_frame` /
  `flim_intensity` fold the histogram.

DO channels are split by name: `/port0/` lines get a clocked `do_task`, anything
else is written once as a static level and inverted on teardown.

### `instruments/` — one real device

Only `TimeTaggerInstrument` is registered. Its driver
(`instruments/time_tagger.py`) and its panel
(`instruments/instrument_widgets/time_tagger_widget.py`) sit in different
folders — the split the new `devices/<name>/` layout closes.

**The galvo does not exist as an object.** `fast_axis_ao`, `slow_axis_ao` and
`active_ai_channels` are loose parameters redeclared in all three
`parameters.py` files.

### `displays/` — two remaining views

`tiled_2d` (one tile per channel, per-tile LUT/autoscale/name) and
`multichan_overlay` (colour-mapped composite). Both accept only
`INTENSITY_FRAME` now.

**Displays own the acquired arrays.** `self._data_chw` in each display *is* the
data, not a cache — closing a display destroys it, and two displays over one run
hold two drifting copies. This is what `data/dataset.py` fixes in phase 3.

### `optocontrols/` — masks as global toggleable objects

`MaskOptoControl` holds `daq_port`, `daq_line`, `mask_path`, `mask_data`, and an
`enabled` flag living outside the modality. `prepare_for_acquisition()` freezes
it into a `MaskContext`, which `extract_mask_contexts()` pulls out at configure
time. Because `enabled` is not a run parameter, **saved metadata does not record
which masks were actually applied.**

The live mask editor is `gui/main_widgets/opto_control_mgr/mask_editor.py`
(threshold + polygon ROIs over a channel composite), mounted inside
`MaskOptoControlWidget`. It reads `display.get_normalized_data_3d()`.

### `services/`, `domain/`, `persistence/`, `gui/`

`AppController` builds five services over one `AppState`. `SessionCoordinator`
autosaves on every inventory/parameter signal; `SessionCodec` round-trips to
JSON at `schema_version = 6`. `gui/main_gui.py` is a PyQt6Ads dock manager with
four fixed panels plus one dock per display.

## 3. What phase 0 removed

| Removed | Why |
|---|---|
| simulated fallbacks in all 3 modalities | acquisition must fail hard |
| `modalities/helpers/toy_data.py` | the only fallback data source |
| `BaseModality.emit_warning`, `acq_warning` signal | existed solely to announce the fallback |
| `pyrpoc/rpoc/` (entire package) | deprecated; `editor.py` and `local_treatment.py` had zero importers |
| `displays/streamed_image_display.py` | streaming, deferred |
| `displays/flim_display.py` | streaming (progressive row-by-row fit), deferred |
| `DataKind.PARTIAL_FRAME`, `FLIM_PARTIAL_HISTOGRAM` | no emitter and no consumer |
| `export_rpoc_input` / `get_rpoc_input` | dead path; `RPOCImageInput` lived in `rpoc/` |
| `backend_utils/array_contracts.py` | referenced only by its own test |
| `instruments/prior_stage.py`, `zaber_stage.py` | empty / two stray imports |
| deps: `cellpose`, `matplotlib`, `pyvisa`, `pillow`, `superqt` | no importer remained |

## 4. Reference arrays

`tests/reference/` records what the surviving arithmetic computes.
`generate_references.py` writes `phase0_references.npz`;
`test_phase0_references.py` compares the live functions against it and is
verified to fail on an injected change.

Phases 1–4 of the implementation plan each say "identical to the phase 0
reference" — this is that reference. **Point these tests at the new
`operations/` functions rather than regenerating the file.**

Covered: `generate_raster_waveform`, both `extract_kept_samples`,
`reshape_to_frame`, `reshape_to_split_frame`, `resize_mask_nearest`,
`preprocess_mask_to_scan_grid`, both `generate_mask_ttl_signals`,
`reshape_flim_frame`, `flim_intensity`.

Not covered, and not coverable on a laptop: whether AI stays aligned to the
sample clock, whether DO pulses land on the right pixel, whether the tagger sees
the frame trigger. That is what the microscope check between phases 6 and 7 is
for.

## 5. Open items found during the survey

1. **Three separate `DaqUnavailableError` classes** — one per
   `acquisition_core.py`, so `except DaqUnavailableError` imported from one
   module does not catch the other two. Should become one error in `core/errors.py`.
2. **`domain/stores.py` is dead** — referenced only by its own test. The refactor
   plan already marks it for deletion; left in place because that is phase 9.
3. **`pyqtdarktheme` is an unused dependency** — theming actually runs through the
   vendored `gui/styles/breeze_all.py`. Left in pyproject; removing it is a
   one-line change if the Breeze theming is staying.
4. **`num_frames` is validated in two places** — `get_frame_limit()` raises below 1,
   and the parameter schema sets `minimum=1`.
5. **`split_confocal` previously swallowed bare `RuntimeError`** into simulated
   data, so unrelated bugs surfaced as toy frames. Gone with the fallback, noted
   because it may have been masking real failures.
