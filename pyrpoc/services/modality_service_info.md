# Acquisition Pipeline — Design Reference

This document describes the full acquisition pipeline from button click to frame display, covering every layer: instruments, optocontrols, modalities, parameters, displays. It also records deliberate design decisions so they are not accidentally reversed.

---

## Layers at a Glance

```
User click
    ↓
GUI handler (handlers.py)
    ↓
ModalityService.configure() + .start()
    ├── InstrumentService  → bound instruments
    ├── OptoControlService → bound optocontrol contexts
    └── BaseModality.configure(params, instruments, opto_controls)
            ↓
        BaseModality.run_acquisition_threaded()
            → loop: acquire_once() → on_frame()
                                        ├── BaseModality.save_acquired_frame()
                                        └── ModalityService.data_ready signal
                                                ↓
                                        DisplayService.push_data()
                                                ↓
                                        BaseDisplay.render()
```

---

## Phase 1: Pre-acquisition (configure)

### 1.1 Parameter collection

Parameters are defined on the modality class as `PARAMETERS: ParameterGroups` — a `dict[str, list[BaseParameter]]`. The outer keys are group names (used only for UI layout; they have no semantic meaning at runtime). Each `BaseParameter` carries a label, type, default, and a `coerce()` method.

When the user clicks Start:

1. `collect_values(param_widgets)` reads current widget values into `dict[label, raw_value]`.
2. `ModalityService.configure(raw_params)` calls `coerce_parameter_values()`, which validates all labels and type-converts each value. The result is `dict[label, coerced_value]`.
3. The coerced dict is passed into `modality.configure()` and stored in `app_state.modality.configured_params` as a list of `ParameterValue` objects. This is the **single authoritative record** of what parameters were used for the current acquisition.

**Design rule — parameters are snapshotted, not re-read.** `coerce_parameter_values` runs once at configure time. The modality copies the result into `self._params = dict(params)` and reads from that snapshot for the rest of the acquisition. Widget state after that point is irrelevant.

### 1.2 Instrument binding

`REQUIRED_INSTRUMENTS` and `OPTIONAL_INSTRUMENTS` are class-level lists of `BaseInstrument` subclasses. During `configure()`, the service calls `instrument_service.get_instances_by_class(cls)` for each and builds a `bound: dict[type[BaseInstrument], BaseInstrument]` dict, then passes it to `modality.configure()`.

**Design rule — instruments are passed as live objects, not snapshots.** The modality stores the reference (`self._daq_instrument = instruments[ConfocalDAQInstrument]`) and reads hardware-configuration fields from it at acquisition time (e.g. `ai_channel_numbers`, `active_ai_channels`, `device_name`, `sample_rate_hz`). This is deliberate:

- Hardware configuration lives on the instrument, not in modality parameters.
  A DAQ channel list is an instrument property, not a scan setting.
- The instrument object is already the canonical runtime representation of the hardware.
  Snapshotting into parameters would create a second source of truth that could drift.
- The instrument is configured (and potentially connected) before `configure()` is called.
  Its state is stable for the lifetime of the acquisition.

**When you implement a new instrument:** put all hardware-configuration fields (device names, channel numbers, connection handles, port addresses) on the instrument object. Do not surface them as modality parameters. The modality receives the whole object and reads what it needs.

**Multi-instance caveat:** the service currently picks `instances[0]` when multiple instruments of the same type exist. This is a known limitation, not an oversight. When multi-instrument support is needed the binding logic in `ModalityService.configure()` will need an explicit selection mechanism.

### 1.3 Optocontrol binding

`ALLOWED_OPTOCONTROLS` lists the `BaseOptoControl` subclasses that the modality can use. During `configure()`, the service iterates `app_state.optocontrols`, filters to those that are `enabled` and `isinstance(control, allowed_types)`, calls `control.prepare_for_acquisition()` on each, and passes the resulting list to `modality.configure()`.

`prepare_for_acquisition()` calls `self.get_context()` internally and returns a **frozen context object** (`BaseOptoControlContext` subclass, e.g. `MaskContext`). The modality receives this context, not the live optocontrol object.

**Design rule — optocontrols are snapshotted via context objects.** Unlike instruments (which need a live reference to drive hardware), optocontrols contribute a configuration payload: a mask array, a DAQ port/line, a waveform, etc. These are captured once at configure time. The modality reads from the frozen context during acquisition and does not call back into the optocontrol object. This keeps the acquisition loop free of side-effects from optocontrol UI changes.

**When you implement a new optocontrol:** override `get_context()` to return a frozen dataclass containing everything the modality will need. Do not retain a live reference to the optocontrol in the modality.

---

## Phase 2: Acquisition loop

### 2.1 Starting

`ModalityService.start(*, force_continuous=False)` calls `_prepare_acquisition_start()`, which:

1. Checks `instance` and `selected_class` are non-None.
2. Checks `configured_params` is non-empty (i.e. `configure()` was called).
3. Checks `running` is False.
4. Re-validates required instruments (they could have been disconnected since configure).
5. Calls `instance.get_frame_limit()` unless `force_continuous=True`, in which case `frame_limit=None`.

`get_frame_limit() -> int | None` is abstract. Return an `int` if the modality acquires a fixed number of frames (read it from `self._params["num_frames"]`). Return `None` if the modality is inherently continuous and should run until Stop is pressed.

Then:

```
instance.prepare_acquisition_storage(frame_limit=frame_limit)
→ instance.run_acquisition_threaded(on_frame, frame_limit, should_stop, on_error, on_finished)
```

### 2.2 Thread loop

`run_acquisition_threaded()` delegates to `_build_worker()`, which returns the callable that runs in a daemon thread:

```
self.start()
while not should_stop():
    frame = self.acquire_once()   # → numpy array
    on_frame(frame)               # → ModalityService.handle_frame()
    if frame_limit is not None and acquired >= frame_limit:
        break
self.stop()
on_finished(acquired, error)
```

**`acquire_once()`** is abstract. It must return a single `np.ndarray` matching the modality's `OUTPUT_DATA_CONTRACT`. It should be a pure hardware-read: no file I/O, no Qt calls, no side effects.

**Override point:** if the acquisition loop is not a simple poll (e.g. a callback-based SDK, an event queue), override `_build_worker()` and return a different worker callable. The public API (`run_acquisition_threaded`) does not change.

### 2.3 Per-frame handling

`ModalityService.handle_frame(data)`:

1. Type-checks `data` is `np.ndarray`.
2. Validates shape/dtype against `OUTPUT_DATA_CONTRACT` (e.g. `chw_float32` = 3D, float32).
3. Calls `instance.save_acquired_frame(data, frame_index=N)`.
4. Emits `data_ready(data)` → `DisplayService.push_data()`.

**`save_acquired_frame()`** defaults to a no-op in `BaseModality`. Override it if the modality writes frames to disk or to a buffer. It is called on the acquisition thread — keep I/O fast or offload it.

### 2.4 Finishing

When the loop exits, `on_finished(frame_count, error)` calls `ModalityService.handle_acquisition_finished()`:

1. Sets `app_state.modality.running = False`.
2. Calls `instance.finalize_acquisition_storage(frame_count, frame_limit, error)`.
3. Emits `acq_stopped`.

**`finalize_acquisition_storage()`** defaults to a no-op. Override it to flush TIFF buffers, write metadata JSON, close file handles, etc.

---

## Phase 3: Display

`DisplayService.push_data(data)` iterates `app_state.displays`. For each display that is `attached=True` and `docked_visible=True`, it checks whether `data`'s contract is in `display.ACCEPTED_DATA_CONTRACTS`, then calls `display.render(data)`.

Displays are **completely decoupled from modalities.** They see only a numpy array and a contract string. A display does not know which modality produced the data.

`ALLOWED_DISPLAYS` on a modality is a list of display registry keys (strings). It is informational — used by the UI to suggest compatible displays when the user adds one. It is not enforced at runtime.

**When you implement a new display:** set `ACCEPTED_DATA_CONTRACTS` to the contracts you handle. Implement `render(data)`. That is all. No wiring changes needed elsewhere.

---

## Storage layer

Three methods on `BaseModality` form the storage lifecycle:

| Method | When called | Default |
|--------|-------------|---------|
| `prepare_acquisition_storage(frame_limit)` | Before thread starts | no-op |
| `save_acquired_frame(frame, frame_index)` | Each frame, on acq thread | no-op |
| `finalize_acquisition_storage(frame_count, frame_limit, error)` | After thread ends | no-op |

If a modality does not save data, inherit the defaults. If it does save, override all three. The confocal modalities override all three to write per-channel TIFFs and a metadata JSON.

Both current saving modalities have nearly identical storage code. When a third saving modality is added, extract the shared logic into a `TiffFrameSaveMixin`.

---

## Instrument-specific parameters vs. modality parameters

The boundary rule:

| Belongs in | Examples |
|------------|---------|
| **Instrument object** | DAQ device name, sample rate, channel numbers, port/address, connection handle, baud rate |
| **Modality PARAMETERS** | Scan pixels, dwell time, number of frames, save path, save enabled toggle, scan amplitude |

A parameter goes in the modality if it changes scan-to-scan or is set by the user before each acquisition. It goes on the instrument if it describes the hardware setup that persists across sessions and is managed via the instrument config panel.

In confocal: `x_pixels`, `y_pixels`, `dwell_time_us`, `num_frames`, `save_path` are modality parameters. `device_name`, `ai_channel_numbers`, `sample_rate_hz` are instrument fields. The modality reads the latter directly from the instrument object during `configure()` and caches what it needs for the acquisition run.

---

## What each layer owns

| Layer | Owns | Does not own |
|-------|------|--------------|
| `BaseInstrument` | Hardware config, connection handle, connect/disconnect | Scan logic, parameter forms, data arrays |
| `BaseOptoControl` | Optical control config, produces frozen context at configure time | Acquisition loop, data |
| `BaseModality` | Acquisition loop, data shape contract, storage lifecycle | UI layout, instrument config, display rendering |
| `ModalityService` | Orchestration, thread lifecycle, signal emission | Hardware details, file I/O |
| `DisplayService` | Routing numpy arrays to displays, contract checking | Acquisition, modality selection |
| `BaseDisplay` | Visual rendering | Acquisition, storage |

---

## Adding a new modality — checklist

1. Create `pyrpoc/modalities/my_modality.py`
2. Decorate with `@modality_registry.register("my_key")`
3. Set class attributes: `MODALITY_KEY`, `DISPLAY_NAME`, `PARAMETERS`, `REQUIRED_INSTRUMENTS`, `OPTIONAL_INSTRUMENTS`, `ALLOWED_OPTOCONTROLS`, `OUTPUT_DATA_CONTRACT`, `ALLOWED_DISPLAYS`
4. Implement abstract methods: `configure()`, `start()`, `acquire_once()`, `stop()`, `get_frame_limit()`
5. Optionally override `prepare_acquisition_storage`, `save_acquired_frame`, `finalize_acquisition_storage` if the modality saves data
6. Optionally override `_build_worker` if the acquisition loop is event-driven or callback-based

No changes to `ModalityService`, `handlers.py`, `BaseModality`, or any other file.

## Adding a new instrument — checklist

1. Create `pyrpoc/instruments/my_instrument.py`
2. Decorate with `@instrument_registry.register("my_key")`
3. Set class attributes: `INSTRUMENT_KEY`, `DISPLAY_NAME`
4. Add hardware-config fields as instance attributes (they are auto-persisted)
5. Implement `get_widget()` to return the config panel widget
6. Optionally override `connect()`, `prepare_for_acquisition()`, `get_collapsed_summary()`
7. In whatever modality uses it, add the class to `REQUIRED_INSTRUMENTS` or `OPTIONAL_INSTRUMENTS`

## Adding a new display — checklist

1. Create `pyrpoc/displays/my_display.py`
2. Decorate with `@display_registry.register("my_key")`
3. Set `ACCEPTED_DATA_CONTRACTS` to the contracts this display can render
4. Implement `configure(params)`, `render(data)`, `clear()`
5. Add the key string to `ALLOWED_DISPLAYS` on any modalities that produce compatible data
