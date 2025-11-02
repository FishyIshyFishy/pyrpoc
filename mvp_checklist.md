# 🧩 pyrpoc MVP Implementation Checklist

This checklist walks through the full skeleton build without overthinking downstream effects.  
Each section has a **Definition of Done (DoD)** so you know when to move on.  
Use a “parking-lot” section at the bottom to jot ideas that pop up mid-build.  

---

## 0. Guardrails
- [ ] Create this file in the repo root: `IMPLEMENTATION_CHECKLIST.md`.
- [ ] Add a “Parking Lot” section at the bottom — any idea that takes >5 min goes there.
- [ ] Work on branch `mvp-skeleton`.
- [ ] Rule: vertical slice > polish. Ship working path first.

---

## 1. Freeze Contracts (names only)
- [ ] Each base class lists **required attributes and methods**:
  - `BaseModality`: `PARAMS`, `REQUIRED_INSTRUMENTS`, `EMITS_LIVE`, `EMITS_FINAL`, `validate(app_state)`, `prepare(ctx)`, `start(ctx)`, `stop()`, `teardown()`, `emit_data_object(data)`.
  - `BaseInstrument`: `PARAMS`, `connect()`, `disconnect()`, `is_connected()`, tiny verbs (`queue_lines`, `read_line`, etc.).
  - `BaseDisplay`: `ACCEPTS`, `attach_signals(sig)`, `set_mode(mode)`, `render(data)`.
  - `BaseLaserModulation`: `PARAMS`, `plan(ctx)`, `arm(ctx, recipe)`, `apply_tick(ctx, i, meta)`, `disarm()`.
- **DoD:** all base classes import and compile with stub methods (no bodies, no UI).


## 2. Signals Backbone
- [ ] In your `signals` module, add:
  - `live_data(BaseData)`
  - `final_data(BaseData or list)`
  - `status(text, level)`
- [ ] Ensure UI “Start/Stop” trigger acquisition start/stop.
- **DoD:** a dummy `live_data` emit reaches a test slot.


## 4. Simple Display MVP
- [ ] Implement `SimpleImageDisplay`:
  - `ACCEPTS = ['image-2d']`.
  - Subscribes to `live_data`, ignores other types.
  - Renders with a QLabel/pixmap or any trivial widget.
- **DoD:** emitting a fake `DataImage` shows something visual.


## 4. Mock Instruments
- [ ] Create `MockGalvo` and `MockDAQInput`.
  - Implement `PARAMS`, `connect/disconnect/is_connected`.
  - Add minimal verbs required by Confocal.
  - Register in the instrument registry.
- **DoD:** they appear in the Instrument Manager and toggle connection state.


## 5. Acquisition Runner
- [ ] Implement `AcquisitionRunner`:
  - Receives modality, instruments, params, and signals.
  - On `start`: create context, call `prepare()`, run `start()` in a worker thread.
  - On `stop`: call `stop()`; clean join and teardown.
- **DoD:** Start/Stop calls occur on a worker thread; UI doesn’t freeze.


## 6. Confocal Modality (Simulation Path)
- [ ] Fill in:
  - `PARAMS`: x/y pixels, dwell time, frames.
  - `REQUIRED_INSTRUMENTS`: galvo + daq input.
  - `EMITS_LIVE = ['image-2d']`, `EMITS_FINAL = ['image-2d']`.
  - `validate(app_state)`: checks params + instrument connections.
  - `start()`: emit synthetic frames until stop flag flips.
- **DoD:** Start streams random frames; Stop halts cleanly.


## 7. Validation Gate
- [ ] Add pre-run checks:
  - Modality validates true.
  - Required instruments connected.
  - Display type matches modality’s `EMITS_LIVE`.
- **DoD:** Start button only enables when config valid.


## 8. Last-Session Persistence
- [ ] Define `AppState`:
  - Selected modality key + params.
  - Instruments + configs + desired “connected” state.
  - Active displays + layout token (placeholder ok).
- [ ] On close: serialize whole `AppState` (msgpack/pickle).
- [ ] On open: load and restore; auto-connect; warn if fail.
- **DoD:** reopen app → same modality/instruments/params restored.


## 9. Logging
- [ ] Set up loggers:
  - `pyrpoc.gui`, `pyrpoc.acq`, `pyrpoc.modalities.confocal`, `pyrpoc.instruments.mock`.
- [ ] Console dock shows warnings/errors.
- [ ] File sink records info+.
- **DoD:** worker exceptions appear in GUI console + log file.


## 10. No-Op Laser Modulation Scaffold
- [ ] Add `BaseLaserModulation` with four hooks.
- [ ] Register `NoOpModulation` in modulation registry.
- [ ] Confocal validation allows none or no-op.
- **DoD:** selecting “no modulation” or “no-op” doesn’t break acquisition.

## 11. Final Output Event + Static Display
- [ ] In Confocal: after last frame, emit one `final_data` payload.
- [ ] Add `FinalImageDisplay` subscribed to `final_data`.
- **DoD:** end of run updates final display.


## 12. Polish & Stability
- [ ] Add basic toolbar: Start/Stop, status LED.
- [ ] Status banner for validation/connect errors.
- [ ] Frame-rate throttle in simulation.
- **DoD:** Start/Stop spam doesn’t crash; missing instruments handled gracefully.


## stuff that is not good to revisit later

