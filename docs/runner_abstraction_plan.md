# Runners: entry points a program declares, hosted by an agnostic app

## Context

Every way of starting a program is currently hard-coded:
- **Start and Continuous** are fixed buttons in the acquisition panel.
- **Click-to-acquire** is a special path through `Application.set_pick_armed`/
  `on_point_picked` (which import `Point`, `PointGroup`, `ScanGroup`),
  `ParamForm.pick_armed`, `FieldWidget.arm`/`set_armed`, and relay code in the
  panel.

**Goal:**
- **Entry points become an abstraction, `Runner`.** Each program declares which
  runners it offers, because the entry points that make sense depend on the
  program.
- **The app only hosts them.** It never learns what a runner or a parameter type is.

Today's runners:
- **Single:** run once.
- **Continuous:** run until stopped.
- **Arm & Run:** arm, then wait for a *pick* from a display. The program turns the
  pick into parameter values, then the run starts.

The abstraction that lets Arm & Run stay generic is the **`Pick`**. It's what
arming provides to the run: a dataset plus a location in it. `PixelPick(dataset,
x, y)` is the only kind today. The program supplies the function that turns a
pick into parameters. Pinpoint Raman's version reads `ScanGroup` from the
dataset's provenance and writes `PointGroup.target` from `voltage_at(x, y)`.

## Design

### Structs
- **`structs/picks.py`:** `Pick(dataset)` and `PixelPick(Pick)` with `x, y`.
  Displays produce these; programs consume them. A box or a line is a new
  subclass later.
- **`structs/runner.py`:**
  - **`Runner`:** the base, a frozen declaration with one method, `attach(ctx)`.
    Each attach builds fresh per-program state, so one declaration can be shared
    as a class attribute.
  - **`RunnerContext`:** the abstract surface the host implements:
    - `execute(continuous=False)`, `stop()`, `running`
    - `on_run_started(cb)` / `on_run_finished(cb)`
    - `params`, the selected program's `BlockMap`, and `params_written()` to
      announce writes made outside the form
    - `request_pick(kind, callback)` / `cancel_pick()`: one request at a time;
      the callback gets the pick, or `None` if cancelled
    - `add_control(control)`
    - `status(text)`, `blockers() -> list[str]`
  - **Control descriptors** (Qt-free primitives, not features):
    - `Button(label, on_press, icon=None, tooltip="")`
    - `Toggle(label, on_change, icon=None, tooltip="")`, with `.set(checked)`
    - both have an `enabled` attribute
    - `icon` is optional and names a file in `pyrpoc/assets/`.
  - **The three generic runners**, which sit here next to the base the same way
    `IntField` sits next to `Field`:
    - **`Single`:** adds `Button("Start", icon="single")` → `ctx.execute()`.
    - **`Continuous`:** adds `Button("Continuous", icon="multi")` →
      `ctx.execute(continuous=True)`.
    - **`ArmAndRun(pick: type[Pick], apply: Callable[[Pick, BlockMap], None], label="Arm")`:**
      - adds a toggle
      - when the toggle turns on: if there are blockers, report them and turn the
        toggle off; otherwise `request_pick(pick, on_pick)`
      - when the toggle turns off: `cancel_pick()`
      - `on_pick(p)`:
        1. Turn the toggle off.
        2. If `p` is None, return.
        3. Call `apply(p, ctx.params)`. A `ParameterError` goes to status.
        4. `ctx.params_written()`, then `ctx.execute()`.
      - it also cancels its pick on `on_run_started`
- **`structs/program.py`:** `Program` gains
  `runners: list[Runner] = [Single(), Continuous()]`. The docstring's "four
  attributes" rule becomes five.

### Programs
- **`programs/pinpoint_raman.py`:**
  - `runners = [Single(), Continuous(), ArmAndRun(PixelPick, aim_at_pick, label="Acquire at point…")]`
  - `aim_at_pick(pick, params)` sets
    `params[PointGroup].target = Point.from_pixel(pick.dataset, pick.x, pick.y)`
- **`programs/components/param_groups.py`:** add `Point.from_pixel`, moving the
  provenance → `ScanGroup` → `voltage_at` logic out of
  `Application.on_point_picked`, and raise `ParameterError` when it can't place
  the pixel.
- **Other programs** inherit the default `runners`.

### App (the host)
- **`app/runner.py` → `app/executor.py`**, class `Runner` → `Executor`: the
  engine (threads, datasets, saving). `RunBridge` wraps it as before.
- **`app/runner_host.py`:** `RunnerHost` implements `RunnerContext`.
  - **Lifecycle:** `select_program` detaches the old program's runners and
    attaches the new one's. Detaching cancels any pending pick and removes that
    program's controls, so an armed display never outlives a program switch.
  - **Picks:** routed to displays that opt in by duck typing:
    `set_pick_mode(kind | None)` and a `picked` signal carrying a `Pick`.
    `image_2d` implements `PixelPick`. Overlay, mask editor and spectrum drop
    their no-op stubs.
  - **Controls:** kept in order; `controls_changed` is emitted.
  - **`execute`:** calls `Application.start_run`. `blockers()` is the check the
    acquisition panel does today (missing devices, empty save name), moved here.
- **`app/application.py`:** remove `pick_armed`, `set_pick_armed`,
  `on_point_picked`, `pick_failed`, `point_acquired`, and the
  `Point`/`PointGroup`/`ScanGroup` imports. Add a `params_written` signal.

### Acquisition panel (a generic renderer)
- **Transport row:**
  - a controls strip rendered from the host's list:
    - `Button` → `QPushButton`
    - `Toggle` → a checkable `QPushButton`
    - an asset icon if one is given, otherwise the label
  - the fixed Stop button
  - the save widgets
- **One enabling policy for every control:** disabled while blockers exist or a
  run is in progress, and also when the runner clears `enabled`.
- **The form reloads on `params_written`.**
- **Removed:** the hard-coded Start/Continuous buttons and every pick relay.

### Parameter form
- `PointPicker` becomes a plain editor: spin boxes and the "from: …" label, with
  no button.
- The `arm`/`set_armed` hooks and `pick_armed`/`show_pick_armed` go away.

### Masks (last phase)
- **References instead of arrays.** `Mask` becomes a reference (`source_id`,
  `source_label`, `port`, `line`). The array is filled in only at resolution, and
  `MasksField.decode` keeps the bindings (no more `()`).
- **`Field.resolve(value, library)`** is added to `structs/params.py`, returning
  the value unchanged by default. `MasksField.resolve` attaches each entry's
  pixels, and raises `ParameterError("mask '<label>' is not open")` for an entry
  that has been closed.
- **The executor resolves at start.** It makes resolved copies
  (`dataclasses.replace`) of the declared blocks for the program's `BlockMap`.
  The shared blocks and the metadata keep only references.
- **Editors move out of the form:**
  - a `Field.editor()` hook is added
  - the acquisition panel passes `FieldContext(library=…)`
  - `MaskTable` and `PointPicker` move to `programs/components/editors.py`
  - `panels/components/param_form.py` then imports only `structs`

## Phases (each leaves the app runnable)
1. **Rename the engine:** `app/runner.py` → `app/executor.py`.
2. **Structs:** add `picks.py` and `runner.py` (the base, `RunnerContext`,
   controls, `Single`/`Continuous`/`ArmAndRun`), plus `Program.runners`.
3. **Host, panel and displays together:**
   - `app/runner_host.py`
   - the generic controls strip in the acquisition panel
   - displays switched to `set_pick_mode`/`picked`
   - the Pinpoint Raman declaration and `Point.from_pixel`
   - delete the old pick path
4. **Masks:** references, `resolve`, executor resolution, editors moved out of the
   form.
5. **Docstrings:** `structs/__init__` lists `picks` and `runner`; the `Program`
   docstring covers `runners`.

## Verification
1. `uv run pyright pyrpoc`: ≤ 37 errors, the current baseline.
2. `grep -rn "PointGroup\|ScanGroup\|\bPoint\b\|\bMask\b" pyrpoc/src/app pyrpoc/src/panels`
   should return nothing.
3. `structs.*` imports headless, without `PyQt6.QtWidgets`.
4. **Offscreen smoke test** (extend `smoke.py`):
   - Simulation shows Start and Continuous; Pinpoint Raman also shows
     "Acquire at point…".
   - Start completes a saved Simulation run. Continuous starts, and Stop ends it.
   - **Arm & Run:**
     1. Select Pinpoint Raman. Add an image panel and a library dataset whose
        provenance carries a `ScanGroup`.
     2. Toggle on, then emit a `PixelPick` from the panel.
     3. Assert `PointGroup.target == voltage_at(x, y)` and that the form reloaded.
        Execute is attempted (it fails without hardware; the failure goes to the
        status line and doesn't crash).
     4. The toggle is off and the displays are back to no pick mode.
   - Switching program while armed cancels the pick.
   - **Masks:**
     1. File a `Mask2D`, bind it, and run Simulation: the program receives the
        array.
     2. Close the entry: the run fails with "is not open".
     3. The binding survives a session round trip.
5. **By hand:** `uv sync`, then `uv run pyrpoc`.
