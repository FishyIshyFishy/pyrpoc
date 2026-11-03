# Phase 0 of the autonomous work: a clear architecture

Before any autonomous work, the software gets one clear architecture: how
its pieces fit together. `structs/` becomes `core/`, with a plain README that
defines every noun and the one flow between them. The Acquisition panel
becomes a panel like any other, and the folders are reorganised to match.

**Two words used throughout:**
- **Collection.** Something the app keeps in memory while it runs, holding
  the current set of one kind of thing. There are three:
  - the **device inventory** (the hardware);
  - the **ProgramInputs** (every setting's current value);
  - the **Library** (the open program outputs).

  Any part of the app can look in a collection, change it through its own
  operations, and be told when it changes.
- **Working state vs screen state.**
  - **Working state** is what the app is: the three collections, and where
    runs save.
  - **Screen state** is only how it's shown: which program the Acquisition
    panel shows, dock layout, zoom.

**Behaviour stays the same,** except where masks and pixel picking deliberately
change (phases 3 and 6). Every phase leaves the app working and the checks
passing, so the migration can stop between phases.

**Naming rule for this work:** specific nouns over vague ones. A name should
say what the thing is to someone who isn't a software engineer, even if
that makes it longer (`ProgramInputs`, not `Inputs`).

---

## 1. Why

**The Acquisition panel is a god object in code.** It's the most important
panel to the user, but that's no reason for the code to treat it as special.

Today the `Acquisition` object (`acquisition/model.py`) holds all of these
together:
- **the selected program:** screen state;
- **the parameters (`BlockStore`):** working state, but keyed by the blocks the
  form draws;
- **the runners:** attached to whichever program the form shows;
- **pick routing** between the image panels and runners, wired up by
  `app/window.py`;
- **runs:** the executor, events and save target.

**Everything else bends around that:**
- **Panel → program has no channel of its own.** It borrowed `Data`: `Pixel`
  and `Mask` are in `data.py`, travel as fake datasets, and fill the Library
  with things no program produced (`Origin.AUTHORED`).
- **Parameters persist through blocks.** So blocks are visual groups,
  persistence units and the identity of an input, all at once.
- **Anything the user edits outside the form** has to be spawned from a form
  row.

---

## 2. The architecture (this becomes `core/README.md`)

### The flow

```
 ProgramInputs ──▶ Run ──▶ ProgramOutputs ──▶ Library ──▶ panels show them
       ▲                                                        │
       └──── panels set program inputs, or answer input requests ◀┘
```

Everything the software does is this loop:
1. Program inputs feed a run.
2. A run publishes program outputs.
3. Outputs sit in the Library, where panels show them.
4. What the user does in panels sets program inputs.

### The nouns

**Running**

| Noun | Definition |
|---|---|
| **Device** | A piece of hardware the microscope has. Found through `device_registry`. |
| **Program** | A declaration of one experiment: the devices it needs, the program inputs it reads (laid out in input sections), the program outputs it emits, and the runners that can start it. Its `run` does the work. |
| **Run** | One execution of a program. It gets a frozen copy of the program's inputs, claims its devices, and publishes into its program outputs. |
| **RunContext** | What a running program is handed: its input values, its devices, its program outputs, and cancel. |
| **Runner** | A way to start a program (once, continuously, arm and run). It's declared by the program, and its controls are drawn by whichever panel shows the program. |

**Program inputs: user → program**

| Noun | Definition |
|---|---|
| **ProgramInput** | The declaration of one value a program reads, like dwell time or a mask. It has a key, a label, a default and rules for valid values. **The same key means the same value in every program that declares it.** That's how a setting carries across modality changes. |
| **InputSection** | A labelled list of program inputs. Purely layout: how a panel groups a program's inputs on screen. |
| **ProgramInputs** | The collection of current values for every program input, by key. Any panel may read it, set a value (validated by the input's rules) or subscribe. It's saved in the session, and a run's values go into its recording. |
| **InputRequest** | A panel or runner asking for a value of one type that another panel can supply. Whichever supplying panel the user acts in answers it. There's no storage: it's a question with one answer. |
| **PixelPick** | The value a click on an image supplies: the program output clicked on, plus `x` and `y` in its pixels. |
| **Mask** | A drawn region, as a program input value. The Mask Editor supplies it; ROI and Modulation inputs hold it. |

**Program outputs: program → user**

| Noun | Definition |
|---|---|
| **Data** | One value a program produced: an image, a spectrum, a mosaic. Immutable. **Only programs make it.** |
| **ProgramOutput** | One named output of one run. It holds the latest `Data`, where it came from (`Provenance`), and its writer. Every published value reaches its files; memory holds only the latest. |
| **Library** | The program outputs that are open: acquired now, or loaded from disk. Nothing else. |
| **Recording** | One run's program outputs on disk, opened through its `_meta.json`. |
| **SaveTarget, Writer** | Where a run saves, and how each kind of `Data` is written and read. |

**Views**

| Noun | Definition |
|---|---|
| **Panel** | A dock widget: something the user sees and acts in. It reads the collections, calls their operations, and may answer input requests for the types it `supplies`. **The Acquisition panel is one.** |
| **DataPanel** | A panel that shows one program output of the kind it `renders`, chosen with a source dropdown. |

### The rules
1. **Data flows one way.** Only a run or loading a recording creates a
   `ProgramOutput`.
2. **Panels never call programs, and programs never call panels.** They meet
   only through ProgramInputs, InputRequests and the Library.
3. **Every panel is equal.** None is special in code. The app hands each
   panel the collections it needs.
4. **Working state and screen state are separate.** The session saves them
   separately.

---

## 3. Target folder layout

### 3.1 The package

Today the top level mixes different kinds of things side by side: the nouns
(`structs/`), add-ons (`plugins/`), image and DLL files (`assets/`), the
app's working parts, generic widgets, and the code that assembles it all.
The new layout has one folder per kind:

```
pyrpoc/
  main.py                 starts the app
  core/                   the nouns: what each piece is (README.md defines them)
  engine/                 the app's working parts, one folder per collection
    device_inventory/       the hardware list, and the Devices panel
    data_library/           the open program outputs, reading and writing files, and the Data Library panel
    acquisition/            ProgramInputs, runs, and the Acquisition panel
  plugins/                things you add more of
    devices/                one folder per instrument
    programs/               one folder per experiment, plus building_blocks/
    panels/                 one folder per add-on panel (was data_panels/)
  shared/                 used by many parts, belonging to none
    widgets/                generic widgets that know no feature (was qt_components/)
    assets/                 files that aren't code
      icons/                  the SVG icons
      sdks/                   vendor DLLs (Prior, Touptek)
  app/                    builds everything, connects it, shows the window, remembers the session
```

**Why `shared/`:** widgets and assets aren't parts of the architecture.
They're supplies the parts use, so they sit together under one name that
says so, rather than beside `core/`, `engine/`, `plugins/` and `app/`.

**What moves:**
- **`structs/` becomes `core/`** (section 3.2).
- **`device_inventory/`, `data_library/` and `acquisition/`** move under
  `engine/`.
- **`qt_components/` becomes `shared/widgets/`**, and **`assets/` becomes
  `shared/assets/`**. `icons.py` and the two `sdk.py` files update their
  paths.
- **`plugins/data_panels/` becomes `plugins/panels/`.**
- **The built-in panels** (Acquisition, Data Library, Devices) stay with
  their engine part. They're constructed like every other panel; living
  there doesn't make them special.

The import rules follow the same order, top to bottom:

```
app → engine → plugins → shared → core
```

`shared/` may import only `core/`. Nothing in it knows a feature.

### 3.2 `core/`

Flat, one noun (or one close family) per file. It no longer mirrors the
package.

```
core/
  README.md           section 2 of this plan, kept current
  registry.py         Registry
  device.py           Device + device_registry
  program.py          Program, RunContext + program_registry
  runner.py           Runner, RunnerContext, Control, Button, Toggle
  program_inputs.py   ProgramInput, the value rules (int, float, choice, ...), ProgramInputs (protocol), Editor, FieldContext
  input_section.py    InputSection
  input_requests.py   InputRequest, InputRequests (protocol), PixelPick
  mask.py             Mask (a program input value type)
  data.py             Data, ArrayData, Image, DecayHistogram, RawSamples, Spectrum, Mosaic, DATA_KINDS
  program_output.py   ProgramOutput, Provenance, Origin
  library.py          Library (protocol)
  saving.py           SaveTarget, Writer + writer_registry
  panels.py           Panel, DataPanel, SourcePicker, DevicePanel + registries (the only Qt file)
```

---

## 4. Docs and docstrings

**All docs are removed in phase 1.** Docs are written fresh once phase 0 is
done and the implementation and experience are settled.
- **Removed:**
  - `docs/how-to-pyrpoc.md`;
  - `docs/internal/` (`releasing.md`, `future_human_todo.md`);
  - the already-deleted `architecture.md`, `data-format.md` and
    `future_autofocusing-curve.md`.

  References to them go too: the architecture-map line in
  `.claude/CLAUDE.md`, and the `data-format.md` pointer in
  `data_library/format.py`.
- **Kept until phase 0 ends:** the planning docs (`plan.md`, `agent_plan.md`,
  this file) and `core/README.md`, which defines the nouns this migration
  builds.
- **Kept for good:** `docs/swabian_demos/`, which is the vendor's example code,
  not our docs.

**Docstrings describe exactly what the code does, and only what's
necessary.** The rule, added to `.claude/CLAUDE.md`'s code style in phase 1:
- **One or two plain sentences:** what this module, class or function is or
  does.
- **Not in a docstring:**
  - history;
  - what other modules do;
  - how callers use it;
  - alternatives that were considered;
  - restatements of the type annotations.
- **A "why"** goes in a comment at the line that needs it, as CLAUDE.md
  already says.

So docstrings don't go stale when other code changes. Each phase rewrites the
docstrings of the files it touches; phase 7 sweeps the rest.

---

## 5. Phases

Every phase ends with:
- the CLAUDE.md checks on the touched files (`ruff check --fix`,
  `ruff format`, `pyright`);
- `lint-imports` on the whole project;
- the **regression walk** below, in the app with `--simulated`.

**The regression walk:**
- **Modality switch.** Switch program, then switch back: shared settings
  (dwell, DAQ) carry over.
- **Single and continuous runs** of Confocal (simulated) show live images, and
  save when saving is on.
- **Pinpoint picking.** Galvo Pinpoint: arm, click an image, and it parks at
  that pixel.
- **Masks.** A Modulation mask drives its line in the simulation.
- **Session.** Quit and relaunch: program, parameters, docks and save
  settings come back.
- **Recordings.** A saved recording loads in the Data Library.

### Phase 1: the new folder layout, `core/` and its README, docs removed
Only files and folders move. **No class is renamed yet.**
- **The package.** Move folders into the layout of section 3.1:
  - `engine/`, `shared/widgets/`, `shared/assets/` and `plugins/panels/`;
  - `icons.py` and the `sdk.py` files point at `shared/assets/`.
- **`core/`.** Move every `structs/` file into the flat layout of section
  3.2. Split `params.py` into `program_inputs.py` and `input_section.py` only
  where it splits cleanly.
- **The README.** Write `core/README.md` from section 2, marking what isn't
  true yet ("from phase N") so it never lies.
- **Contracts.** Update `pyproject.toml`'s import-linter contracts to the
  section 3.1 order and folder names. Update `.claude/CLAUDE.md`'s
  architecture section, and every import (about 190 for `structs` alone).
- **Packaging.** The DLLs and icons are still included in the built package
  from their new places (check `pyproject.toml`'s package data).
- **Docs.** Remove the docs listed in section 4, and add the docstring rule
  to `.claude/CLAUDE.md`.
- **Done when:**
  - no old folder path (`structs`, top-level `assets`, `qt_components`,
    `data_panels`, top-level `acquisition`, `data_library`,
    `device_inventory`) is referenced;
  - the Prior stage and Touptek camera still load their DLLs;
  - icons show;
  - the regression walk passes.

### Phase 2: the renames
Mechanical, no behaviour change:

| From | To |
|---|---|
| `Dataset` | `ProgramOutput` |
| `dataset.py` | `program_output.py` |
| `dataset.output` | `program_output.name` |
| `RunContext.dataset_for` | `output_for` |
| `Program.uses` | `Program.devices` |

- **Done when:** no `Dataset` or `uses` name remains, and the regression walk
  passes.

### Phase 3: InputRequests replace pixel routing; `Pixel` leaves `Data`
- **InputRequests in core.** Add `InputRequest`, the `InputRequests` protocol
  and `PixelPick` to `core/input_requests.py`, with one implementation the
  app owns and hands to every panel and every runner context.
- **Panels declare what they answer.** `supplies = (PixelPick,)` on the Image
  panel. The panel arms itself (crosshair) while a `PixelPick` request is
  open, and answers it on a click.
- **ArmAndRun asks through InputRequests.** It calls `request(PixelPick, ...)`
  in place of `request_pixel`. `Point.from_pick` reads
  `pick.output.provenance`.
- **Deleted:**
  - `Pixel` and its `DATA_KINDS` entry;
  - `pick_of`;
  - `picked` and `set_picking` on `DataPanel`;
  - `RunnerHost`'s pick routing;
  - the pick wiring in `app/window.py` (`connect_panel` / `disconnect_panel`).
- **Done when:** Galvo Pinpoint and Pinpoint Raman behave exactly as before,
  and no code outside `input_requests.py` routes clicks.

### Phase 4: the Acquisition panel becomes a panel
Split today's `Acquisition` object into the working parts and the panel.
The keys stay what they are today, so nothing else changes yet.
- **The ProgramInputs collection** (`engine/acquisition/program_inputs.py`, no Qt) replaces
  `BlockStore`.
  - Values are keyed by the existing dotted paths (`ScanGroup.dwell_us`).
  - It implements `core.ProgramInputs`: get, set (validated), subscribe, and
    to/from dict.
  - `ctx.params[ScanGroup]` still works, built from the collection, so no program
    changes in this phase.
- **A Runs object** keeps the executor, events, save target and live runs.
  - `start(program_key)` and `blockers(program_key)` take a program; nothing
    reads a "selected program".
- **The Acquisition panel owns the selection.**
  - Which program it shows is its own screen state.
  - It attaches that program's runners through a `RunnerContext` built from
    Runs, ProgramInputs and InputRequests.
  - `params_written` disappears: the ProgramInputs collection notifies every panel
    of every change.
- **The app hands panels the collections.** `window.py` constructs the Acquisition
  panel the way it constructs the others. Nothing outside it reads
  `acquisition.selected_program` or `acquisition.host`.
- **The session splits** into working state (program input values, save
  target, devices, Library settings) and screen state (docks, each panel's
  state, including the Acquisition panel's selected program).
- **Done when:**
  - the regression walk passes;
  - nothing in `engine/acquisition/` except its panel holds a selection;
  - `window.py` has no Acquisition-specific wiring beyond creating its dock.

### Phase 5: program inputs declared on their own; blocks become input sections
- **Standalone declarations.** Each program input is declared once with its
  own key, e.g. `DWELL = ProgramInput[float]("scan.dwell_us", ...)`, next to
  the code that shares it (`building_blocks/program_inputs/`, replacing
  `parameter_groups/`).
- **Programs declare `sections`**, each an `InputSection`: a label plus a
  list of program inputs. They read `ctx.inputs[DWELL]`, which is typed
  through `ProgramInput[T]`.
- **Logic that lived on blocks moves.** `ScanGroup.voltage_at` and similar
  move to functions, or to small value types built from program inputs.
- **Conflicts fail at registration.** Two declarations of one key with
  different types are an error.
- **Recordings.** `parameters` are keyed by input key. This is a new recording
  format version; see open question 1.
- **Done when:**
  - no `Group`, `BlockStore` or `BlockMap` remains;
  - the regression walk passes, the modality switch especially.

### Phase 6: masks become program input values
- **`Mask` moves to `core/mask.py`** as a program input value type, not
  `Data`.
  - ROI and Modulation inputs hold masks directly.
  - Masks are encoded inline as shape plus base64 `np.packbits`, in the
    session and in recordings.
- **The Mask Editor stays a dock panel**, opened from the panels menu as
  today, and declares `supplies = (Mask,)`.
- **The new workflow:**
  1. Draw in the Mask Editor, as today.
  2. In the Acquisition panel, click **Take from Mask Editor** on a mask
     input (the ROI mask, or a Modulation row's mask cell). This opens a
     `Mask` input request.
  3. Every open Mask Editor lights up a **Send** button.
  4. Clicking it answers the request, and the input holds the mask.

  **Clear** empties the input, and an empty mask input means "no mask".
  **Save / Open file** in the editor stay as today, for the user's own reuse.
- **Deleted:**
  - `Mask` from `data.py`;
  - the editor's "add to library";
  - `MaskRef` and `resolve_mask`;
  - `MaskSourceCombo` and `CLOSED_SUFFIX`;
  - `Origin.AUTHORED` and its purge exception.
- **Done when:**
  - the Library holds only program outputs;
  - `data.py` holds only what programs produce;
  - a mask survives a relaunch, drives the same TTL as before, and appears in
    the recording's `parameters`.

### Phase 7: docstring sweep
- Rewrite every docstring the earlier phases didn't touch to the rule in
  section 4.
- Drop every "from phase N" mark in `core/README.md`.
- **Done when:** every module, class and public function's docstring passes
  the rule, and the README describes the code exactly as it is.

---

## 6. Open questions

1. **Old recordings' parameters.** Phase 5 changes how `parameters` are
   keyed. With `data-format.md` gone, its "load forever" promise is gone too,
   and the autonomous plan says no backwards compatibility.
   - Option one: keep a decoder for the old shape, so old recordings still
     load and only lose `Point.from_pick` on their images.
   - Option two: drop it.
2. **Large program input values** (unit sets, later). Masks are small enough
   to inline, but units are not. This doesn't block phase 0, since no unit
   input exists yet. When it does, the ProgramInputs collection needs a way to
   keep a value beside the session and the recording instead of inside them.
3. **Answering an input request.** For masks: always the explicit **Send**,
   or answered at once when exactly one Mask Editor is open? This plan uses
   **Send** everywhere, matching "arm, then act in the panel" for pixel
   picks.
4. **Where the InputRequests implementation lives.** It's tiny and used by
   panels and runners alike. The options are `acquisition/` (beside the
   runners), or a small file the app owns.
5. **More names.** `Run`, `Runner`, `Library` and `Data` are short. Do any of
   them read as vague to you?

## 7. Later, not phase 0
- **Live program inputs.** A run reads the latest value of an input between
  steps; the Experiment panel's queue needs this.
- **Experiment = folder,** grouping in the Library and in source dropdowns.
- **The autonomous nouns** (units, loop output) in `core/`.
