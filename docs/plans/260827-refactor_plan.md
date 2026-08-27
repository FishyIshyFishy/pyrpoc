# Plan to enable new stuff
## Current problems
Currently, base class of `pyrpoc.modalities.base_modality.py` owns the loop. There is 
```python
while not should_stop: acquire_once()
```
meaning that every workflow must be some sort of looped acquisition. Click-driven with context, autofocusing, etc. are not one frame repeated. Further, this is an example of the fact that modalities are function calls with a kill switch. They are not extensible programs.

Additionally, the UI owns all parameters. `pyrpoc.gui.main_widgets.acquisition_mgr.py` has `collect_values()` which owns scraping widgets for parameters. This violates the state living in a model, and the UI being a view on top of the model. The state is not the view, the view should just be something that reads and writes to the state. 

There is also no place for feature logic that spans acquisition and display to live. This is why `export_rpoc_input()` exists for every display, and every modality has to have a list of allowed displays. While these are necessary functionalities, it is weird to have them within abstract display/modality logic.

## Proposed new file layout
```
pyrpoc/
├── app.py              entry point; builds the app, wires top-level pieces
│
├── core/               shared vocabulary. no Qt, no hardware, no I/O.
│   ├── space.py        coordinate spaces, Frame, units
│   ├── context.py      Point, Region, DatasetRef — the plug/socket types
│   ├── streams.py      stream identity + shape contracts (Image2D, Cube3D, Trace1D)
│   ├── params.py       parameter field types, validation, coercion
│   ├── modulation.py   MaskSpec — what a mask is, not how it's drawn or applied
│   └── errors.py
│
├── devices/            one folder per hardware thing; driver + its panel together
│   ├── base.py
│   ├── registry.py
│   ├── daq/
│   │   ├── device.py
│   │   ├── simulated.py
│   │   └── panel.py
│   ├── galvo/          device.py holds the volts↔sample calibration
│   ├── time_tagger/
│   └── stage/
│
├── operations/         bounded hardware actions. plain functions. no self, no loop.
│   ├── raster.py       waveform gen + synchronized AO/AI/DO scan
│   ├── point.py        dwell at one position
│   ├── tagger.py       read a FLIM frame
│   └── modulation.py   MaskSpec → per-pixel TTL waveform
│
├── data/
│   ├── dataset.py      array + frame + incremental writes + change notification
│   ├── library.py      the live collection of datasets, with identity
│   ├── io.py           save/load: tiff, npz, ome
│   └── transforms.py   normalize, project, slice — shared by views and recipes
│
├── run/                executes procedures
│   ├── procedure.py    Procedure base + RunContext
│   ├── runner.py       worker thread, inbox, cancellation, status
│   └── claims.py       device locks / exclusivity
│
├── recipes/            THE feature folder — the only one you add to for a feature
│   ├── base.py         @recipe declaration schema
│   ├── registry.py
│   ├── confocal/       recipe.py · params.py · procedure.py
│   ├── point_scan/
│   ├── z_stack/
│   ├── mosaic/
│   ├── flim/
│   └── fit_lifetime/   a recipe that touches no hardware
│
├── views/              render datasets, offer context, emit interaction events
│   ├── base.py         renders streams; declares `offers = [Point, Region]`
│   ├── registry.py
│   ├── image_2d.py
│   ├── overlay.py
│   ├── mosaic_canvas.py
│   ├── decay.py
│   └── mask_editor.py
│
├── shell/              application chrome
│   ├── window.py
│   ├── launcher.py     recipe launcher (replaces the modality dropdown)
│   ├── param_form.py   generic form generated from a params model
│   ├── run_bar.py      what's running, what step, stop/pause
│   ├── menus.py        needs/offers matching → context menus
│   ├── docking.py
│   └── theme/
│
└── session/            config persistence only
    ├── state.py        what exists, how it's configured, layout
    └── store.py        JSON read/write
```

## Core concepts

The layout above only makes sense once these nouns are pinned down. Each is a distinct
kind of thing with a distinct reason to exist.

**Device** — a handle on one piece of hardware. Owns its connection, settings, calibration, and its settings panel. Has a simulated sibling (`simulated.py`) so the whole app runs with nothing plugged in.

**Operation** — one bounded hardware action. A plain function: explicit arguments in, arrays out. No `self`, no loop, no saving, no Qt, no knowledge of what it is being used for. The unit is a *clock domain*, not a device — the raster operation drives AO + AI + DO together because they share a sample clock, and splitting them into separate "positioner" and "detector" objects would misrepresent the hardware.

**Procedure** — the experiment, written as ordinary Python running on a worker thread. It decides which operations to call, in what order, with what arguments, and what to do with the results. This is the piece that is currently cut in half between `acquire_once()` and `build_continuous_worker()`; here it is whole, and it belongs to whoever is writing the feature rather than to a base class.

```python
class ZStack(Procedure):
    def run(self, ctx):
        for z in ctx.params.z_positions:
            ctx.devices.stage.move_z(z)
            ctx.publish("stack", raster_scan(**ctx.params.scan), z=z)
```

**RunContext (`ctx`)** — the service surface handed to a procedure. Deliberately small; it will attract additions and should be defended.

```python
ctx.params                            # the parameter object for this run
ctx.devices                           # resolved device handles
ctx.publish(stream, data, **coords)   # write into a dataset
ctx.wait_for(event, timeout=None)     # blocking read of the inbox
ctx.poll(event)                       # non-blocking
ctx.sleep(seconds)                    # cancellable
ctx.status(text)                      # progress -> run bar
ctx.check_cancel()                    # raises Cancelled
```

`wait_for` is what makes click-driven acquisition possible at all: a running procedure has an inbox, so input can arrive from a view, a script, a scheduler, or another procedure without the procedure knowing who sent it. Today the only signal that can enter a running acquisition is the stop flag.

**Recipe** — the feature unit, and what "modality" becomes. A declaration binding together: the context it consumes, the devices it requires, its parameter model, its procedure, and the streams it produces.

```python
@recipe
class PointScan:
    name      = "Scan here"
    needs     = [Point, Galvo, DAQ]
    params    = PointScanParams        # optional; absent -> runs immediately
    procedure = PointScanProcedure
    produces  = {"preview": Image2D}
```

A modality is *not* a subclass of a recipe. "Modality" stops existing as a type. How a
recipe is launched — main launcher, view context menu, toolbar, script — is a property
derived from what it `needs`, not a class distinction. That way a new launch surface
never requires a new class.

**Context types** (`Point`, `Region`, `DatasetRef`) — the plug-and-socket vocabulary. A
view declares what it can offer; a recipe declares what it needs; the shell computes
which recipes appear where. Nobody writes "PointScan goes in the image_2d context menu."
Write a new point-offering view and it immediately hosts every point-recipe, including
ones written later. Write a new point-recipe and it appears in every point-offering view.
This is what replaces `allowed_displays`.

**Stream** — a named output of a running procedure, carrying a shape contract
(`Image2D`, `Cube3D`, `Trace1D`). Two things that `DataKind` currently conflates are
deliberately separated: *what the array is* (the contract — used to check a view can
render it) and *which output it came from* (the name — used to bind it to a specific
view). That separation is what lets one FLIM run send its DAQ image to one display and
its Swabian image to another, which is impossible today without minting a fake DataKind.

**Dataset** — where acquired arrays actually live. Owns the array, its coordinate frame,
incremental writes, and change notification. Views render datasets; they never own
arrays. Aggregating structures (a mosaic, a z-stack) are datasets that accept writes at
coordinates — which is why "record many spots into one plane" needs no special display
type.

**View** — renders one or more streams from datasets, declares what context it can offer,
emits interaction events. Never references a recipe or a procedure.

**Parameter model** — a dataclass that holds the authoritative values, with the form
generated from its field metadata and writing back into it. The form is a view onto the
parameters, not the place they live.

## "Derivation" of the layout

A folder structure does exactly two jobs: it tells you where a new thing goes, and it
constrains what may depend on what. The second job is the one usually forgotten, and it
is the one that decides whether the codebase stays workable — folders are the only
lightweight mechanism available for saying "this part may not know about that part." If
folders do not encode dependency constraints they are filing cabinets, and filing does
not prevent tangling. Everything below follows from those two jobs.

### 1. Things that change together live together

The test for whether two files belong in one folder is not "are they similar." It is:
**when you make a typical change, do you touch both?**

This is where a `gui/` folder fails. The time tagger's driver and its settings panel
change together constantly — add a trigger-voltage parameter to the device, add a field
to the panel, same sitting, same reason. The time tagger's panel and the mosaic canvas
never change together and have nothing to do with each other. But a `gui/` folder files
the second pair as siblings and separates the first pair, because it groups by *which
library the file imports*, and "imports PyQt6" is not a thing that changes together.

The cost is visible in the current tree. Adding FLIM touched five folders:
`modalities/flim/`, `instruments/time_tagger.py`,
`instruments/instrument_widgets/time_tagger_widget.py`, `displays/flim_display.py`, plus
new entries in `backend_utils/acquired_data.py`. To understand FLIM you read five places;
to remove it you hunt five places; nothing tells you when you have found them all.

So: group by subject, not by technology. There is no `gui/` folder because "is Qt" was
never the thing that mattered.

### 2. Some things serve one feature, some serve many

That rule alone would put everything in feature folders, which does not work. A 2D image
view is not owned by confocal — every recipe producing images uses it. A raster scan is
not owned by z-stack — confocal, mosaic, and z-stack all call it.

So a second question, applied to every file: **does this serve one thing or many?**

- Serves one -> lives inside that thing's folder (the time tagger's panel; confocal's procedure).
- Serves many -> lives in a folder of its kind, alongside interchangeable siblings (`views/`, `operations/`).

This is why the tree is mixed: `devices/time_tagger/panel.py` contains Qt while `views/`
is a separate folder that is entirely Qt. That looks inconsistent until the rule is
applied, and then it is forced — the panel serves exactly one device, the image view
serves every recipe.

The same question keeps producing answers. Why is `procedure.py` inside
`recipes/confocal/` while `raster.py` sits in a shared `operations/` folder? A procedure
belongs to exactly one recipe; a raster scan belongs to many.

### 3. Dependency order comes from what changes most

Order the layers by asking what you want to change freely.

The thing changed constantly is **features**, so features sit at the bottom of the
dependency graph with everything else ignorant of them. Hence the strongest rule in the
tree: *nothing imports `recipes/`.* If any part depended on a specific recipe, that recipe
could never be deleted and the next one would have to imitate it.

The thing changed least is the **shared vocabulary** — the nouns two folders both need to
say, like `Point` and `Frame`. Those sit at the top importing nothing. When two folders
need the same word, the word cannot live in either, or one starts importing the other for
no reason.

The middle falls out of physical reality:

- Hardware exists regardless of what you do with it -> `devices` depends only on vocabulary.
- An action on hardware needs the hardware -> `operations` depends on `devices`.
- Data exists regardless of how it was acquired -> `data` depends only on vocabulary. This is why saving is not a modality's job.
- Running a program needs actions to run and somewhere to put results -> `run` depends on `operations` and `data`.
- A feature composes all of the above -> `recipes` at the bottom.

Then views. Views render data, so they need `data` and vocabulary. Nothing else — and
nothing else should be *permitted*. "Acquiring data and showing data are distinct things"
stops being a principle to remember and becomes `views/` being unable to import `run/` or
`recipes/`. The structure holds the rule so discipline does not have to.

### 4. `core/` is the folder to distrust

`core/` exists for one reason: two folders need to say the same word. That is the only
honest justification and it is narrow.

It is also the folder that rots, because "shared" is slippery and anything homeless
drifts there. That is exactly what `backend_utils/` became — `array_contracts`,
`state_helpers`, `parameter_utils`, `registry`, `contracts` landed there not because many
folders needed them but because they were not obviously GUI. "Not GUI" is not a subject.

One test keeps it honest: **if only one folder imports it, it does not belong in `core/`.**

### 5. Two kinds of state, split by lifetime

Configuration is small, JSON, and exists so the workbench comes back on relaunch.
Acquired data is gigabytes, lives in TIFF, and exists because it is the experimental
result — opened as a file months later, possibly in another program.

Different size, format, lifetime, and reason to exist: four independent reasons they are
different subjects. Merge them and the session file starts trying to hold arrays, or the
TIFFs start trying to remember dock layout.

### 6. One place is allowed to know everything

Ignorant parts still have to be connected by something. A display that knows nothing
about acquisition must nonetheless end up feeding a click into a running scan, and that
connecting code has to physically exist somewhere.

That cannot be designed away — the only choice is whether it lives in one identifiable
place or smeared across the parts that were supposed to stay ignorant. Smeared is the
current state: `export_rpoc_input()` on every display and `allowed_displays` on every
modality are both connection logic hiding inside generic classes.

`shell/` is that one place and it is allowed to import everything. Naming the promiscuous
module explicitly is the whole trick; it also makes it obvious when it grows too big,
which a smeared version never does.

This also gives a second, independent reason `views/` and `shell/` are separate despite
both being Qt: they have different import permissions. `views/` may not touch `run/`;
`shell/` must. Different permissions means different folders, and Qt never entered into it.

## Hierarchy of folders/desired dependencies and why

```
core         -> (nothing)
devices      -> core
operations   -> core, devices
data         -> core
run          -> core, data, operations, devices
views        -> core, data
recipes      -> core, data, operations, devices, run
session      -> core
shell        -> everything
```

Read top to bottom as "may import." Nothing may import in the other direction.

### The rules that carry the weight

1. **`views/` may not import `run/` or `recipes/`.** This is the display/acquisition
   separation, enforced by the import graph rather than by discipline. A view knows how
   to render a dataset and what context it can offer; it never knows what will consume
   that context.

2. **Nothing may import `recipes/`** except `shell/` (to list them) and
   `recipes/registry.py` (to collect them). No recipe can become load-bearing for another
   part of the system, so any recipe can be deleted outright.

3. **`operations/` may not import `data/`.** Operations return arrays. Turning an array
   into a dataset write is the procedure's job. This keeps operations pure functions,
   testable with no state, no Qt, and no store.

4. **Qt appears only in `views/`, `shell/`, and `devices/*/panel.py`.** Everything else
   imports in a test with no display. That is not true today.

5. **`shell/` is the only module allowed to know everything.** Wiring lives there and
   nowhere else. When it gets fat, that is a signal to look at, not a thing to hide by
   pushing wiring back into `views/` or `recipes/`.

### Consequences worth noticing

- A feature is added by creating one folder under `recipes/` and editing nothing else.
- A recipe is deleted by deleting its folder.
- `run/` never knows what any procedure does; it only knows how to execute one.
- The entire non-visual system (`core`, `devices`, `operations`, `data`, `run`, `recipes`,
  `session`) is importable and testable headless.

## Where the current code lands

| today | new home | notes |
|---|---|---|
| `instruments/` + `instrument_widgets/` | `devices/<name>/` | driver and panel rejoined |
| `modalities/*/acquisition_core.py` | `operations/` | mostly as-is; this is the healthiest code in the repo |
| `modalities/*/parameters.py` | `recipes/<name>/params.py` | schema and dataclass unified into one definition |
| `acquire_once` + `build_continuous_worker` | `recipes/<name>/procedure.py` | the two halves rejoined into one program |
| `modalities/*/storage.py` (x3, near-identical) | `data/io.py` | one copy |
| `modalities/helpers/toy_data.py` | `devices/*/simulated.py` | kills the repeated `except DaqUnavailableError` fallback in every modality |
| `displays/` | `views/` | minus the arrays they currently own |
| `optocontrols/` | split three ways | `core/modulation.py` (the spec), `operations/modulation.py` (TTL generation), `views/mask_editor.py` (the editor) |
| `rpoc/editor.py` | `views/mask_editor.py` | |
| `backend_utils/` | dissolved | `contracts`/`parameter_utils` -> `core/`; the rest into whichever folder actually uses it |
| `services/` | dissolved | instrument -> `devices/registry`; display -> `views/registry`; modality -> `run/runner` + `recipes/registry`; interpreter -> `shell/` binding; session -> `session/` |
| `gui/` | `shell/` | |
| `domain/stores.py` | delete | dead code; only referenced by its own test |

## Design decisions already made

- **The procedure owns the loop.** No base class supplies `while not should_stop`. Frame
  counting, `num_frames`, and the storage hooks disappear from any base class along with it.
- **A running procedure has an inbox.** This is the only structural way for anything
  outside to reach a live acquisition. Clicks are one sender among many.
- **Parameters live in a model; the form reads and writes it.** No widget scraping.
- **Acquired arrays live in datasets, not in widgets.** Data outlives the display that
  showed it, two views can share one dataset, and saving is a dataset operation.
- **Recipe `needs`/view `offers` matching generates context menus.** No hardcoded lists in
  either direction. `allowed_displays` and `export_rpoc_input` both die here.
- **Streams are named and explicitly bound to views**, with shape contracts used to
  validate the binding rather than perform it.
- **Simulation is a device variant**, not a `try/except` inside each acquisition.
- **Session config and acquired data are separate subsystems** with separate formats and
  lifetimes.
- **Masks become datasets plus a recipe parameter**, not a global inventory filtered by
  `allowed_optocontrols`.

## Open questions to settle before or during migration

1. **Click semantics.** Two UX shapes look identical and are mechanically unrelated:
   launching a new run from a right-click, versus feeding a point to a procedure already
   parked on `ctx.wait_for`. Proposed rule: left-click feeds a waiting run (with a visible
   affordance on views that can currently feed it), right-click offers recipes to launch.
   Needs a decision before `shell/menus.py` is written.
2. **Mask UX regression.** Masks-as-datasets removes the dedicated optocontrol manager
   panel in favour of "pick from the dataset library." Confirm that is acceptable.
3. **Hardware contention policy.** `needs = [Galvo, DAQ]` doubles as a lock declaration.
   Proposal: launching an exclusive recipe while another holds the hardware prompts to
   stop the current one rather than silently queueing — queueing is surprising on a
   microscope where the sample is changing.
4. **Coordinate/calibration model.** `point_scan` and `mosaic` both need a real mapping
   between display pixels, sample coordinates, and galvo volts / stage position. Today
   `fast_axis_offset`/`amplitude` are dimensionless volts private to each modality. The
   calibration belongs on the galvo and stage devices, with `core/space.py` holding the
   `Frame` type. This is not needed for the confocal or z-stack migration and can be
   deferred, but it blocks the click-driven and mosaic recipes.
5. **Dataset backing for long runs.** A multi-hour mosaic may exceed memory. Decide
   whether `Dataset` needs on-disk backing (memmap / zarr) or whether in-memory plus
   periodic flush is enough for now.
6. **Naming: `views/` vs `shell/`.** The boundary is real (different import permissions)
   but the names are weak. If it reads ambiguously later, rename `shell/` rather than
   merging them.

## Invariants to verify when the migration is done

Mechanical checks, suitable for a lint script or a test:

- `views/` imports nothing from `run/` or `recipes/`.
- Nothing imports `recipes/` except `shell/` and `recipes/registry.py`.
- `operations/` imports nothing from `data/`.
- `PyQt6` appears only under `views/`, `shell/`, and `devices/*/panel.py`.
- No module imports anything above it in the hierarchy table.
- The test suite runs headless with no Qt application and no hardware.

And the human check, applied to the next feature added: list the folders touched. If the
count tracks how much genuinely *new capability* was added, the structure is working. If
it tracks nothing in particular, it is not.

## Notes for the migration-plan session

- Repository is `pyrpoc/` in this workspace. Current tree is to be replaced wholesale, not
  incrementally patched in place — but the migration itself should be phased so the app
  stays runnable between phases.
- The highest-value existing code to carry over largely unchanged is
  `modalities/*/acquisition_core.py` (the NI-DAQ waveform generation, synchronized task
  setup, and reshaping) and `instruments/time_tagger.py`. Preserve the hardware maths;
  restructure everything around it.
- `tests/` mirrors the new tree one-to-one.
- Suggested phase ordering, to be confirmed and expanded: (1) `core/` vocabulary and the
  parameter model, (2) `run/` plus `Procedure`/`RunContext` with confocal ported as the
  first recipe, (3) `data/` and datasets, moving arrays out of views, (4) `views/` and the
  `needs`/`offers` matching, retiring `allowed_displays` and `export_rpoc_input`, (5)
  remaining recipes, (6) `shell/` and `session/`, (7) delete the old tree.
