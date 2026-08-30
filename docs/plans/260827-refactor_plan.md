# pyrpoc redesign

Design document for a from-scratch rebuild of the pyrpoc backend. Scoped deliberately to what exists in the codebase today: confocal, split confocal, and FLIM acquisition; DAQ/galvo and TimeTagger hardware; four displays; RPOC masks. Section 13 lists what was considered and left out, so the omissions are visible without cluttering the rest.

## 1. Why
Currently the base class `pyrpoc.modalities.base_modality` owns the loop:

```python
while not should_stop: acquire_once()
```

Every workflow must therefore be some form of looped acquisition. A modality is a function call with a kill switch, not an extensible program. The cost is already visible: because `acquire_once()` must be self-contained, `FlimModality` calls `setup_tagger()` and `teardown_tagger()` on **every single frame** — creating and freeing the TimeTagger once per frame — since there is nowhere to put per-run setup.

Additionally, the UI owns all parameters. `gui.main_widgets.acquisition_mgr.collect_values()` scrapes widgets for parameter values. State should live in a model, and the UI should be a view that reads and writes it. Because the form *is* the state, nothing except the form can parameterise an acquisition.

There is also no place for feature logic that spans acquisition and display. This is why `export_rpoc_input()` exists on every display and why every modality carries a list of allowed displays. Both are necessary functionality filed inside abstract display/modality classes because there was no third place to put them.

Two further problems, discovered while designing the replacement:

Acquired arrays live inside display widgets (`self._data_chw` in `tiled_2d_display.py` and its equivalents). That array is the original, not a cache. Closing a display destroys the data; two displays over one run keep two drifting copies; and `export_rpoc_input()` exists only because a widget is the sole owner of pixels.

Storage is duplicated per modality. `confocal/storage.py` and `split_confocal/storage.py` are the same file plus an auxiliary-payload block, and that block exists only because split-confocal produces a second output stream and there was no way to declare one.

## 2. What is settled
1. **The loop belongs to the experiment, not to a base class.** Whoever writes the experiment writes its control flow, including per-run setup and teardown.
2. **Acquired data cannot live in widgets.** It must outlive the display that showed it and be shareable between displays.
3. **Parameters cannot live in widgets.** They live in a model; the form is a view onto it.
4. **Bounded hardware actions are separable from the program that calls them.** The scan is a function; the experiment is a program that calls it.
5. **Views and acquisitions must not reference each other.** Something else connects them, and that something has one identifiable home.

## 3. The definition test
One design tool, used repeatedly below and worth keeping afterwards
```
Can you state what a thing is in one sentence without listing its fields? And does a new requirement get handled by *using* it, or by *extending* it?
```

An abstraction that gains a field every time a new question is asked is not abstracting anything — its shape is set by whichever question was asked last. That is what happened to `BaseModality` (`emitted_kinds`, `allowed_displays`, `get_frame_limit`, `prepare_acquisition_storage`, ... each added for one caller), and an earlier draft of this document repeated the mistake with a "Recipe" type that accumulated a field per feature discussed. Both were deleted.

Every concept in section 4 passes this test. If one of them starts growing a field per conversation, it has the same disease.

## 4. Core concepts

**Program** — something with a `run()` that drives hardware or pseudo-hardware and emits data over time. This is what a "modality" should have been. Its attributes are not a declaration format; they are the three things the runner must know in order to start it. Note how there is nothing about labels, menus, or how it was launched, because a program should not know it is in a dropdown.

```python
# programs/confocal.py
class Confocal(Program):
    uses   = [Galvo, DAQ]            # what to claim
    params = ConfocalParams          # what to configure
    emits  = {"intensity": Image2D}  # what datasets to create

    def run(self, ctx):
        p = ctx.params
        for i in range(p.num_frames):
            ctx.status(f"frame {i+1}/{p.num_frames}")
            ctx.publish("intensity", raster_scan(**p.scan, **p.daq))
            ctx.check_cancel()
```

**RunContext (`ctx`)** — the service surface handed to a running program. Deliberately small.
```python
ctx.params                            # the parameter object for this run
ctx.devices                           # resolved device handles
ctx.publish(stream, data, **coords)   # write into one of this run's datasets
ctx.status(text)                      # progress -> run bar
ctx.check_cancel()                    # raises Cancelled
```

It is one concrete class in `run/`, written once and never subclassed. Programs call it; they never implement it. It is the same idea as today's `acquire_continuous(on_frame, frame_limit, should_stop, on_error, on_finished)` — those five loose arguments already form a run context — given a name and one place to live.

| `ctx` member | today's equivalent |
|---|---|
| `ctx.params` | `self.parameters` |
| `ctx.devices` | the dict passed to `load_instruments()` |
| `ctx.publish(...)` | the `on_data` / `on_frame` callback |
| `ctx.check_cancel()` | `should_stop()` |
| `ctx.status(...)` | — (the widget sets its own status label) |

`ctx.publish` buys little over `on_frame` on its own; the leverage is that the stream name is declared in `emits`, so a view binding exists before the run starts rather than being inferred from a `kind` tag mid-flight. That is what makes FLIM's two outputs addressable separately.

**Operation** — one bounded hardware action. A plain function: explicit arguments in, arrays out. No `self`, no loop, no saving, no Qt, no knowledge of what it is used for. The unit is a *clock domain*, not a device — the raster operation drives AO + AI + DO as one synchronised NI task because they share a sample clock. Splitting it into separate "positioner" and "detector" objects would misrepresent the hardware.

Operations come in **two forms**, and the distinction matters more than it looks:

```python
def raster_scan(...) -> np.ndarray:              # returns one complete frame
    ...

def raster_scan_streaming(..., chunk_rows=16):   # yields progressively complete frames
    for chunk in read_in_chunks(...):
        yield reshape_partial(...)
```

An operation that produces data incrementally is a **generator, never a callback taker**. Handing an operation an `on_partial=` callback would invert control again — the program's logic ends up executing inside the operation's stack frame, stopping early becomes awkward, and every new emission point has to be anticipated as another callback parameter. A generator lets the program pull, decide what to do with each yield, and break out. Two separate functions rather than one with a flag, so programs that do not want partial frames pay nothing.

This — not `ctx.publish` — is what would make progressive display work. Today `DataKind.PARTIAL_FRAME` is defined, three displays accept it, and `streamed_image_display.py:97` handles it, but **nothing emits it**, because `run_daq` performs one blocking `ai_task.read(number_of_samples_per_channel=total_samples)` and the partial data does not exist. No amount of callback plumbing fixes that; the read has to become incremental.

**Device** — an addressable piece of the instrument, with configuration, calibration, a panel, and persistence. Two properties vary:

```python
class DAQ(Device):
    owns_connection = True     # an NI resource with open/close

class TimeTagger(Device):
    owns_connection = True     # its own SDK and USB handle

class Galvo(Device):
    backed_by = DAQ            # no connection of its own
    # fast_ao, slow_ao, per-axis limits
```

A galvo has no driver — it is mirrors moved by voltages on someone else's AO channels. But it has a wiring config you set up once and reuse, so it needs identity, a panel, and persistence. Separating *identity* from *connection* gives it those without pretending it opens a port. Today the galvo is invisible: `required_instruments` is empty for confocal and `fast_axis_ao` / `slow_axis_ao` / `active_ai_channels` are loose per-modality parameters, duplicated across all three modalities.

**Claims propagate up `backed_by`.** Claiming the galvo claims its DAQ. Today only one modality runs at a time so this is trivial, but it is where the existing `validate_required_instruments()` logic lands.

The galvo is *data* consumed by operations that know they are driving NI AO. It must not grow a `move()` method that pretends the raster could call it polymorphically.

**Dataset** — where **acquired** arrays live: a run output, with incremental writes, change notification, provenance (which program, which parameters, when), and a save policy. Views render datasets; they never own arrays. A multi-frame run is one dataset that grows, which is why frame counting leaves the program entirely.

The **dataset library** is the collection of these — the run outputs currently open in the app, like a list of open documents.

**Presets are not datasets.** A mask is authored, not acquired: you draw it once, name it, save it, and load it into runs for months. It has no provenance and no run that produced it. Acquired data is an *output*; a preset is an *input*. Filing both in the library means a few masks hidden among hundreds of acquisition results, so presets are plain files referenced by a path parameter, and the mask editor reads an acquired dataset and writes a file. The asymmetry is deliberate.

**View** — renders one or more streams from datasets, emits interaction events. Never references a program.

```python
class Tiled2D(View):
    renders = [Image2D]
```

**Parameter model** — a dataclass holding the authoritative values, with the form generated from its field metadata and writing back into it. Parameters are organised into **shared groups** — `ScanGroup`, `DaqGroup`, `SaveGroup`, `ModulationGroup` — referenced by several programs. Confocal, split confocal, and FLIM currently redeclare the same nine scan fields and five DAQ fields in three separate `parameters.py` files; shared groups mean one definition and one configured value. This is the existing `parameter_groups` idea, promoted from a form-layout hint to the actual model.

**Catalog entry** — a way to launch a program. This is presentation data and lives in the shell:

```python
# shell/catalog.py
Entry(Confocal,      label="Confocal",       group="Imaging")
Entry(SplitConfocal, label="Split Confocal", group="Imaging")
Entry(FLIM,          label="FLIM",           group="Imaging")
```

The selector's contents are curated by hand, because what belongs in a dropdown is a design decision. Keeping labels here rather than on the program is what lets one program be offered more than once later, and keeps `Program` from growing presentation fields.

**What is deliberately not a concept:** there is no "modality" type, no "recipe" type, and no "feature" type. Those name categories, not things. A view is not a program. A device is not a program. An operation is not a program — it returns, it has no lifecycle and no claims.

## 5. File layout

```
pyrpoc/
├── app.py              entry point; builds the app, wires top-level pieces
│
├── core/               shared vocabulary. no Qt, no hardware, no I/O.
│   ├── streams.py      shape contracts (Image2D, Cube3D, Trace1D)
│   ├── params.py       parameter field types, shared groups, validation, coercion
│   ├── modulation.py   MaskBinding (mask file path + port/line); load/save of mask files
│   └── errors.py
│
├── devices/            one folder per addressable thing; driver and panel together
│   ├── base.py         Device; owns_connection / backed_by
│   ├── registry.py
│   ├── daq/
│   │   ├── device.py
│   │   ├── simulated.py
│   │   └── panel.py
│   ├── galvo/          AO channel assignment and limits
│   ├── time_tagger/
│   └── stage/          prior, zaber
│
├── operations/         bounded hardware actions. plain functions. no self, no loop.
│   ├── raster.py       waveform gen + synchronised AO/AI/DO scan
│   ├── split_raster.py t0/t1 gated variant; different DO gating and sample splitting
│   ├── tagger.py       FLIM scan trigger + histogram frame read
│   └── modulation.py   MaskBinding + loaded mask array -> per-pixel TTL waveform
│
├── data/
│   ├── dataset.py      array + incremental writes + change notification + save policy
│   ├── library.py      the live collection of datasets, with identity
│   ├── io.py           save/load: tiff, npz
│   └── transforms.py   normalise, project, slice — shared by views and programs
│
├── run/                executes programs
│   ├── program.py      Program base + RunContext
│   ├── runner.py       worker thread, cancellation, status, dataset setup
│   └── claims.py       device claims; propagation along backed_by
│
├── programs/           one file per experiment
│   ├── confocal.py
│   ├── split_confocal.py
│   └── flim.py
│
├── views/              render datasets, emit interaction events
│   ├── base.py         View; renders = [...]
│   ├── registry.py
│   ├── image_2d.py     (was tiled_2d)
│   ├── streamed.py
│   ├── overlay.py      (was multichan_overlay)
│   ├── decay.py        (was flim_display)
│   └── mask_editor.py  (was rpoc/editor.py)
│
├── shell/              application chrome; the only module that may know everything
│   ├── window.py
│   ├── catalog.py      ways to launch programs: label, group
│   ├── launcher.py     the selector, built from catalog entries
│   ├── param_form.py   generic form generated from a params model
│   ├── run_bar.py      what is running, what step, stop
│   ├── docking.py
│   └── theme/
│
└── session/            config persistence only
    ├── state.py        what exists, how it is configured, layout
    └── store.py        JSON read/write
```

## 6. Derivation of the layout

A folder structure does exactly two jobs: it tells you where a new thing goes, and it constrains what may depend on what. The second is the one usually forgotten, and the one that decides whether the codebase stays workable — folders are the only lightweight mechanism available for saying "this part may not know about that part." If folders do not encode dependency constraints they are filing cabinets, and filing does not prevent tangling.

### 6.1 Things that change together live together

The test is not "are they similar" but **when you make a typical change, do you touch both?**

This is where a `gui/` folder fails. The time tagger's driver and its settings panel change together constantly — add a trigger-voltage parameter to the device, add a field to the panel, same sitting, same reason. The time tagger's panel and the tiled image display never change together. But a `gui/` folder files the second pair as siblings and separates the first, because it groups by which library a file imports, and "imports PyQt6" is not a thing that changes together.

The cost is visible today. Adding FLIM touched five folders: `modalities/flim/`, `instruments/time_tagger.py`, `instruments/instrument_widgets/time_tagger_widget.py`, `displays/flim_display.py`, and new entries in `backend_utils/acquired_data.py`. Five places to read to understand it, five to hunt to remove it, and nothing tells you when you have found them all.

So: group by subject, not technology. There is no `gui/` folder because "is Qt" was never the thing that mattered.

### 6.2 Some things serve one feature, some serve many

That rule alone would put everything in feature folders, which does not work. A 2D image view is not owned by confocal; the raster waveform generator is used by confocal, split confocal, and FLIM alike.

Second question, applied to every file: **does this serve one thing or many?**

- Serves one -> lives inside that thing's folder (the time tagger's panel).
- Serves many -> lives in a folder of its kind, alongside interchangeable siblings (`views/`, `operations/`).

This is why the tree is mixed: `devices/time_tagger/panel.py` contains Qt while `views/` is a separate folder that is entirely Qt. Forced by the rule, not inconsistent — the panel serves exactly one device, the image view serves every program.

### 6.3 Dependency order comes from what changes most

The thing changed constantly is **programs**, so programs sit at the bottom with everything else ignorant of them: *nothing imports `programs/`.* If any part depended on a specific program, that program could never be deleted and the next would have to imitate it.

The thing changed least is the **shared vocabulary** — the nouns two folders both need to say. Those sit at the top importing nothing. When two folders need the same word, the word cannot live in either, or one starts importing the other for no reason.

The middle follows physical reality: hardware exists regardless of what you do with it, so `devices` depends only on vocabulary. An action on hardware needs the hardware, so `operations` depends on `devices`. Data exists regardless of how it was acquired, so `data` depends only on vocabulary — which is why saving is not a program's job. Running a program needs actions and somewhere to put results, so `run` depends on both.

Views render data, so they need `data` and vocabulary and **nothing else should be permitted**. Settled statement 5 stops being a principle to remember and becomes `views/` being unable to import `run/` or `programs/`.

### 6.4 `core/` is the folder to distrust

It exists for one reason: two folders need to say the same word. That justification is narrow, and the folder rots because "shared" is slippery. `backend_utils/` became exactly this — `array_contracts`, `state_helpers`, `parameter_utils`, `registry`, `contracts` landed there not because many folders needed them but because they were not obviously GUI. "Not GUI" is not a subject.

One test keeps it honest: **if only one folder imports it, it does not belong in `core/`.**

### 6.5 Two kinds of state, split by lifetime

Configuration is small, JSON, and exists so the workbench returns on relaunch. Acquired data is large, lives in TIFF, and exists because it is the experimental result — opened as a file months later, possibly in another program. Different size, format, lifetime, and reason to exist. Merge them and the session file tries to hold arrays, or the TIFFs try to remember dock layout.

### 6.6 One place is allowed to know everything

Ignorant parts still have to be connected. The connecting code has to exist somewhere, and the only choice is whether it lives in one identifiable place or smeared across the parts meant to stay ignorant. Smeared is the current state: `export_rpoc_input()` on every display and `allowed_displays` on every modality are both connection logic hiding inside generic classes.

`shell/` is that place and may import everything. Naming the promiscuous module explicitly is the trick; it also makes it obvious when it grows too big, which a smeared version never does. This is also why `shell/catalog.py` is the right home for launch presentation — a presentation layer is *supposed* to be an enumerated pile of design decisions.

It gives a second, independent reason `views/` and `shell/` are separate despite both being Qt: different import permissions. `views/` may not touch `run/`; `shell/` must.

## 7. Dependency hierarchy

```
core         -> (nothing)
devices      -> core
operations   -> core, devices
data         -> core
run          -> core, data, operations, devices
views        -> core, data
programs     -> core, data, operations, devices, run
session      -> core
shell        -> everything
```

Read as "may import." Nothing may import in the other direction.

### Rules that carry the weight

1. **`views/` may not import `run/` or `programs/`.** The display/acquisition separation, enforced by the import graph rather than by discipline.
2. **Nothing may import `programs/`** except `shell/` (to launch them) and a registry (to collect them). Any program can be deleted outright.
3. **`operations/` may not import `data/`.** Operations return arrays; turning an array into a dataset write is the program's job. Keeps operations pure and testable with no state.
4. **Qt appears only in `views/`, `shell/`, and `devices/*/panel.py`.**
5. **`shell/` is the only module allowed to know everything.** When it gets fat, that is a signal to examine, not a thing to hide by pushing wiring back into `views/` or `programs/`.

### Consequences

- A new experiment is one file in `programs/` plus one row in `shell/catalog.py`.
- Deleting an experiment is deleting that file and that row.
- `run/` never knows what any program does; it only knows how to execute one.
- Everything outside `views/` and `shell/` is importable and testable headless.

## 8. Worked examples

All four exist in the codebase today.

### 8.1 Confocal

**User:** pick "Confocal" from the selector. The parameter panel fills with the Scan, DAQ and Save groups. Type a filename. An image view opens because the program declares it emits one. Hit play; the run bar reads "frame 3/10". It stops on its own or you stop it. The dataset remains in the library afterwards.

**Code:** `programs/confocal.py` as shown in section 4.

Note what is absent: **no saving**. The runner reads `emits` plus the Save group and creates the dataset with a save policy before calling `run()`. Frames are written as they are published. This is why all three `storage.py` files collapse into `data/io.py`, and why `prepare_acquisition_storage` / `save_acquired_frame` / `finalize_acquisition_storage` disappear from every program.

Also absent: frame counting. `num_frames` is a number the program loops over, so `get_frame_limit()`, `_run_frame_limit`, `_saved_frame_count`, and `_frames_emitted` all go away.

### 8.2 Confocal with a preset mask

**User:** draw a mask in the mask editor from an existing image; it saves as a file. In the **Modulation** group of the parameter panel, a small table: mask file | port | line. Pick the file, set port 0 line 3. Hit play. Because Modulation is a shared group, Split Confocal and FLIM see the same setting without re-picking.

```python
# core/modulation.py
@dataclass(frozen=True)
class MaskBinding:
    path: Path      # a .tif/.npy the user drew and saved
    port: int
    line: int
```

```python
# programs/confocal.py — run() gains two lines
    ttl = mask_ttl(p.modulation, scan=p.scan, device=p.daq.device_name)
    for i in range(p.num_frames):
        ctx.publish("intensity", raster_scan(**p.scan, **p.daq, ttl=ttl))
```

`mask_ttl` is the existing `preprocess_mask_to_scan_grid` + `generate_mask_ttl_signals`, taking loaded mask arrays instead of `MaskContext` objects.

**What disappears:** `BaseOptoControl`, `BaseOptoControlWidget`, the optocontrol registry and manager panel, `prepare_for_acquisition`, `get_context`, `MaskContext`, `extract_mask_contexts`, `allowed_optocontrols`, and the `optocontrols` list in `AppState`. Roughly 700 lines become a frozen dataclass, a parameter field, and a pure function.

**Tradeoff to accept:** masks stop being globally toggleable objects and become run parameters. You lose the enable/disable card. You gain that a run's parameters fully describe what happened — saved metadata records exactly which masks on which lines, which today it does not, because the enabled flag lived outside the modality.

### 8.3 Split confocal

**User:** "Split Confocal" sits next to "Confocal" in the selector. Same Scan / DAQ / Save / Modulation groups plus a Split group with `t0_samples` and `t1_samples`. It emits two streams, so a second view can be opened for the raw pixel stream — or ignored, and it is still saved.

```python
# programs/split_confocal.py
class SplitConfocal(Program):
    uses   = [Galvo, DAQ]
    params = SplitConfocalParams        # ConfocalParams + t0_samples, t1_samples
    emits  = {"intensity": Image2D, "raw_pixel_stream": Image2D}

    def run(self, ctx):
        p = ctx.params
        ttl = split_mask_ttl(p.modulation, scan=p.scan, t0_samples=p.t0_samples,
                             device=p.daq.device_name)
        for _ in range(p.num_frames):
            split, raw = split_raster_scan(**p.scan, **p.daq, ttl=ttl,
                                           t0_samples=p.t0_samples, t1_samples=p.t1_samples)
            ctx.publish("intensity", split, channels=p.split_channel_labels())
            ctx.publish("raw_pixel_stream", raw)
            ctx.check_cancel()
```

Two points, because this is the case that motivated the redesign:

**The `run()` body duplicates confocal's shape, and that is correct.** Do not give them a shared base class. Ten lines of duplicated orchestration is far cheaper than a template method with a mode flag — that is the trap `BaseModality` fell into, and the second flag is always worse than the first. Sharing happens where code is genuinely identical: `generate_raster_waveform` lives in `operations/` and both call it. Where they genuinely differ — DO gating on `t0_samples`, splitting samples into alternating t0/t2 channels — they call different operations.

**The auxiliary payload machinery dies.** Today the raw stream travels as `_pending_auxiliary["raw_pixel_stream"]`, picked up by `append_auxiliary_payload` and `flush_auxiliary_payloads` into an npz side-channel invisible to displays. It is a second output smuggled through storage because there was no way to declare one. Here it is a declared stream: viewable, bindable to its own view, saved by the same path as everything else.

### 8.4 FLIM

**User:** pick "FLIM". Greyed out with "needs TimeTagger" if none is configured. Two streams, so an image view and a decay view both open. Hit play.

```python
# programs/flim.py
class FLIM(Program):
    uses   = [Galvo, DAQ, TimeTagger]
    params = FlimParams
    emits  = {"intensity": Image2D, "histogram": Cube3D}

    def run(self, ctx):
        p = ctx.params
        tagger = ctx.devices[TimeTagger]
        flim = tagger.start_flim_measurement(p)        # once, before the loop
        try:
            for _ in range(p.num_frames):
                flim_scan(**p.scan, **p.daq, **p.triggers)
                ctx.sleep(p.frame_settle_s)
                cube = read_flim_frame(flim, **p.histogram)
                ctx.publish("histogram", cube)
                ctx.publish("intensity", flim_intensity(cube))
                ctx.check_cancel()
        finally:
            tagger.stop_flim_measurement(flim)          # once, after
```

This is the clearest illustration of settled statement 1. Today `FlimModality.acquire_once` calls `setup_tagger()` at the top and `teardown_tagger()` in a `finally` — **per frame** — because `acquire_once` must be self-contained and there is nowhere else for per-run setup to go. A ten-frame FLIM run creates and frees the TimeTagger ten times. Once the program owns the loop, setup is simply outside it.

It also shows why streams are named rather than tagged. Today the two outputs are distinguished by minting `DataKind.FLIM_RAW_FRAME`, which is a stream name wearing a type's clothing — hence `DataKind` splitting into shape contracts (`core/streams.py`) and stream names (keys in `emits`).

## 9. Where the current code lands

| today | new home | notes |
|---|---|---|
| `instruments/` + `instrument_widgets/` | `devices/<name>/` | driver and panel rejoined |
| galvo/AI channels (loose params today) | `devices/galvo/` | gains identity, panel; stops being redeclared per modality |
| `modalities/*/acquisition_core.py` | `operations/` | mostly as-is; the healthiest code in the repo |
| `modalities/*/parameters.py` | `core/params.py` groups + `programs/*` | schema and dataclass unified; scan/DAQ/save groups shared across all three |
| `acquire_once` + `build_continuous_worker` | `programs/*.py` | the two halves rejoined into one program |
| `modalities/*/storage.py` (x3) | `data/io.py` | one copy; save policy is a dataset property |
| `modalities/helpers/toy_data.py` | `devices/*/simulated.py` | kills the repeated `except DaqUnavailableError` fallback in all three modalities |
| `displays/` | `views/` | minus the arrays they currently own |
| `optocontrols/` | split three ways | `core/modulation.py`, `operations/modulation.py`, `views/mask_editor.py` |
| `rpoc/editor.py`, `rpoc/segmentation_methods.py` | `views/mask_editor.py` | mask authoring reads a dataset, writes a file |
| `backend_utils/` | dissolved | `contracts`/`parameter_utils` -> `core/`; the rest into whichever folder uses it |
| `services/` | dissolved | instrument -> `devices/registry`; display -> `views/registry`; modality -> `run/runner`; interpreter -> `shell/`; session -> `session/` |
| `gui/` | `shell/` | plus the new `catalog.py` |
| `domain/stores.py` | delete | dead code; referenced only by its own test |
| `DataKind` | split | shape contract (`core/streams.py`) vs stream name (a key in `emits`); the two `*_PARTIAL_*` members are dead today and become ordinary streams if streaming reads land |

## 10. Decisions made

- **The program owns the loop.** No base class supplies `while not should_stop`. Frame counting, `num_frames`, per-run setup/teardown, and the storage hooks leave the base class with it.
- **Parameters live in a model, organised into shared groups.** The form reads and writes it.
- **Acquired arrays live in datasets.** Data outlives its display, two views can share one dataset, and saving is a dataset property configured by the runner from `emits` + the Save group.
- **Launch presentation lives in `shell/catalog.py`,** not on programs.
- **Streams are named in `emits` and carry shape contracts.** Contracts validate a view binding; names perform it.
- **Devices separate identity from connection.** `owns_connection` and `backed_by`; claims propagate along `backed_by`.
- **Simulation is a device variant,** not a `try/except` inside each acquisition.
- **Session config and acquired data are separate subsystems.**
- **Masks are files referenced by a parameter,** not a global inventory filtered by `allowed_optocontrols` and not entries in the dataset library. Presets are authored inputs; datasets are run outputs.
- **Operations that produce data incrementally are generators, not callback takers.**
- **Similar programs duplicate their `run()` rather than sharing a base class.** Sharing happens at the operation level, where the code is genuinely identical.

## 11. Open questions

1. **Streaming reads.** Decided: For now, do not try to implement any streaming, but ensure the architecture will be extensible to that. 
2. **Are claims needed yet?** Decided: only one modality runs at a time today. But make sure the software is extensible. 
3. **What `mask_editor` is.** Decided: it authors a preset from an acquired dataset. That makes it a view with a save action. The segmentation code in `rpoc/segmentation_methods.py` is depcrecated and can be fully removed. 
4. **Naming: `views/` vs `shell/`.** Decided: The boundary is real (different import permissions) but the names are weak. Keep views/ and shell/, and put the rule in each __init__.py docstring so the name doesn't have to carry it: """Renders datasets. Must not import run/ or programs/.""" One line, and it's where anyone adding a file will see it.

## 12. Invariants to verify

Mechanical, suitable for a lint script or test:

- `views/` imports nothing from `run/` or `programs/`.
- Nothing imports `programs/` except `shell/` and the program registry.
- `operations/` imports nothing from `data/`.
- `PyQt6` appears only under `views/`, `shell/`, and `devices/*/panel.py`.
- No module imports anything above it in the section 7 table.
- The test suite runs headless with no Qt application and no hardware.
- No `Program` subclass defines an attribute outside `uses`, `params`, `emits`, `run`.

And the human check, applied to the next feature: list the folders touched. If the count tracks how much genuinely *new capability* was added, the structure is working.

## 13. Deliberately out of scope

Considered during design and cut, because nothing in the codebase needs them yet. Recorded so they are visibly omissions rather than oversights, and so the hook each one would attach to is known.

**An inbox on the running program** (`ctx.wait_for(Type)`). Today the only signal that can enter a running acquisition is the stop flag — the "kill switch" half of section 1 is not fixed by this document. The hook is one queue on `RunContext` plus the runner recording what type the program is blocked on. Needed by anything where input arrives *during* a run: click-driven acquisition, live parameter changes, hardware triggers, a scheduler.

**Context types and generated context menus** (`Point`, `Region`, and a `View.offers` declaration matched against catalog entries). This is the un-hardcoded replacement for `allowed_displays` in the display->acquisition direction. Depends on coordinate frames to be useful.

**Nested runs** (`ctx.run_sub`). Needed by any outer loop that wraps an inner acquisition — z-stacks, mosaics, time-lapse. Without it, `z_stack_flim` and `mosaic_flim` become separate programs and the combinatorial explosion returns one level up. This is the one omission that would change what `run/runner.py` has to be, so check it before the runner is considered finished.

**On-disk dataset backing** (memmap/zarr). Needed when a single run exceeds memory. Today the largest run is `num_frames` full frames held in RAM, which is fine.

## 14. Notes for the migration-plan session

- Repository is `pyrpoc/` in this workspace. The current tree is replaced wholesale, but migration should be phased so the app stays runnable between phases.
- Highest-value code to carry over largely unchanged: `modalities/*/acquisition_core.py` (NI-DAQ waveform generation, synchronised task setup, reshaping) and `instruments/time_tagger.py`. Preserve the hardware maths; restructure around it.
- `tests/` mirrors the new tree one-to-one.
- Suggested phase ordering, to be confirmed and expanded: (1) `core/` vocabulary and the parameter model with shared groups; (2) `run/` plus `Program`/`RunContext`, with confocal ported as the first program; (3) `data/` and datasets, moving arrays out of displays; (4) `views/` and `shell/catalog.py`, retiring `allowed_displays` and `export_rpoc_input`; (5) split confocal and FLIM; (6) `shell/` and `session/`; (7) delete the old tree.
- Section 2 is the acceptance criteria. Section 3 is the tool to apply if any new concept starts accumulating fields during implementation. Section 13 is the list to re-read before declaring `run/` finished.
