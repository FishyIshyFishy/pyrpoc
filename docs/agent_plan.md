# Autonomous microscopy: review checklist

A checklist to hold `docs/plan.md` against. It lists everything the three
phases have to account for, so the plan can be checked for gaps. It is not a
spec: code-level detail comes later, from an agent working off the finished
`plan.md`.

**How to read it:**
- **Decide:** a choice that `plan.md` has to make.
- **Leaning:** where earlier discussion pointed. These can still change.
- Everything else is something to build or account for.

**Ground rules:**
- **No backwards compatibility.** Old recordings, sessions and keys need no
  support.
- **No speculative options.** Every input, field and argument exists because
  something reads it.
- **Words come from `pyrpoc/core/README.md`:** Program, Run, ProgramInputs,
  InputRequest, Data, ProgramOutput, Library, Recording, Panel, DataPanel.
- **Simulation** exists only under `uv run pyrpoc --simulated`, in
  `plugins/programs/simulated/` (and `plugins/devices/simulated/`), registered
  from `register()`.

---

## The throughline: calibration → survey → units → loop

Each step is its own program and its own recording, in one experiment
folder. Each run names the recording it read as its `input`.

| Step | Produces | The next step needs from it |
|---|---|---|
| Galvo Calibration | `um_per_volt` | pixel size in µm |
| Survey | raw tiles, overview, the **dish frame** (dish µm), focus surface, reference sites | where things are, what focus to use, sites to re-anchor and check focus on |
| Units | the units: id, labels, outline in dish µm, when last seen | what to experiment on, and a template for the first tracking |
| Autoloop | log, frames, the units as frozen at start, provenance | n/a (analysis) |

Each handoff has to stay clean:
- **Scan geometry.** Pixel size depends on `um_per_volt`, amplitude and
  pixel count. Outlines kept in dish µm survive a resolution change between
  survey and loop.
- **Time.** Cells move between survey and loop start. Each unit needs its own
  "last seen at" (when its tile was taken, not when the survey started), so
  the first tracking search is sized right.
- **Stage.** The stage may drift, or the dish be re-seated, between survey and
  loop. Re-anchor at loop start.
- **Focus.** The focus surface is valid only over the surveyed area, and only
  for a while.

---

## Phase 1: Programs

The programs, their inputs and the `Data` they produce. The autonomous API
and the experiments that use it. The simulated dish they're developed against.

### 1.1 The programs
- [ ] **Program group** "Full Dish Experimentation" (via `Program.group`).
  Workflow families (calibration, typical imaging, autonomous) may later shape
  the folder layout too.
- [ ] **Galvo Calibration**, which measures `um_per_volt`.
  - Decide: is the result a recording in each experiment folder, or Galvo
    device config? It describes the rig, not the experiment.
  - Decide: how it measures (what it scans, what it fits).
- [ ] **Survey**, which replaces `mosaic`:
  - tiles the region (from the extent, field of view and overlap) and walks
    it as a snake;
  - builds a focus map;
  - publishes raw tiles as they're taken;
  - stitches the overview live, then refines it at the end;
  - picks reference sites;
  - returns to the start position.
  - Decide: how the focus map is built.
    - Leaning: climb only at anchor tiles that pass a content gate, then fit
      a plane, or a quadratic from six or more anchors.
    - Leaning: drop climbs that end at the edge of the search.
  - Decide: stitching. Leaning: NCC refinement plus feathered blending.
  - Decide: how reference sites are picked.
    - Leaning: by content, at least one field apart, not on units.
  - Decide: what the region is (a centred extent, or drawn).
- [ ] **Units**: drawn by the user or found by identification.
  - Decide: drawn units can't become `Data` (only programs make it). Either:
    - they reach the loop as a program input value (like `Mask`); or
    - a "Define Units" program takes them as input and outputs the same
      units recording that Identify Units does, so the chain is uniform.
  - Decide: is identification its own program or part of Autoloop?
    - Leaning: its own program.
  - Decide: the methods it starts with, and how cells group into multi-cell
    units.
    - Leaning: Otsu, then a watershed.
    - Leaning: group by proximity, keep groups of exactly N cells, and report
      how many were dropped.
- [ ] **Autoloop**: one or more closed-loop experiments, each a program built
  on the API (1.4). A new experiment is a new program, not a mode of an
  existing one.
- [ ] **Deleted:**
  - the `mosaic` program and its program inputs;
  - `mosaic_simulation`.

  The `Mosaic` data kind stays for the survey's raw tiles.
- [ ] **Shared code moves** where the import rules allow programs to use it:
  - `stitching.py` (now in the mosaic panel);
  - `climb.py` and `tenengrad.py` (now in autofocus).

### 1.2 Program inputs
- [ ] **Calibration:** whatever its measurement needs.
- [ ] **Survey:**
  - extent and overlap;
  - focus-anchor settings and content gate;
  - the existing scan, DAQ and focus-search inputs.
- [ ] **Identification:** channel, size limits, grouping rules.
- [ ] **The loop engine:** focus-check period, tracking search margin, and
  anything else the orchestrator reads.
- [ ] **Each experiment's own inputs:** treatment line, period, duration,
  frame count, ...
- [ ] **Decide: how a program gets its upstream recordings** (the survey and
  units it reads).
  - Phase 3 answers this through the Experiment folder.
  - Phase 1 either builds that part early, or uses a stopgap that Phase 3
    replaces.
- [ ] **Decide: large program input values.** Unit sets are too big to
  store inline the way `Mask` is. The value needs to be kept beside the
  session and the recording, not inside them.

### 1.3 The `Data` kinds
Anything a panel renders lives in `core/`, since panels and programs never
import each other.
- [ ] **Survey map:** overview image, origin, µm per pixel, focus surface and
  reference sites. The raw tiles stay `Mosaic`.
- [ ] **`Unit`** and the unit set:
  - id and labels (e.g. group);
  - outline in dish µm;
  - current position;
  - last seen at.
- [ ] **The loop's step output:** one published value per step, carrying all
  that its log row and image need. These are kind, unit, routine, series,
  index, times, due time, position, details and image.
  - Designed now, because Phase 3 writes it under the commit rule (3.4).
- [ ] **The loop's live state:** units, the queue and recent events, for
  Phase 2 to render.
  - Small, because it is republished every step.
- [ ] **Curves or summary:** derived results.

### 1.4 The autonomous API
A sizable module in `plugins/programs/` that experiments use or inherit from.
It supplies defaults that an experiment may override: tracking, focus checks
and the unit lifecycle.

- [ ] **Decide: the API's shape.**
  - What an experiment must supply and what it inherits: choosing units,
    assigning groups, routines, the tracking rule, the focus check, the
    lifecycle, the summary.
  - Composition or inheritance.
  - Where the module lives, and which nouns go to `core/`.

**Four concepts, kept strictly apart:**

| Concept | What it is | Changes the unit? |
|---|---|---|
| **Unit** | The thing experimented on: the smallest thing an experiment treats and measures independently (one cell, a pair, a colony) | n/a |
| **Tracking** | Keeping up with where each unit is and what shape it has | **Yes, and nothing else does** |
| **Routine** | Something done *to* a unit: monitor, optical treatment, FLIM, ... | Never |
| **Queue** | The actions that are due | Never |

- [ ] **Units move.** A cell can be 10 µm away after 30 minutes, and the loop
  must keep following it.
- [ ] **Tracking:**
  - finds the unit near where it was last seen, searching a region sized by
    how far it could have moved since;
  - may update the unit's position and outline;
  - every result is recorded, so each trajectory and shape over time is
    data.
  - **It is not a routine.** The orchestrator decides when to track.
    - Default: before a routine step, if the unit hasn't been tracked
      recently. An experiment may override that.
  - Users see it as "tracking" (e.g. "unit 7 tracked: moved 3.1 µm").
  - **The first visit after the survey** searches across the survey → loop
    gap, using the survey crop as its template.
  - Decide: the default method, and the score below which a unit counts as
    lost.
- [ ] **Routines:**
  - act on a unit as it currently is;
  - yield steps (`Scan(...)`, `Wait(600)`, `Record(key, value)`);
  - see only their unit and their own images, never the stage or other
    units.
  - A unit may run several routines at once, e.g. monitor every 30 min plus
    one treatment at 1 h. They share nothing but the unit.
  - Decide: the full yield vocabulary, and what a `Scan` returns to the
    routine. Leaning: its images.
  - **Treatment is not a concept.** A `Scan` may drive digital lines over the
    region it scans; what that means is the experiment's business.
- [ ] **The orchestrator:**
  - runs every (unit, routine) pair;
  - owns the hardware, timing, tracking, focus and saving;
  - runs units in series or in parallel depending on whether routines wait;
    there is no mode switch.
  - Leaning:
    - steps run whole, never interrupted;
    - lateness (start minus due) is recorded;
    - ties are broken by stage distance;
    - it sleeps until the next due time when idle;
    - cancel lands between steps.
  - **Room for live edits.** There is a point between steps where it takes
    in the user's queue edits and added units (from Phase 2).
- [ ] **The queue's model.** The queue lists the actions that are due; the
  log records what was done.
  - Each (unit, routine) shows its next action, because a routine decides
    what comes next only after its current step.
  - Decide: whether fixed-schedule routines may declare future actions up
    front.
  - Experiment logic does not edit the plan mid-run. Only the user does.
- [ ] **Decide: motion vs drift.**
  - Stage drift, focus drift and cell motion all move a unit. Estimate them
    separately, or let tracking absorb all of it?
  - What reference sites are for follows from that.
  - Leaning: re-anchor on the reference sites at loop start, and refuse to
    start if too few register.
- [ ] **Decide: focus during the loop.** Survey surface plus an offset, or a
  refit, from periodic checks at reference sites.
- [ ] **Decide: dose accounting.**
  - Per unit only, or a dose map in dish coordinates?
  - Only the map catches a control unit dosed by a neighbour's whole-field
    scan.
  - Tracking scans cost dose too.
- [ ] **Decide: the unit lifecycle.** What "lost" and "failed" mean, and what
  happens then.
- [ ] **Hardware:**
  - Decide: the API talks to devices directly, or through a `Scope`-like
    protocol with real and simulated versions.
  - ROI scans must play TTL lines over their region. They can't yet.
  - Single-field units only, for now.

### 1.5 First experiments
- [ ] Decide which ones. Earlier candidates:
  - **`bleach_compare`:** repeated ROI scans, with treated units scanned
    with a line on and controls without, then intensity against frame per
    group.
  - **`longitudinal_monitor`:** scan every period for a duration, with an
    optional treatment, then intensity against time per group.
- [ ] Group assignment.
  - Leaning: groups are labels. An experiment assigns the missing ones, and
    the assignment is saved with the loop.

### 1.6 Simulation
- [ ] **A `SimulatedDish` device** with:
  - seeded cells that move and change shape;
  - empty regions;
  - a tilted focal surface;
  - xy and z drift over time;
  - defocus blur and shot noise;
  - bleaching driven by accumulated dose, so a line adds dose;
  - a clock faster than wall time.
- [ ] **Simulated versions** of calibration, survey, identification and each
  experiment, registered from `register()`.

### 1.7 Done when (on the simulated dish)
- [ ] Every survey tile is within one focus step of true focus, and no climb
  happens on an empty tile.
- [ ] The overview's placement error is under 1 px.
- [ ] Identification finds most ground-truth cells, and groups only valid
  multi-cell units.
- [ ] Routines without waits run back to back. Routines that wait interleave
  on time when the microscope can keep up, and report lateness when it can't.
- [ ] Moving cells are tracked to within ~1 µm, and drift is handled as
  decided.
- [ ] Control units get no dose beyond their own scans.
- [ ] A treated curve separates from a control curve.
- [ ] Afterwards, one run on the rig.

---

## Phase 2: Panels and live feedback

How the user sets up, watches and steers an autonomous run. The live channel
is the core of this phase.

### 2.1 The live channel
- [ ] **Loop to panels:** the live-state `Data` (1.3), republished after
  each step through its `ProgramOutput`. Subscribers are called on the worker
  thread and must not touch Qt.
- [ ] **Panels to loop: the open problem.**
  - Today a panel reaches a program only through program inputs (frozen when
    a run starts) or input requests (one question, one answer). Neither
    reaches a run that is already going.
  - Leaning: **live program inputs**, which a run re-reads between steps.
    That fits the flow in `core/README.md` without a new channel.
  - Decide: which inputs are live, how a run notices a change safely across
    threads, and how a live input is told apart from a frozen one.
- [ ] **Edits the user can make:**
  - move an overdue action to the top (the core interaction, e.g. take the
    FLIM image now);
  - add a unit (its routines' first actions join the queue);
  - add a routine to a unit;
  - pause or remove queued actions;
  - Decide: change an action's due time?
- [ ] **Queue edits are not logged.** Only what actually runs is.
- [ ] **One panel for every Autoloop version.**
  - One shared noun that every version uses.
  - Versions differ in what they put in it (routine, series and record
    names), not in its type.
  - The panel shows the contents generically, as a table shows whatever
    columns exist.

### 2.2 The Map panel
- [ ] Shows:
  - the overview, filling in live during the survey;
  - unit outlines;
  - tracked positions during the loop;
  - reference sites.
- [ ] Decide: two sources in one panel. The overview comes from the survey
  and positions from the loop, but a `DataPanel` shows one `ProgramOutput`.
  - Either the loop's state points to or includes the overview, published
    once and not every step;
  - or the Map gets its own way to show two sources.
- [ ] **Drawing units** (if drawn by hand):
  - editable before a loop, read-only during one;
  - Leaning: the Map supplies `Units` to an input request, as the Mask
    Editor supplies `Mask`;
  - share the drawing canvas with the Mask Editor.

### 2.3 The Experiment table
- [ ] **Before a run:** units, labels and group assignment.
- [ ] **During a run:**
  - units with their state, last tracked position and dose;
  - the queue, with overdue actions first;
  - each (unit, routine) pair's next action.
- [ ] Decide: does one kind of `Data` serve both modes?

### 2.4 Linking the Map and the table
- [ ] Clicking a unit in either panel selects it in the other, without
  either panel knowing the other.
- [ ] Decide: this is shared screen state between panels, and nothing in
  `core/` provides it yet.

### 2.5 Analysis and Curves panels
- [ ] **Curves:** renders the summary kind, live during a run.
- [ ] **Analysis:** the timeline of everything that happened ("at time X,
  this happened to unit 7"). Each row links to its images, and a per-unit
  view is a filter on it.
- [ ] Both are separate from the Map and table pair.

### 2.6 Deleted
- [ ] The mosaic panel.

### 2.7 Done when
- [ ] A survey fills in live on the Map.
- [ ] Units can be drawn, edited and handed to a loop.
- [ ] During a simulated loop, moving an overdue action to the top makes it
  run next, and an added unit starts its routines.
- [ ] Selection stays in sync between the Map and the table.

---

## Phase 3: The Experiment, saving and files

The Experiment is the folder runs save into. The Library groups by it,
programs check it before running, and the autonomous files are written into
it.

### 3.1 The Experiment
- [ ] **An experiment is a folder.** The save folder that is set *is* the
  experiment folder, with no new noun in code. A saved output's experiment
  is `meta_path.parent`.
- [ ] **Programs declare what they need from the folder.**
  - Most don't care, and save into whatever folder is set, mixed or not.
  - Those that care refuse to start, and say why. For example:
    - Survey needs a calibration (unless calibration moves to device config);
    - Autoloop needs a survey and units, and no earlier loop.
  - Leaning: checked in `Runs.blockers(program_key)`, so Play is disabled
    with the reason shown.
- [ ] **Decide: how a program finds a recording in the folder.**
  - Filenames are unreliable: users name runs, and repeats get numbered
    (`survey_2`).
  - Leaning: look up `program_key` in each `_meta.json`.
  - Decide: which one wins when there are two.
  - Decide: does a crashed loop count as "an earlier loop"?
- [ ] **Saving is mandatory** for programs that later ones read from.
- [ ] **Every run records its `input`:** the recording it read.

### 3.2 The Data Library
- [ ] **Groups by experiment.** Unsaved outputs fall into one "Unsaved"
  group.
- [ ] **Source dropdowns group the same way.** Each panel already offers only
  the kinds it renders.
- [ ] Decide: does opening a loop recording also open its inputs (survey,
  units)? Is the `input` chain shown, or navigable?
- [ ] **Loop frames load lazily,** within the Library's 1 GiB limit.

### 3.3 The experiment folder on disk
Names are placeholders. See the filenames question in 3.5.

```
<experiment>/
  galvo_cal_meta.json                       um_per_volt (unless it becomes device config)
  survey_meta.json                          input: galvo_cal
  survey_tiles_tiles.json, survey_<ch>.tiff        raw tiles (Mosaic)
  survey_map.json, survey_map_<ch>.tiff            overview, focus surface, reference sites
  units_meta.json, units.json               input: survey; drawn or identified
  loop_meta.json                            input: units
  loop_units.json                           units as frozen at loop start, with assigned labels
  loop_events.csv                           the log
  loop_frames/<series>/unit_NNNN_<ch>.tiff  every image
  loop_summary.npz                          derived
```

### 3.4 The loop recording
Analysis is built around **one timeline**, and the files are shaped to serve
it.

| Part | What | Format |
|---|---|---|
| **Log** | Everything the orchestrator did, in order | `loop_events.csv`, append-only |
| **Units** | The units as frozen at loop start, with assigned labels | `loop_units.json` |
| **Frames** | Every image, grouped by series and unit | multi-page float32 TIFF |
| **Provenance** | Parameters, devices, inputs, seed, version | `loop_meta.json` |

- [ ] **One log for the whole run,** not one per unit. Some events have no
  unit: focus checks, re-anchoring, start and stop.
- [ ] **Log columns:**

  | Column | Meaning |
  |---|---|
  | `seq` | order in the run |
  | `t_s`, `wall_time` | time since loop start, and actual time |
  | `kind` | `scan`, `track`, `record`, `state`, `focus_check`, `reanchor`, `run`, `error` |
  | `unit_id` | empty for global events |
  | `routine` | the routine the step belongs to; empty for tracking and global events |
  | `due_s` | when the step was due; lateness is `t_s - due_s` |
  | `series`, `index` | which image, if any |
  | `x_um`, `y_um` | the unit's tracked position: dish position of the page centre |
  | `details` | JSON for kind-specific fields (stage xyz, tracking score and shift, lines, dose added, reason for a state change) |

- [ ] **CSV,** because appending rows is crash-safe. Convert to Parquet at the
  end only if logs get large.
- [ ] **The commit rule:** a row is written only after its image is on disk.
  - Every row's pointer is then valid, and a crash or cancel leaves whole
    steps.
- [ ] **Series:**
  - A series groups a unit's images by purpose, named by the routine (e.g.
    `bleach`, `context`).
  - Rows refer to images by meaning (`series=bleach, unit=7, index=23`),
    never by filename.
- [ ] **Padding,** so stacks open in ImageJ:
  - every page in a file is the same size, padded with NaN on write;
  - NaN means "not scanned" (padding or outside the outline), so no mask
    files are needed, and `nanmean` is the mean over what was scanned;
  - each page is centred on the unit's tracked position, so in ImageJ the
    cell sits still;
  - each row's `x_um, y_um` is its page centre.
- [ ] **Growing the box:**
  - It starts at the unit's minimum bounding box.
  - When an image doesn't fit, rewrite that unit's file with every page
    re-padded, then swap it in with one atomic rename. A crash leaves the old
    file intact.
  - Files are rewritten only while recording, and never after it ends.
  - Decide: on Windows, a file open in ImageJ blocks both the rename and
    appending. What does the loop do then?
- [ ] **Measurements:**
  - raw images are always saved: they are the ground truth;
  - values a routine decides on become `record` rows;
  - derived results (curves, per-group summaries) are a convenience, and
    can always be recomputed from the log and frames.

### 3.5 Saving changes
- [ ] **The writer model.** Today there's one writer per output, chosen by
  `Data` kind, writing `<root>_<output>…`.
  - The loop needs a CSV plus per-series, per-unit TIFFs, under the commit
    rule.
  - Leaning: each step is one published value whose writer writes the image,
    then the row.
- [ ] **Decide: filenames.** The fixed names in 3.3 conflict with user-set
  `SaveTarget` names inside an experiment folder.
- [ ] **Recording format 3** adds `input`. The format-1 decoder kept in
  phase 0 can go.
- [ ] **Survey map file:** overview TIFF plus JSON (origin, µm per pixel,
  focus surface, reference sites).
- [ ] **Units file:** outlines encoded like `Mask` (bounding box plus packed
  bits), with labels. Units may overlap, so one label image won't do.
- [ ] **No custom format.** CSV, TIFF and JSON open in ImageJ, pandas and
  napari with no extra work, and every format has to keep loading in later
  versions.

### 3.6 Reading it back
- [ ] **A no-Qt reader** for notebooks and the Analysis panel: log, units,
  and frames by series and unit.
- [ ] Decide: where it lives.

### 3.7 Done when
- [ ] Each program refuses to start in a folder missing what it needs, and
  says why.
- [ ] The Library groups a full simulated calibration → survey → units → loop
  run as one experiment.
- [ ] A cancelled or crashed loop leaves a log whose every row points to an
  image on disk.
- [ ] Every unit's stack opens in ImageJ with the cell centred.
- [ ] The summary can be recomputed from the log and frames alone.

---

## Dependencies between phases
1. **Phase 1 reads upstream recordings before Phase 3 builds the
   Experiment.** Either build the folder-reading part early, or use a
   stopgap.
2. **Phase 1's orchestrator leaves room for Phase 2's edits:** a point
   between steps where it reads them.
3. **Phase 1's step output decides what Phase 3 can write.** It has to carry
   everything the row and the padded page need.

## Later
- Resume after a crash.
- Periodic re-survey during a long loop.
- Units larger than one field.
- A barrier step ("wait until every unit reaches here").
- A schedule estimate before starting.
- Cellpose for identification.
