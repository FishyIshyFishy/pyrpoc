# Closed-loop dish imaging: plan

The source is `docs/plan.md`, plus design discussions with the user on
2026-11-02 and 2026-11-03. Open questions are also tracked on PR #53.

This file is still a **planning document**. Section 1 is settled, section 2 is
the data on disk (settled), and section 3 is still open. Implementation
details and phases come after section 3 is resolved, so do not implement from
this file yet.

**Ground rules for this work:**
- **No backwards compatibility.** Old recordings, old sessions and removed
  keys need no support. Delete freely; change the recording format freely and
  update `docs/internal/data-format.md` to match.
- **No speculative options.** Every parameter, field and argument exists
  because something reads it. If something seems missing, ask.

---

## 1. Settled

### 1.1 The framing: an autonomous microscopy API
- The closed loop is built as a reusable **autonomous microscopy API**: a
  sizable module inside `plugins/programs/`. Specific experiments use it, or
  inherit from it.
- The API supplies **default methods** that an experiment may override:
  - **tracking**: when and how units are tracked (1.4);
  - **focus checks** at reference sites;
  - **unit lifecycle**: what "lost" and "failed" mean, and what happens then.
- Experiments are programs. A new experiment is a new program built on the
  API, not a new mode of an existing one.

### 1.2 Programs and the menu
- New program group **"Full Dish Experimentation"**, using the existing
  `Program.group`. It holds:
  - **Galvo Calibration**: measures `um_per_volt`;
  - **Survey**;
  - **Autoloop**: the closed-loop experiments, in one or more versions.
- Specific names don't matter yet. Workflow families (calibration, typical
  imaging, autonomous) may later be reflected in the folder layout as well.

### 1.3 Two phases: survey, then loop
- **Survey** does all the setup the loop needs:
  - tiles the region;
  - builds a focus map;
  - stitches an overview;
  - picks reference sites.
- Its real product is the **dish frame**: a coordinate system in dish µm,
  with a focus surface on it.
- **Loop** runs **routines** on **units**, as **tracking** keeps the units
  current (1.4, 1.5).
- They are separate programs and separate recordings in one experiment folder
  (1.7). Each run names the recording it read as its `input`, so the chain is
  explicit: calibration → survey → units → loop.

### 1.4 Units and tracking
These four concepts are kept strictly apart:

| Concept | What it is | Changes the unit? |
|---|---|---|
| **Unit** | The thing experimented on: id, labels, current position and outline | n/a |
| **Tracking** | Keeping up with where each unit is and what shape it has | **Yes, and nothing else does** |
| **Routine** | Something done *to* a unit: monitor, optical treatment, FLIM, ... | Never |
| **Queue** | The actions that are due (1.6) | Never |

- A unit is the **experimental unit**: the smallest thing an experiment
  treats and measures independently. It may be one cell, a pair or a colony.
- **Units move.** Cells can be 10 µm away after 30 minutes, and the loop must
  keep following them.
  - Tracking finds the unit near where it was last seen. It searches a region
    sized by how far the unit could have moved since then.
  - Tracking may update the unit's position and its outline.
  - Every tracking result is recorded, so each unit's trajectory and shape
    over time is data.
- **Tracking is not a routine.** The orchestrator decides when to track. By
  default that's before a routine step, if the unit hasn't been tracked
  recently; an experiment may override that rule. Users see it under the
  name "tracking" (e.g. "unit 7 tracked: moved 3.1 µm").
- Units come from a survey, either drawn by the user or found by
  identification.

### 1.5 Routines and the orchestrator
- A **routine** acts on a unit as it currently is. It yields steps, e.g.
  `Scan(...)`, `Wait(600)`, `Record(key, value)`.
- **A unit may run several routines at once**, e.g. monitor every 30 min plus
  one optical treatment at 1 h. They never share state; the only thing they
  share is the unit, which only tracking changes.
- An **orchestrator** runs every (unit, routine) pair and performs routine
  steps and tracking. Series vs parallel follows from whether routines wait;
  there is no mode switch.
- **The split of responsibilities:**
  - routines see their unit and their own images, never the stage or other
    units;
  - the orchestrator owns the hardware, the timing, tracking, focus and
    persistence.
- **Treatment is not a concept.** A `Scan` may drive digital lines over the
  region it scans. What that means is the experiment's business.

### 1.6 The queue: human in the loop
- The **queue** is the user-facing list of actions that are due, across all
  units and routines. The **log** (2.2) is the separate record of what was
  done. Editing the queue is not logged; only what actually runs is.
- **The core interaction is prioritizing overdue work.** When the microscope
  is behind, ordering among overdue actions is arbitrary until the user moves
  an action (e.g. a FLIM image) to the top, so it runs next.
- **Other edits:** add a unit (its routines' first actions join the queue),
  add a routine to a unit, pause or remove queued actions. Changing an
  action's due time may also be wanted.
- **Experiment logic does not edit the plan mid-run** for now. Only the human
  does.
- Each (unit, routine) shows its **next action**, because a routine decides
  what comes next only after its current step. Whether fixed-schedule
  routines may declare their future actions up front is an implementation
  detail.

### 1.7 The experiment is the save folder
- **An experiment is a folder.** The save folder that is set *is* the
  experiment folder. No new noun in code.
- **Programs declare what they need from the folder.**
  - Most programs don't care, and save into whatever folder is set, mixed or
    not.
  - Programs that do care check the folder before a run and refuse to start
    when it fails. Autoloop, for example, requires a survey and units, and no
    earlier loop.
- **The Data Library groups datasets by experiment.** A saved dataset's
  experiment is its `meta_path.parent`. Unsaved datasets fall into one default
  "Unsaved" group.
- **TODO, panel source dropdown:**
  - Group the entries by experiment, the same way the Library does.
  - Each panel offering only the kinds it can display already happens:
    `SourcePicker` lists `library.matching(self.renders)`.

### 1.8 UI
- **Two panels for the autonomous workflow:**
  - **Map:** the full overview, with unit outlines and tracked positions.
  - **Experiment table:** the units and the queue, for setup and monitoring
    in one panel.
- **Analysis and Curves** are separate panels outside this pair.
- Neither panel may reach into a program, nor a program into a panel (see
  3.1).

### 1.9 Simulation (done)
- Everything simulated exists only under `uv run pyrpoc --simulated`. It lives
  in `plugins/programs/simulated/`, and registers from `register()`.
- New simulated programs and devices follow the same rule.

### 1.10 Old mosaic
The `mosaic` program and panel, `MosaicGroup` and `mosaic_simulation` were
placeholders and get deleted. The `Mosaic` data kind stays, because the
survey's raw tiles are one.

---

## 2. Data on disk

Analysis is built around **one timeline**. The analyst sees a table of
everything that happened ("at time X, this happened to unit 7"), and each row
links to the images it produced. The files are shaped to serve that view.

### 2.1 The four parts of a loop recording

| Part | What | Format |
|---|---|---|
| **Log** | Everything the orchestrator did, in order | `loop_events.csv`, append-only |
| **Units** | The units as frozen at loop start, with assigned labels | `loop_units.json` |
| **Frames** | Every image, grouped by series and unit | multi-page float32 TIFF |
| **Provenance** | Parameters, devices, inputs, seed, version | `loop_meta.json` (existing format) |

### 2.2 Log
- **One log for the whole run**, not one per unit.
  - The timeline is the main view; a per-unit view is a filter on it.
  - Some events have no unit (focus checks, re-anchoring, start and stop).
- **Core columns:**

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
  | `details` | JSON for fields specific to the kind (stage xyz, tracking score and shift, lines, dose added, reason for a state change) |

- **Why CSV:** appending rows is crash-safe. Convert to Parquet at the end only
  if logs get large.
- **The commit rule:** a row is written only after its image is on disk. Every
  row's pointer is valid, and a crash or cancel leaves whole steps.

### 2.3 Frames: series
- A **series** groups a unit's images by purpose. It's named by the routine
  (e.g. `bleach` for repeated ROI scans, `context` for occasional whole-field
  images).
- Log rows refer to images **by meaning**: `series=bleach, unit=7, index=23`.
  They never hold a file name, so the storage can change without touching
  the log or the analysis code.
- **Files:** `loop_frames/<series>/unit_NNNN_<ch>.tiff`, one page per image.

### 2.4 Frames: padding, so stacks open in ImageJ
- Users must be able to drag a stack into ImageJ and browse it, so
  **padding happens on write**. Every page in a file is the same size.
- **Padding is NaN.**
  - NaN means "not scanned", whether that's padding or outside the scanned
    outline. So each page records its own footprint, and no mask files are
    needed.
  - `nanmean` over a page is the mean over what was actually scanned.
- **Each page is centred on the unit's tracked position** and padded evenly
  on all sides.
  - In ImageJ the cell sits still while its surroundings move.
  - Each log row's `x_um, y_um` is its page centre, which padding never moves.
- **The box grows when needed.** It starts at the unit's minimum bounding
  box. When a new image doesn't fit, the writer rewrites that unit's file with
  every page re-padded to the larger box, then swaps it in with one atomic
  rename.
  - Growth is rare, so the rewrite costs little.
  - A crash during it leaves the old file intact.
  - Files are only rewritten while recording; once a recording ends, its data
    files are never modified.

### 2.5 Measurements
- **Raw images** are always saved. They are the ground truth.
- **Values the routine decides on** (an assignment, a computed value) are
  yielded as `Record(key, value)` and become `record` rows.
- **Derived results** (decay curves, per-group summaries) are saved as a
  convenience. Analysis must be able to recompute them from the log and
  frames alone.

### 2.6 The experiment folder

```
<experiment>/
  galvo_cal_meta.json                      um_per_volt
  survey_meta.json                         input: galvo_cal
  survey_tiles_tiles.json, survey_<ch>.tiff        raw tiles (Mosaic)
  survey_map.json, survey_map_<ch>.tiff            overview, focus surface, reference sites
  units_meta.json, units.json              input: survey; drawn or identified
  loop_meta.json                           input: units
  loop_units.json
  loop_events.csv
  loop_frames/<series>/unit_NNNN_<ch>.tiff
  loop_summary.npz                         derived
```

### 2.7 No custom format
- `data-format.md` promises that every recording loads in every later
  version. A custom container would mean maintaining its decoder forever.
- CSV, TIFF and JSON open in ImageJ, pandas and napari with no extra work.

---

## 3. Open

### 3.1 The main open problem: live state and queue edits
- **The problem.** The Experiment table panel must show the running loop's
  units and queue, and send the user's queue edits back to it, while:
  - the panel never imports or calls the program;
  - every Autoloop version works with the same panel, without a new `Data`
    kind for each message or each version.
- **Today's channels.** Programs publish `Data` into a `Dataset`; panels render
  it. Panels add datasets to the Library, which a later run reads at start
  (`MaskRef`). During a run, the only control is cancel. Nothing yet carries
  edits into a running program.
- **The direction so far.**
  - One shared noun for the autonomous workflow that every version uses.
  - Versions differ in what they put in it (routine, series and record names),
    not in its type.
  - The panel shows those contents generically, the way a table shows
    whatever columns exist.
- **To decide.**
  - What that noun is. A `Data` kind? Something beside `Dataset`?
  - How a growing log and a changing queue fit the "one current value" model.
  - How queue edits travel back into the running loop safely across threads.

### 3.2 Open on PR #53 (status)
1. **Policy during the loop.** Set aside: only the human edits the plan, via
   the queue (1.6).
2. **Live editing.** Yes, through the queue (1.6). How it's carried is 3.1.
3. **Footprint shape.** Tracking may change position and outline (1.4).
   Default tracking method still to choose.

### 3.3 Open, high level
4. **The API's shape.**
   - What an experiment must supply and what it inherits (choose and assign
     units, routines, tracking rule, focus check, lifecycle, summarise).
   - Composition or inheritance.
   - Where the module lives, and which of its nouns belong in `structs/`.
5. **The yield vocabulary.** `Scan`, `Wait`, `Record` so far. What else, and
   what a `Scan` returns to the routine.
6. **Motion vs drift.**
   - Stage drift, focus drift and cell motion all move a unit. Whether to
     estimate them separately, or let tracking absorb all of it, is not
     settled.
   - The role of reference sites follows from that.
7. **Dose accounting.** Per unit only, or a dose map in dish coordinates? Only
   the map catches a control unit dosed by a neighbour's whole-field scan.
   Tracking also costs dose.
8. **Folder requirements.** How a program declares what it needs from the
   experiment folder, and where `acquisition/` checks it before play.
9. **Analysis side.** A no-Qt reader (log, units, frames by series and unit)
   for notebooks, and what the Analysis panel shows.
10. **Linking the two panels.** Clicking a unit in the Map selects its row in
    the table and vice versa, without either panel knowing the other.
11. **Identification.** Whether finding units automatically is its own
    program or part of Autoloop, and what methods it starts with.
12. **Galvo Calibration.** Previously out of scope, now in the group. What it
    does.
13. **Later.** Resume after a crash, periodic re-survey, units larger than one
    field.

### 3.4 Carried over from the 2026-11-02 draft, to revisit
These were drafted before the design above and may no longer fit:
- **A `Scope` protocol** between the API and the hardware, with real and
  simulated versions, and a `SimulatedDish` device (cells, empty regions,
  tilted focus, drift, blur, noise, bleaching driven by dose).
- **Survey details:** focus anchors gated on content, a plane or quadratic
  focus fit, NCC stitching, reference sites chosen by content.
- **Scheduler details:** steps run whole and are never interrupted; lateness
  is recorded; ties are broken by stage distance.
- **Identification:** Otsu plus watershed, and grouping cells into multi-cell
  units by proximity.
- **First experiments:** `bleach_compare` and `longitudinal_monitor`.
