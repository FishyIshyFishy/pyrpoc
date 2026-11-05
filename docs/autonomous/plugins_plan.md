# Autonomous microscopy, part 2: programs and panels

This part builds the autonomous programs, data kinds, writers and panels on top
of part 1 (`engine_plan.md`): Experiments, earlier runs, live program inputs and
inputs kept out of JSON.

It is a plan for an agent to turn into code, phase by phase. Words come from
`pyrpoc/core/README.md`.

## Ground rules

- **An MVP.** It should work end to end and be pleasant to use. It doesn't have
  to be final. Where a technical choice is open, make a principled one: a
  method with a reason, not a tuned constant. Note the reason in a comment.
- **No backwards compatibility** and **no speculative options.**
- **Easy to take apart.** Each piece that decides something is its own module
  that can be replaced without rewriting the program around it:
  - feature detection;
  - autofocus;
  - the focus surface;
  - measuring motion;
  - stitching;
  - tracking;
  - the queue.
- **Real and simulated versions of every program.**
  - Shared modules hold the logic, and both versions call them.
  - A real version adds only the thin hardware step, written from the existing
    hardware code (`Raster`, `RoiRaster`, the stage moves in `autofocus` and
    `mosaic`).
  - A simulated version adds only its simulated step.
  - Real versions can't be run here. Their PR row says "not yet run on the rig".
- **Simulation never shapes real code.** It exists to make development easy
  away from the rig. No real program, device or building block changes form to
  fit it.
- **No screen text says "Experiment".** To the user it is a folder.
- **One commit per phase, one stacked PR per phase.** The checks in
  `.claude/CLAUDE.md` pass on every commit. Each phase is checked with scratch
  scripts outside the repo, with screenshots of anything visible.

**Names:**
- An *Experiment* is the save folder from part 1.
- The closed-loop programs are **loop programs** (for example
  `longitudinal_monitor`), so the two words never mean the same thing.

---

## The workflow

Everything saves into one folder.

```
Galvo Calibration ─▶ Survey ─▶ draw units in the Autonomy panel ─▶ loop program ─▶ watch and steer it
     (a run)         (a run)        (written to units.json)           (a run)      (live program inputs)
```

| Step | What you do | What it leaves behind | What the next step needs from it |
|---|---|---|---|
| Galvo Calibration | Run it | A recording with µm per volt | How far a volt moves the beam |
| Survey | Set the region's size, run it from a corner, watch the mosaic fill in | Raw tiles, the stitched overview, the focus surface, the measured motion rate | Where things are, what focus to use, how far cells move |
| Units | Draw outlines over the overview in the Autonomy panel | `units.json` in the folder | What to run routines on |
| Loop | Choose a loop program, set its inputs, run | The log, every frame, the units as they were at the start | n/a (analysis) |
| Steering | Edit things in the Autonomy panel while it runs | Live program input changes, and what the loop did about them | n/a |

The blockers come from part 1's `EarlierRun`:
- Survey says "needs Galvo Calibration saved in this folder".
- A loop says "needs Survey saved in this folder". It also refuses to start
  with no units.

## The two panels

- **Mosaic** (`plugins/panels/mosaic/`, rewritten). This is a general viewer
  for large tiled XY data, the way the Image panel is for one frame. It shows:
  - any `Mosaic` output, filling in live;
  - the refined overview once there is one.

  It knows nothing about surveys or units, so the Survey, a later tiled
  acquisition or anything else that publishes a `Mosaic` can use it.
- **Autonomy** (`plugins/panels/autonomy/`, new). This is the one place for
  units: drawing them, watching the loop and steering it.
  - Its backdrop is a survey's overview, drawn with the Mosaic panel's image
    view.
  - Over it go the units' outlines and, during a loop, their tracked positions.
  - Beside it: the units list, the queue, recent events and the steering
    controls.

  It doesn't share anything with the Mask Editor.

---

## Phase 1: The simulated dish

Everything after this is built against the simulation first.

### `utils/simulation/`

A package that imports only `core/` and numpy. It's a sample with its own state
over time, which simulated devices acquire from the way real devices acquire
from hardware.

- [ ] **`clock.py`: the dish's own time.**
  - It runs `speed` times faster than wall time, so a 30-minute survey takes
    about a minute.
  - Everything in the dish, and every simulated device, reads this clock.
  - Real programs never see it.
- [ ] **`cells.py`: seeded cells on an endless plane.**
  - Cells are laid out on a lattice, each lattice cell seeded from its own
    index, as `mosaic_simulation/slide.py` does now. A wide region therefore
    costs no more memory than the field being imaged.
  - Each cell is a soft-edged ellipse with its own brightness in each channel.
  - Large empty regions come from a smooth density field that is zero over
    part of the plane.
  - Motion is a persistent random walk with a set mean speed in µm per minute.
    Each cell's position is a function of dish time, so it's the same however
    often it's sampled.
- [ ] **`optics.py`: what a scan sees.**
  - **Focal surface:** a tilted plane that drifts slowly in dish time:
    `z(x, y, t) = z0 + a·x + b·y + c·t`.
  - **Defocus:** a Gaussian blur whose width grows linearly with the distance
    from focus.
  - **Stage drift:** a slow offset, added in dish time, between where the
    stage says it is and where it is.
  - **Galvo:** a true µm per volt. Calibration measures it, and nothing reads
    it directly.
  - **Signal:** expected photon counts scale with dwell time, then go through
    Poisson shot noise, so dwell time means what it does on the rig.
  - **Bleaching:** each cell keeps a light dose, the sum of dwell × intensity
    over the pixels that covered it. Its brightness falls as `exp(-dose / D)`.
- [ ] **`dish.py`: `Dish`**, which ties the above together. Its one entry point
  for imaging is a scan request:
  - the stage position;
  - the scan's volts and pixel counts;
  - the dwell time.

  It returns a `(C, H, W)` frame and adds the scan's dose. A `DishSettings`
  holds the seed, cell density, mean speed, clock speed, tilt, drift rates and
  true µm per volt.

### Simulated devices: `plugins/devices/simulated/`

These register only through `register()`, which `main.add_simulations()` calls
next to the simulated programs.

- [ ] **Simulated Dish:** a device that owns one `Dish` and its settings, and the stage
  the dish sits on. Its stage calls are named as the Prior stage's are
  (`xy_position`, `move_xy_to`, `move_z_to`, `wait_until_stopped`,
  `stop_smoothly`). Its panel shows the settings, moves the stage and shows
  the dish's time. The dish lives as long as the device does, so cells, dose
  and drift carry over from run to run, as a real sample's would.
- [ ] **Simulated Scanner,** `backed_by` the Simulated Dish: "scan these volts
  at this dwell and return the frame", taking as long as the scan would.
- [ ] Waits are in dish seconds, so the faster clock shortens real waits.

### Simulated programs

- [ ] **Simulated Confocal** replaces the `mosaic_simulation` program. It
  images the dish through the Simulated Scanner at the stage's position, with
  the confocal scan inputs.
- [ ] The `simulation` program ("2D image") stays. It exercises the display
  panels with any number of channels, and the dish doesn't replace that.
- [ ] Delete `mosaic_simulation/` and `specimen.py`.
- [ ] **Done when:** under `--simulated`, Simulated Confocal shows:
  - cells that move between frames;
  - blur as the stage leaves focus;
  - dimming as the same cells are scanned again.

## Phase 2: Galvo Calibration

Galvo Calibration measures µm per volt with the stage as the ruler.

Pixels are only how a scan is cut up. The physical size of a scan comes from
its volts and this calibration. After this phase, programs that need physical
sizes read µm per volt from the folder's calibration, never from a pixel-size
input.

- [ ] **The method,** in `plugins/programs/galvo_calibration/` with its logic
  in shared modules:
  1. **Find a feature.** Scan a frame. Pick the channel with the most signal
     over noise. Choose the brightest connected region that stands clear of
     its background, by Otsu's threshold and contrast. If nothing is clear
     enough, fail with "no clear feature in view".
  2. **Move and measure.** Move the stage by a known distance in x, rescan,
     and find the shift by phase correlation with sub-pixel peak fitting.
     Pixels shifted, converted with the scan's volts per pixel, gives volts.
     Do the same in y.
  3. **Repeat at several distances** in both directions on each axis, between
     a few pixels and half the field. The slope of the fit of volts against
     µm is the µm per volt for that axis. The fit's residuals report how good
     it was.
  4. **Return** the stage to where it started.
- [ ] **A calibration data kind:** µm per volt for each axis, the fit's
  residual, the channel and feature used, and the scan it used. It's small and
  goes in the recording's metadata.
- [ ] **Real version:** the existing `Raster` and the Prior stage.
- [ ] **Simulated version:** the Simulated Scanner and Simulated Dish.
- [ ] **Done when:** on the dish, the measured µm per volt is within 2% of the
  true value on both axes, and a field with nothing in it fails with a clear
  message.

## Phase 3: Survey and the Mosaic

The survey prepares the loop. It images the whole region, corrects focus as it
goes and measures how fast cells move. A survey may run for 10 to 30 minutes or
longer.

### Moving shared code

- [ ] `climb.py` and `tenengrad.py` move out of `autofocus/` into
  `programs/building_blocks/`, so every program can use them.
- [ ] `stitching.py` moves out of the mosaic panel into `program_io/outputs/`.
  The survey and the Mosaic panel both use it, and programs and panels may
  import `program_io/` but never each other.

### The `Mosaic` data kind and panel (general purpose)

- [ ] **Delete all of the old mosaic:**
  - the `mosaic` program and its inputs (`MosaicGrid` and the tile inputs);
  - the `Mosaic` kind and its writer;
  - the mosaic panel.
- [ ] **A new `Mosaic` kind:** tiles, each a frame with its position in dish
  µm and its µm per pixel, plus an optional refined overview with its origin
  and µm per pixel. A point on it is therefore a known dish position.
  - The writer appends each tile as it arrives, so a cancelled run keeps its
    tiles, and saves the overview when there is one.
  - If the recording layout changes, bump `FORMAT_VERSION`.
- [ ] **The Mosaic panel, rewritten for it:**
  - tiles placed live by their positions;
  - the refined overview once published;
  - channel and contrast controls as the Image panel has them.

### The Survey

- [ ] **Inputs:**
  - **The region's size** in µm (width × height), starting from the stage's
    current position as one corner.
  - **The field of view** in µm and pixels per field. Volts come from the
    calibration.
  - **The tile overlap** as a fraction.
  - **Dwell time.** It stays a dwell time, because "time per square
    micrometre" is hard to reason about.
- [ ] **Walking the region:** the survey tiles the region and walks it as a
  snake. It publishes each tile into the `Mosaic` as it's taken, then returns
  to the corner it started from.
- [ ] **Content detection** (its own module):
  - A tile has content when some channel's high-percentile signal stands above
    that channel's noise floor by a set number of noise widths.
  - The noise floor is estimated from the tile itself, with a robust median
    absolute deviation.
  - Only the channels that have signal are used for focus.
- [ ] **Autofocus during the survey:**
  - It runs only on tiles with content, so an empty tile's low sharpness never
    moves focus.
  - It reuses `climb.py` and `tenengrad.py`, searching around the focus
    surface's prediction instead of around the last z.
- [ ] **The focus surface** (its own module):
  - It's fitted from every tile focused so far, as a plane by robust least
    squares, so an outlier focus can't tilt it.
  - It predicts focus for the next tile, including tiles with no content.
  - It's saved with the survey, for the loop to start from.
- [ ] **Measuring motion** (its own module):
  - After the snake, the survey revisits the content tiles it took first.
    These are the ones with the most elapsed time.
  - For each, it measures how far the content moved by phase correlation.
  - The motion rate is the median speed in µm per minute across revisits. The
    spread is saved with it.
- [ ] **The overview:**
  - Live, tiles are placed from stage positions alone.
  - At the end, the overlaps between tiles are registered (the moved
    `stitching.py`), and the overview is blended across seams with edge
    weights.
  - Its origin and µm per pixel are saved.
- [ ] **The survey data kind:** the overview's reference (origin, µm per
  pixel), the focus surface, the motion rate and its spread, and what was
  surveyed. It's small, and goes in the metadata. The tiles and overview are
  the `Mosaic` output beside it.
- [ ] **Real version:** `Raster` and `RoiRaster`, and the Prior stage. Its
  display name is "Survey".
- [ ] **Simulated version:** the simulated devices.
- [ ] **Done when, on the dish:**
  - every tile is within one focus step of true focus;
  - no autofocus happens on an empty tile;
  - the overview is placed to within 1 px of the true positions;
  - the measured motion rate is within its reported spread of the simulated
    rate.

## Phase 4: Units and the Autonomy panel

A **unit** is the thing a loop experiments on: a region of the dish that may
hold one cell, a pair or more. Drawing units is not an acquisition.

- [ ] **`UNITS`,** a shared program input in `plugins/program_io/inputs/`,
  declared with `persisted=False`. A unit has:
  - an id;
  - an outline: a polygon in dish µm, so it survives a change of resolution
    between the survey and the loop;
  - labels, such as its group (treated or control). Loop programs read the
    group from the label.
- [ ] **`units.json` in the save folder** is where units are kept.
  - The Autonomy panel writes it on every edit, and loads it whenever the save
    folder changes, including at launch.
  - The next day, the same folder brings the units back. A new folder starts
    with none.
  - The panel is the only reader and writer.
- [ ] **A small engine hook,** if needed, so a panel can read the current save
  folder and hear when it changes. It stays within the import rules.
- [ ] **The Autonomy panel, first half:**
  - the latest survey's overview in this folder as its backdrop;
  - draw, move, reshape and delete outlines;
  - a units list with each unit's labels, which you edit there.
- [ ] **Done when:** after a simulated survey, you draw units, restart the app
  with the same folder and they're back, and the loop's blocker clears.

## Phase 5: The loop

The loop runs routines on units for a long time, tracking each unit as it moves
and keeping focus as it drifts.

**Four ideas, kept apart:**

| Idea | What it is | Changes the unit? |
|---|---|---|
| **Unit** | The thing experimented on | n/a |
| **Tracking** | Finding the unit again near where it was last seen | **Yes, and nothing else does.** In this MVP it moves the outline and keeps its shape as drawn. |
| **Routine** | Something done *to* a unit (monitor, treat, ...). A generator that yields steps (`Scan`, `Wait`, `Record`) and sees only its own unit and images. | Never |
| **Queue** | The steps that are due, ordered by due time | Never |

### The loop API

- [ ] **A module in `programs/building_blocks/`** that loop programs build on.
  - A loop program supplies its routines.
  - It inherits tracking, focus correction and the queue, and may override
    any of them.
  - A new experiment is a new loop program, not a mode of an existing one.
  - "Treatment" isn't a concept: a `Scan` may drive digital lines over the
    region it scans.
  - The loop reads time from its devices' clock, so the same code runs in dish
    time under simulation.

### The orchestrator

- [ ] It runs every (unit, routine) pair, taking the earliest due step next.
  There's no series or parallel switch: routines that don't wait run back to
  back.
- [ ] **Tracking before each step** (its own module):
  - It searches a window around the last position. The window's radius is the
    motion rate × the time since last seen, plus a margin of the rate's
    spread.
  - It matches by normalised cross-correlation against the unit's last crop.
  - The first visit uses the survey crop and the gap since the survey.
  - A weak match is logged as lost, not guessed.
- [ ] **Focus:** each scan starts from the survey's focus surface, plus an
  offset that periodic focus checks during the loop update. The surface is
  never refitted mid-loop.
- [ ] Each step runs whole. It records when it was due and when it ran, so
  lateness is visible.
- [ ] Between steps, the loop reads its live inputs (part 1, phase 4) and
  applies them.

### `longitudinal_monitor`, the first loop program

- [ ] It scans each unit every period for a duration, with an optional
  treatment scan at the start that drives the digital lines.
- [ ] It produces intensity against time, per group.
- [ ] **Real version:** `Raster` with the stage.
- [ ] **Simulated version:** the simulated devices.

### The loop recording, in the folder

- [ ] **One log for the whole run,** a JSON-lines file: every scan, tracking
  result, focus check, steering change applied and error, in order,
  append-only.
- [ ] **Every frame,** grouped by unit and routine, cropped around the unit's
  tracked position, so a stack opens in ImageJ with the cell still.
- [ ] **The units as they were at the start,** plus what it read: the survey
  and the parameters.
- [ ] A log row is written only after its image is on disk, so a cancelled or
  crashed loop leaves whole steps.
- [ ] **Done when, on the dish:**
  - moving cells are tracked to within about 1 µm;
  - focus holds as the surface drifts;
  - routines that wait interleave on time when the microscope keeps up, and
    report lateness when it can't;
  - the log and frames alone recompute the curves.

## Phase 6: Watching and steering

- [ ] **A loop state data kind.** It's small and republished after every
  step:
  - each unit's state, last tracked position and next step;
  - the queue;
  - recent events.
- [ ] **The Autonomy panel, second half,** which works for any loop program:
  - tracked positions over the overview;
  - each unit's state and next step;
  - the queue, with overdue steps first;
  - recent events.
- [ ] **Steering, all through live program inputs** that the loop reads
  between steps:
  - run a step now;
  - add a unit (drawn in the panel) or remove one;
  - pause or resume a unit;
  - change a routine's settings for the rest of the run.

  The loop compares what it reads with what it read last, applies the change
  and logs what it actually did.
- [ ] **The Curves panel:** intensity against time per group, updated live
  from the loop's output.
- [ ] **Done when, during a simulated loop:**
  - "run now" on an overdue step makes it run next;
  - an added unit starts its routines;
  - a paused unit stops being scheduled until it's resumed;
  - a changed period takes effect on that unit's next step.

---

## Later

- Moving `MASKS` out of the session as well. By the same test as units,
  closing the app and coming back the next day shouldn't bring them back.
- A better home for units than a file the panel owns.
- Identifying units automatically: an "Identify" action in the Autonomy panel
  that fills `UNITS` from the overview, for you to edit.
- Smarter autofocus: per-channel weighting and better search strategies.
- Tracking that updates a unit's shape, not just its position.
- A loop program's own logic changing the plan mid-run, such as stopping on a
  plateau or reassigning groups. Nothing should rule it out.
- More loop programs, such as `bleach_compare`.
- Selection linked between the Autonomy panel and the Mosaic panel.
- Re-anchoring on reference sites, re-surveying during a long loop, units
  larger than one field, and resuming after a crash.
