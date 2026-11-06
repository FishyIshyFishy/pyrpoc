# Autonomous microscopy, part 2: programs and panels

**Status:** built, in the terms of the simplified core rather than the ones below. The
pipeline itself is #68's `programs/building_blocks/multisite/`; its README describes units,
commands and the loop's scratch. The Autonomy panel shows a `SurveyOutput`'s overview, edits the
`UNITS` input before a loop, and sends `UNIT_COMMAND`s while one runs. The Curves panel plots a
`LoopLogOutput`'s series, chosen by routine and name.

This is the second of two implementation PRs. It builds the autonomous programs,
data kinds, writers and panels in `plugins/`, on top of part 1
(`engine_plan.md`): Experiments, upstream runs, live program inputs and large
input values.

It is a plan to review, not a spec. An agent turns each phase into code. Words
come from `pyrpoc/core/README.md`.

**Ground rules:**
- **No backwards compatibility** and **no speculative options.**
- **Easy to take apart.** Each piece that does something (focus correction, the
  focus surface, measuring motion, refining the stitch, tracking) is a separate,
  named part that can be removed or replaced without rewriting the program
  around it.
- **Simulation** exists only under `uv run pyrpoc --simulated`. Every program
  here has a simulated version that is developed and tested against before the
  rig.
- **One commit per phase.**

**Names:** an *Experiment* is the save folder from part 1. The closed-loop programs
are **loop programs** (for example `longitudinal_monitor`), so the two words
never mean the same thing.

---

## The workflow

All of it saves into one Experiment, the save folder.

```
Galvo calibration ──▶ Survey ──▶ you draw units on the Map ──▶ Loop program ──▶ you watch and steer it
   (a run)            (a run)        (a program input)            (a run)          (live program inputs)
```

| Step | What you do | What it leaves behind | What the next step needs from it |
|---|---|---|---|
| Galvo calibration | Run it | A recording with µm per volt | Pixel size in µm |
| Survey | Set the extent, run it, watch the overview fill in | Raw tiles, the overview, the focus surface, the measured motion rate | Where things are, what focus to use, how far cells move |
| Units | Draw outlines on the overview in the Map | The `UNITS` program input | What to run routines on |
| Loop | Choose the loop program, set its inputs, run | The log, every frame, the units as frozen at start | n/a (analysis) |
| Steering | Edit things in the loop's panel while it runs | Live program input changes, logged | n/a |

The loop refuses to start without a survey in the Experiment and at least one
unit, just as Survey refuses without a calibration (part 1, phase 3).

---

## Phase 1: The simulated dish

Everything after this phase is built and checked against the simulation first.

- [ ] **A simulated dish under `--simulated`,** with:
  - seeded cells that move at a set rate (µm per minute);
  - empty regions;
  - a tilted focal surface that drifts slowly over time;
  - xy stage drift;
  - defocus blur and shot noise;
  - bleaching that grows with how much each spot has been scanned;
  - a clock faster than wall time, so a 30-minute survey and a long loop can
    be tested in minutes.
- [ ] Simulated stand-ins for the galvo, DAQ and stage that image this dish,
  so real programs run against it unchanged where possible.
- [ ] Delete `mosaic_simulation` and the current placeholder specimen it uses.
- [ ] **Done when:** a confocal frame under `--simulated` shows cells that
  move, blur away from focus and bleach.

## Phase 2: Galvo calibration

- [ ] **A Galvo Calibration program** that measures µm per volt and saves it as
  a recording in the Experiment.
- [ ] Decide when building: what it scans and what it fits. This is a
  technical choice, not a product one.
- [ ] A simulated version against the simulated dish, with a known true value.
- [ ] **Done when:** on the simulated dish, the measured µm per volt is within
  2% of the truth.

## Phase 3: Survey

The survey sets up the loop. It images the whole region you care about,
corrects focus as it goes, and measures how much cells move. A survey may run
for 10 to 30 minutes or longer. A 5×5 grid is only for first testing.

- [ ] **The Survey program** replaces `mosaic`:
  - reads the galvo calibration from the Experiment (part 1, phase 3);
  - tiles the region from its extent, field of view and overlap, and walks it
    as a snake;
  - publishes each raw tile as it's taken;
  - returns to its start position at the end.
- [ ] **Focus correction during the survey,** using a **focus surface**:
  - Autofocus only on tiles that have content, so an empty tile's low
    sharpness never moves focus.
  - Tiles aren't treated as independent: the surface is fitted from the tiles
    focused so far, and predicts focus for the next tile.
  - The fitted surface is saved with the survey, for the loop to start from.
- [ ] **Measuring how fast cells move.** The survey revisits some tiles later
  in its run and measures how far their content moved. It saves a motion rate
  (µm per minute) for the loop to size its search with.
- [ ] **The overview:**
  - stitched live from stage positions as tiles arrive;
  - refined at the end (registration between overlapping tiles, then blended
    seams);
  - saved with its origin and µm per pixel, so a point on it is a known dish
    position.
- [ ] **A survey data kind:** the overview, its origin and µm per pixel, the
  focus surface and the motion rate. The raw tiles remain their own output.
- [ ] **Shared code moves** so programs can use it: `climb.py` and
  `tenengrad.py` out of `autofocus/`, `stitching.py` out of the mosaic panel.
- [ ] **Deleted:** the `mosaic` program and its inputs. The mosaic panel is
  replaced by the Map in phase 4.
- [ ] **Done when, on the simulated dish:**
  - every tile is within one focus step of true focus;
  - no autofocus happens on an empty tile;
  - the overview is placed to within 1 px;
  - the measured motion rate is close to the simulated one.

## Phase 4: The Map and units

A **unit** is the thing a loop experiments on: a region in the dish, which may
hold one cell, a pair, or more. Units are a program input, like masks: you set
them in a panel, and the loop reads them. Drawing units is not an acquisition.

- [ ] **`UNITS`, a shared program input,** declared in
  `plugins/program_io/inputs/`. Each unit has:
  - an id;
  - an outline in dish µm, so it survives a change in resolution between the
    survey and the loop;
  - labels, such as its group (treated or control).

  It's a large value (part 1, phase 5).
- [ ] **The Map panel** replaces the mosaic panel. It shows:
  - the survey overview, filling in live during a survey;
  - the units' outlines.
- [ ] **Drawing units on the Map:**
  - draw, edit and delete outlines, and set their labels;
  - shares the drawing canvas with the Mask Editor.
- [ ] **Done when:** after a simulated survey, you draw units on the Map, they
  persist in the session, and the loop's blocker clears.

## Phase 5: The loop

The loop runs routines on units over a long time, tracking each unit as it
moves and keeping focus as it drifts.

**Four ideas, kept apart:**

| Idea | What it is | Changes the unit? |
|---|---|---|
| **Unit** | The thing experimented on | n/a |
| **Tracking** | Finding the unit again near where it was last seen | **Yes, and nothing else does.** In this MVP it moves the outline; the outline's shape stays as drawn. |
| **Routine** | Something done *to* a unit (monitor, treat, ...). It yields steps such as `Scan`, `Wait` and `Record`, and sees only its own unit and its own images. | Never |
| **Queue** | The actions that are due | Never |

- [ ] **The loop API,** a module in `plugins/programs/` that loop programs
  build on.
  - A loop program supplies its routines and how it assigns groups. It
    inherits tracking, focus correction and the queue by default, and may
    override them.
  - A new experiment is a new loop program, not a mode of an existing one.
  - "Treatment" is not a concept: a `Scan` may drive digital lines over the
    region it scans.
- [ ] **The orchestrator:**
  - runs every (unit, routine) pair, interleaving them by due time. There's no
    series/parallel switch: routines that don't wait simply run back to back;
  - before a routine step, tracks the unit, searching a region sized by the
    motion rate and the time since it was last seen. The first visit searches
    across the gap since the survey, using the survey crop as its template;
  - corrects focus from the survey's focus surface, adjusted by periodic
    checks during the loop;
  - runs each step whole, and records when it was due and when it ran;
  - between steps, reads its live inputs (part 1, phase 4) for your edits.
- [ ] **`longitudinal_monitor`, the first loop program:**
  - scans each unit every period for a duration, with an optional treatment at
    the start;
  - produces intensity against time, per group.
- [ ] **The loop recording,** saved in the Experiment:
  - **one log for the whole run:** every scan, tracking result, focus check,
    edit and error, in order, append-only;
  - **every frame,** grouped by unit and routine, and centred on the unit's
    tracked position so a stack opens in ImageJ with the cell still;
  - **the units as frozen at loop start;**
  - **what it read:** the survey and the parameters.

  A log row is written only after its image is on disk, so a cancelled or
  crashed loop leaves whole steps.
- [ ] Parameters are logged as with any run. There's no separate dose
  accounting.
- [ ] **Done when, on the simulated dish:**
  - moving cells are tracked to within about 1 µm;
  - focus holds as the surface drifts;
  - routines that wait interleave on time when the microscope can keep up, and
    report lateness when it can't;
  - the log and frames are enough to recompute the curves.

## Phase 6: Watching and steering the loop

- [ ] **The Loop panel,** which works for any loop program. It shows:
  - each unit with its state, last tracked position and next action;
  - the queue, overdue actions first;
  - recent events.

  It renders the loop's live state, a small data kind republished after every
  step.
- [ ] **The Map during a loop:** tracked positions over the overview.
- [ ] **Steering from the Loop panel,** all through live program inputs:
  - run an action now;
  - add a unit (drawn on the Map) or remove one;
  - pause or resume a unit;
  - change a routine's settings for the rest of the run.

  Edits aren't logged as edits: the loop logs what it actually did after
  reading them.
- [ ] **The Curves panel:** intensity against time per group, updated live.
- [ ] **Done when, during a simulated loop:**
  - "run now" on an overdue action makes it run next;
  - an added unit starts its routines;
  - a paused unit stops being scheduled until it's resumed;
  - a changed period takes effect on that unit's next action.

---

## Later
- Identifying units automatically: an "Identify" action in the Map panel that
  fills `UNITS` from the overview, for you to edit.
- Tracking that updates a unit's shape, not just its position.
- A loop program's own logic changing the plan mid-run (stop on a plateau,
  reassign groups). Not in the MVP, but nothing should rule it out.
- More loop programs, such as `bleach_compare`.
- Selection linked between the Map and the Loop panel.
- Re-anchoring on reference sites, re-surveying during a long loop, units
  larger than one field, resuming after a crash.
