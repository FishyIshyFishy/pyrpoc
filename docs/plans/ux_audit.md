# UX audit and redesign: multisite imaging (PR #67)

This document has three parts:

- **Part A, the plan:** what I will change and why. It comes first so it can be read alone.
- **Part B, the audit:** what I found by running the app and using it.
- **Part C, reconciling #67 with #68:** the groundwork, done before any UI change.

The results after the changes are in `ux_results.md`, beside this file.

A few words used below:
- **panel**: one of the boxes inside the app window, such as Acquisition or Data Library.
- **loose window**: a panel that floats as its own window instead of sitting inside the app window.
- **site**: one region of the dish that the loop keeps coming back to. The code calls it a *unit*.
- **loop**: the Longitudinal Monitor program, which visits each site again and again for hours.

---

# Part A: Redesign plan

## The ten biggest problems, and what I will do about each

Ranked by how much each one hurts a first-time user, times how often they run into it.

| # | Problem | What I will do |
|---|---|---|
| 1 | **The app opens on the wrong page.** The first screen is the Data Library tab saying "Nothing open". Acquisition, where every job starts, is hidden behind a tab. | Open on Acquisition. |
| 2 | **New panels open as loose windows stacked on top of the app.** The Simulated Dish settings, the Mosaic, the Autonomy and the Curves panels each open as a separate window over the main one, covering the menu and each other. | Put new panels inside the app window: display panels to the right of Acquisition, device panels beside the Devices tab. They can still be dragged out. |
| 3 | **The multisite screens speak the program's internal language.** "Autonomy", "unit", "Run now", "waiting/lost/done", "Queue, overdue first", "monitor: intensity", "unit 3 monitor due". | Rename everything to plain words (vocabulary table below): **Sites** panel, **site**, **Visit next**, **Activity**, and so on. |
| 4 | **The Sites panel gives no guidance.** With no survey it shows blank axes and "0 units"; with a survey it never says how to draw; before a loop it never says what to do next; after a stopped loop it says "Loop finished" and shows stale "next visit" times. | One status line that always says where you are and what to do next ("Click Draw site, then click the corners…", "3 sites drawn: start Longitudinal Monitor to follow them", "Loop stopped early at 12 min: start Longitudinal Monitor again in this folder to carry on"). |
| 5 | **Clicking Pause, Visit next, or drawing during a loop gives no sign that anything happened**, for 0.3–0.7 s (longer while a scan is in progress). If the loop refuses a change, the reason is buried in the event list, and the status line is overwritten with "4 units". | Say "Asked the loop to pause site 2…" at once, then "Site 2 paused". Show a refusal and its reason in the status line. |
| 6 | **The loop's controls are greyed out with no reason**, and the right side of the panel is a stack of three lists, two of which repeat each other ("Queue" repeats the table's "Next" column). The event list is mostly noise ("due", "scanned, found", "measured", several times a minute). | Remove the Queue list. Label the buttons for the chosen site ("Pause site 2"). When none is chosen, say "Click a site to pause it or visit it next". Keep only meaningful events in Activity: added, removed, paused, resumed, lost, found again, finished, failed, refused. |
| 7 | **Deleting a site during a loop is permanent and has no confirmation.** Delete sits right beside Draw, and the Delete key works too. A deleted site can't come back: drawing it again makes a new site with a new number. | Ask before removing a site from a running loop. Before a loop, deleting stays instant, since it costs nothing. |
| 8 | **Opening the Sites panel takes 1.4 s, and the loop view slows down as a loop runs.** 0.5 s goes on colouring the survey picture. Each redraw rebuilds the loop's whole history: 3 ms at 1,000 events, 250 ms at 100,000. A day-long loop on 100 sites would freeze the panel. | Colour the picture faster. Have the loop's log keep its summary up to date as each event arrives, so a redraw no longer replays every event. |
| 9 | **Messages blame the wrong thing or show code names.** After a successful run the status says "stopped". A missing calibration reads "needs a calibration (GalvoCalibrationOutput) saved in this folder". A folder that isn't clean reads "needs a folder holding only what it reads (move out acquisition)". | "finished" after a successful run. "needs a calibration saved in this folder". "needs a folder with nothing else in it: choose a new folder, or move out acquisition". |
| 10 | **The main action is a tiny unlabelled ▶ icon** (36×25 px), next to an equally small ■. | Show the words beside the icons: "▶ Start", "■ Stop". |

Smaller fixes in the same changes:
- Curves: per-site lines are nearly invisible. I'll make them readable.
- Curves: show the measurement's own name, not "routine: name".
- Curves: show a hint while there is nothing to plot.
- Sites panel: the Group column says how to edit it (double-click).

## Vocabulary

These are the words the screen will use. The code keeps its own names; only what is shown changes.

| Now | Where | What it means | New | Leak from inside the program? |
|---|---|---|---|---|
| Autonomy | panel name, Add Display menu | the map where you draw sites and watch the loop | **Sites** | yes: an internal project name |
| Autonomous | Add Display heading | the multisite panels | **Multisite** | yes |
| Curves | panel name | each group's measurements over time | **Site Curves** | |
| unit, Draw unit, "Unit" column, "3 units" | Autonomy panel | a region of the dish the loop follows | **site**, **Draw site**, "Site", "3 sites" | yes: #68's internal word |
| Delete | Autonomy panel | stop following a site | **Remove** | |
| Run now | Autonomy panel | visit this site next, ahead of the others | **Visit next** | |
| Pause / Resume | Autonomy panel | skip / stop skipping this site | **Pause site N / Resume site N** | |
| waiting | State column | will be visited when due | **Waiting** | yes: a raw program value |
| lost | State column | the last scan could not find it | **Lost track** | yes |
| done | State column | its routine has finished | **Finished** | yes |
| paused | State column | skipped until resumed | **Paused** | yes |
| State | column heading | | **Status** | |
| Next | column heading | when it is next due | **Next visit** | |
| Queue, overdue first | list heading | the order of upcoming visits | (removed) | yes |
| Recent | list heading | the latest things the loop did | **Activity** | |
| "unit 3 monitor due", "unit 3 intensity measured", "unit 3 scanned, found" | Recent list | low-level loop events | (hidden), "site 3 lost track", "site 3 found again", … | yes |
| "Loop running, 4.2 min" | heading | | **"Loop running for 4 min"** | |
| "Loop finished" (after Stop) | heading | the loop was stopped before its routines ended | **"Loop stopped early…"** (only "finished" when it really finished) | |
| "No running loop takes unit commands" | status line | the change was not sent | **"No loop is running, so nothing was changed"** | yes |
| monitor: intensity | Curves "Measurement" | routine name: measurement name | **intensity** (routine shown only if two routines measure the same thing) | yes |
| ungrouped | Curves legend | sites with no group | **no group** | |
| stopped | Acquisition status, after a run that ended normally | the run ended | **finished** | |
| a calibration (GalvoCalibrationOutput) | Acquisition "needs …" | | **a calibration** | yes: a class name |
| a folder holding only what it reads (move out acquisition) | Acquisition "needs …" | the multisite steps each want a clean folder | **a folder with nothing else in it: choose a new folder, or move out acquisition** | |

## Each job, simplified

The click counts are honest estimates: most of the gain is in clarity, not clicks.

**Every job: the window.** Opens on Acquisition, and added panels sit inside the window.
```
┌ Panels  Style ──────────────────────────────────────────────────────────────┐
│ [Acquisition][Devices][Data Library] │ [Sites][Site Curves][Mosaic]         │
│  Program: Simulations › Survey       │                                      │
│  [▶ Start] [■ Stop]                  │   (the panel chosen on the right)    │
│  ☑ Save  exp / acquisition           │                                      │
│  ● ready                             │                                      │
│  ▸ Earlier recordings  …             │                                      │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

**Draw sites, then start the loop.** In the Sites panel:
```
 Source: [Latest ▾]
 [Draw site]  [Remove]
┌───────────────────────────────┬────────────────────────────────────┐
│   survey map, sites outlined  │ Site │ Group  │ Status │ Next visit │
│   and numbered                │  1   │ treated│        │            │
│                               │  2   │        │        │            │
│                               ├────────────────────────────────────┤
│                               │ (loop controls appear once a loop  │
│                               │  runs)                             │
└───────────────────────────────┴────────────────────────────────────┘
 3 sites drawn. Start Longitudinal Monitor in Acquisition to follow them.
```

**Watch and steer the running loop.** Same panel, loop section filled in:
```
├────────────────────────────────────┤
│ Loop running for 12 min            │
│ [Pause site 2]  [Visit site 2 next]│
│ Activity                           │
│  12.4 min  site 4 lost track       │
│  12.1 min  site 4 added            │
│   6.0 min  site 1 paused           │
└────────────────────────────────────┘
 Asked the loop to pause site 2…   →   Site 2 paused.
```

**Read the results.** Site Curves: one bold line per group, each site faint but visible behind it,
and "Measurements appear here once a loop has measured its sites" while empty.

## What I am deliberately not changing

- **#68's design**: how sites, commands, the loop's save files and folder checks work. The plan
  needs no architectural change. Two small additions inside #68's code (listed in Part C) keep
  to its interfaces.
- **The "clean folder" rule** (each multisite step wants a folder holding only the steps before
  it). It traps a user after a failed calibration, but changing it is a decision about #68's
  design, so I only make the message say what to do. See "Found but not fixed".
- **The flaky simulated calibration** (fails now and then at the default speed). It's not a
  screen problem; #68 already lists it.
- **The Acquisition settings cards, the Data Library, the Mask Editor, the theme.** They have
  problems (Part B) but they are shared by every program, not part of the multisite work. Changing
  them would spread this PR thin.
- **The default recording name "acquisition"**, which makes every recording in the multisite
  folder read "acquisition", "acquisition_2", "acquisition_3". Changing it changes how every run
  is named, so it is listed as a recommendation, not done.
- **Click counts for drawing.** A square site takes 5 clicks (4 corners and a close). A
  drag-to-draw-a-rectangle mode would make it 1, but it is a new feature; I list it as a
  recommendation.

---

# Part B: The audit

## How I tested

The app is a desktop app (PyQt), not a web page, so a browser tool can't drive it. I drove the
real app window off-screen with Qt's own test tools: real mouse clicks on real buttons, and real
clicks on the map to draw sites. I took a screenshot at every step and timed each response. Some
things the test couldn't do for real:

- **Windows-only parts.** The app targets Windows. On the Linux test machine, three Windows-only
  calls in the camera drivers were stood in for by the test script, outside the app's code.
- **File dialogs.** Where a real user would pick a folder in a dialog, the script set the folder
  directly and counted the clicks the dialog would take (2).
- **Loose windows.** The off-screen system has no window manager, so loose windows have no title
  bar and pile up in one corner. For the whole-screen pictures, the script cascades them the way
  Windows places new windows, and draws a grey title strip over each. Single-panel pictures are
  exact.

All runs used the simulated dish (`pyrpoc --simulated`) at its default settings.

## The user jobs (my assumptions)

Taken from the code, the README files and the plan documents. Correct me if these are wrong.

1. **Set up the instrument:** add the dish and scanner (simulated here; real hardware otherwise).
2. **Calibrate** the scanner against the stage (Galvo Calibration).
3. **Survey** the dish: map it tile by tile into one overview picture (Survey).
4. **Choose sites:** draw the regions of the dish to follow, and put them in groups (treated,
   control).
5. **Start the loop** (Longitudinal Monitor) on those sites.
6. **Watch and steer the loop:** pause a site, visit one next, add or remove sites while it runs.
7. **Read the results:** each group's measurements over time (Curves), then reopen the experiment
   later.

I assume a first-time user is a microscopist: "galvo", "dwell time" and "focus" are their
words, but they don't know the program's internal parts.

## Clicks per job (before)

Counted from first launch, doing each job the shortest way the screens allow.

| Job | Clicks | Where they go | Could be fewer |
|---|---|---|---|
| 1 Set up devices | 7 | Panels › Add Device › Simulated Dish (3), the same for the Scanner (3), the Devices tab (1) | The Devices tab click is only needed to check; the dish panel's loose window has to be moved out of the way (not counted). |
| 2 Calibrate | 8 | Acquisition tab (1), program list (2), Save box (1), folder button (1), the dialog (2), Start (1) | The Acquisition tab click should not be needed (problem 1). |
| 3 Survey | 3 | program list (2), Start (1) | — |
| 4 Choose 3 sites | 22 | Panels › Add Display › Autonomous › Autonomy (4), Draw unit (1), 3 sites × 5 clicks, Draw off (1), choose a site (1) | Drawing by dragging a rectangle would make each site 1 action (recommendation). |
| 5 Start the loop | 3 | program list (2), Start (1) | — |
| 6 Steer: pause one, visit one next, add one | 11 | choose (1), Pause (1), choose (1), Run now (1), Draw (1), 5 clicks, Draw off (1) | — |
| 7 Curves | 4 | Panels › Add Display › Autonomous › Curves | — |
| **Total** | **58** | | |

Hidden costs that don't show in clicks:
- After each panel opens, its loose window must be dragged off the main window to see both.
- If the calibration fails (it did in 1 of 4 runs), the folder is left blocked. The user has to
  work out from "move out acquisition" that they need a new folder: about 4 more clicks and some
  head-scratching.

## Speed (before)

| What | Time | Notes |
|---|---|---|
| App start to window | 0.64–0.87 s | fine |
| Switch program in Acquisition | 31–77 ms | fine |
| Click Start → status changes | 25–31 ms | fine |
| Calibration | 5.1–5.8 s | simulated |
| Survey | 19.5–21.9 s | simulated |
| **Open the Autonomy panel** | **0.9–1.4 s** | 0.5 s of it is colouring the survey picture, on the screen's thread, so the whole app freezes meanwhile |
| Open Curves / Mosaic panel | 0.28 s / 0.40 s | fine |
| Close a drawn outline | 30–46 ms | fine |
| Loop start → shows in the panel | 0.6 s | the panel redraws at most 4 times a second |
| Pause → shows as paused | 0.29–0.66 s | waits for the loop to take the command; **no feedback meanwhile** |
| Site drawn mid-loop → appears | 0.30–0.43 s | same |
| Each panel redraw during a loop | 23 ms (max 36 ms) | at 60 events. It grows with the log: 3 ms at 1,000 events, 24 ms at 10,000, **250 ms at 100,000**. |
| Reopen a saved experiment | 1.1–1.2 s | fine for a one-off |

How the redraw cost was measured: the loop logs about 700 events per hour of dish time with 4 sites.
At 100 sites over a day that is about 400,000 events, so each redraw would take about 1 s, four
times a second.

No flicker was seen: the map only rebuilds its outlines when they change.

## Every problem, ranked

**Impact:** how badly it stops a first-time user (3 = blocks or misleads, 2 = slows or confuses,
1 = annoys). **Frequency:** how often they meet it (3 = every session, 2 = every multisite
experiment, 1 = sometimes). **Score** = impact × frequency.

| Score | Problem | Kind |
|---|---|---|
| 9 | Opens on Data Library ("Nothing open") instead of Acquisition | hierarchy |
| 9 | Added panels open as loose windows on top of the app and each other | inconvenient |
| 6 | Internal words throughout the Autonomy panel (unit, Run now, waiting/lost/done, Queue) | nomenclature, leak |
| 6 | Autonomy panel: no guidance in empty, drawing, before-loop and stopped states | feedback |
| 6 | Opening the Autonomy panel freezes the app for 0.5–1.4 s | lag |
| 6 | Loop controls greyed out with no reason; three lists, one repeating the table | useless, hierarchy |
| 6 | No feedback for 0.3–0.7 s after Pause / Run now / drawing in a loop; refusals hidden | feedback |
| 6 | Reopened stopped loop says "Loop finished" and shows stale next-visit times | feedback |
| 6 | After a successful run, the status says "stopped" | nomenclature |
| 6 | Main action is a tiny unlabelled ▶ icon | hierarchy |
| 6 | A failed calibration blocks the folder; "move out acquisition" doesn't say what to do | feedback |
| 4 | Removing a site mid-loop is permanent, with no confirmation, beside Draw | destructive |
| 4 | Class name in "needs a calibration (GalvoCalibrationOutput) …" | nomenclature, leak |
| 4 | Event list noise: "due", "scanned, found", "measured" several times a minute | useless |
| 4 | The loop's redraw slows as the log grows (250 ms at 100,000 events) | lag (future) |
| 4 | Curves: per-site lines nearly invisible; "monitor: intensity", "ungrouped" | nomenclature, contrast |
| 4 | Every recording is named "acquisition", "acquisition_2", … | nomenclature |
| 3 | No visible keyboard focus | accessibility |
| 2 | Group can only be set by double-clicking an empty cell, with no hint | inconvenient |
| 2 | The group average jumps when a site joins mid-loop | misleading |
| 2 | Map axes have numbers but no units (µm) | hierarchy |
| 2 | Placeholder and greyed-out text contrast is 2.3–2.9:1 (below the 4.5:1 rule of thumb) | accessibility |
| 2 | Settings cards summarise as one dense line of names and numbers | hierarchy |
| 1 | "Source: Latest" on every data panel doesn't say what it follows | nomenclature |

## The findings in detail

### Useless things on screen
- **"Queue, overdue first" list** (Autonomy, during a loop). It repeats the table's "Next" column
  in another order. Cut it.
- **Most of the "Recent" list.** "unit 3 monitor due", "unit 3 intensity measured", "unit 3
  scanned, found" appear for every site on every visit. They bury the events a person needs: a
  site lost, a command refused, a routine failed. Keep only those.
- **Stale "Next" times after a loop has stopped** (`before_reopened_stopped_loop.png`). They
  count down from nothing. Hide them once the loop is not running.
- **The 0–1 axes on empty plots** (`before_empty_panels.png`). They look like data. A sentence
  is more useful.

### Inconvenient controls
- **Loose windows** (`before_loose_windows.png`, `before_loop_running_whole_screen.png`). Every
  panel added from the menu, and every device's own panel, opens as a separate window on top of the
  main window. The second covers the first. The user has to drag each one away or dock it.
- **Four menu levels to the Autonomy panel** (Panels › Add Display › Autonomous › Autonomy),
  with nothing pointing the way after a survey. Fixed partly by naming (Multisite › Sites) and the
  status line; the depth stays.
- **Remove sits right beside Draw**, and the Delete key removes the chosen site. During a loop
  that can't be undone. Needs a confirmation.
- **Group editing** needs a double-click on an empty cell; nothing says so.
- **Drawing a square site is 5 clicks.** Drag-to-draw would be 1 (recommendation).
- **Retrying a failed calibration** needs a new folder, which the message doesn't say.

### Lag and jank
- Opening the Autonomy panel: 0.9–1.4 s, the app frozen for it.
- Loop redraw cost grows with the length of the loop (250 ms per redraw at 100,000 events).
- After a click on Pause or Run now: 0.3–0.7 s before anything changes on screen. That isn't
  slowness of the screen (the loop takes commands between steps), but without feedback it feels
  like nothing happened.
- No flicker or visible re-render was seen.

### Missing feedback
- Autonomy panel with no survey: blank axes and "0 units". With a survey: never says how to draw
  (only a tooltip on the Draw button does).
- Before the loop: never says the next step is to start Longitudinal Monitor.
- After Pause / Run now / a drawn site during a loop: nothing until the loop takes it.
- A command the loop refuses (`CommandRefused`) only shows as a line in the event list. The status
  line is then overwritten with "N units".
- **Fixed already, in Part C:** after Stop, the panel kept saying "Loop running" and still
  offered drawing. Nothing told it the run had ended.
- After a successful run, Acquisition says "stopped", the same word as for a run you stopped.
- The calibration failure box says "try another spot" but not how. After it, the folder is
  refused for a retry with "move out acquisition".
- "Run now" and "Pause" are greyed out until a site is chosen, with no hint.

### Inconsistency
- The same thing has two names: "unit" in the Autonomy panel, "units" in the program's settings
  ("The regions drawn over the survey").
- Reopened loops say "Loop finished" whether they finished or were stopped.
- "Delete" (Autonomy) and "Close" / "Remove" (Data Library, Devices) for similar actions.
- Panel menu groups: "Spectral", "2D Image Views", "Autonomous", "Mask editors": three are
  kinds of view, one is a way of working.

### Visual hierarchy
- **First screen:** the most prominent thing is "Nothing open" in the Data Library. It should be
  the Acquisition panel.
- **Acquisition:** the program name and the settings cards are large; the Start button is a 36×25
  icon. The most important action is the smallest thing.
- **Autonomy panel:** the map is the right focus, but it opens at 628×483 as a loose window. The
  right side gives equal weight to three lists.
- **Curves:** the group mean is bold, as it should be, but the per-site lines (alpha 60 of 255)
  can barely be seen on black.

### Accessibility basics
- **Contrast** (measured from the screenshots): normal text is 10.9–13.1:1 (good). The "Choose
  folder" placeholder is 2.9:1. Greyed-out buttons are 2.3:1, which also makes it hard to tell
  what they are.
- **Keyboard:** Tab moves through the Acquisition panel, but no focus mark is visible on buttons
  or boxes. A keyboard user can't see where they are.
- **Text size:** 12–13 px throughout; the map's numbers are small but readable. No text below
  10 px.
- **Colour alone:** site groups are told apart only by outline colour, but each outline also
  carries its number, and the table names the group. Acceptable.

## Screenshots (before)

All in `ux_audit_assets/`, named `before_*.png`:

| File | State |
|---|---|
| `before_first_launch.png` | first launch: the Data Library tab |
| `before_loose_windows.png` | after adding the dish: its panel as a loose window over the app |
| `before_acquisition_no_devices.png` | Acquisition with no devices: "needs …" |
| `before_calibration_ready.png` | Calibration ready to run: the tiny ▶ |
| `before_calibration_failed.png` | the calibration failure box |
| `before_retry_refused.png` | the retry refused: "move out acquisition" |
| `before_survey_ready.png` | Survey ready: the earlier-recordings card |
| `before_empty_panels.png` | Autonomy and Curves with nothing to show |
| `before_sites_no_sites.png` | Autonomy with a survey and no sites: "0 units" |
| `before_sites_drawing.png` | drawing a site, corner by corner |
| `before_sites_chosen.png` | three sites, one chosen |
| `before_sites_loop_running.png` | the loop running: table, Queue, Recent |
| `before_loop_running_whole_screen.png` | the whole screen during a loop: loose windows |
| `before_curves.png` | Curves during a loop |
| `before_reopened_stopped_loop.png` | a stopped loop reopened: "Loop finished", stale times |
| `before_data_library.png` | Data Library: every recording named "acquisition" |

---

# Part C: Reconciling #67 with #68 (done first)

## Which branch is based on which
- `git merge-base` of the two branches is `ff6a8a4`, an **early** version of #68's work.
- #67 (`claude/autonomy-port`) holds one commit of its own (`e5a2b4b`, the panels), and a merge
  of that early #68.
- #68 (`claude/autonomy-part1`) has 12 commits since `ff6a8a4`.
- GitHub lists #67's base as `claude/autonomy-part1`, so #67 *is* meant to sit on #68, but it was
  built against #68 as it stood 12 commits ago. A plain rebase would not have been enough: the
  interfaces the panels use had changed.

## What #67 relied on that #68 changed

| #67 used | #68 has instead | What I did |
|---|---|---|
| `Unit(id: int, …)`, numbered by the panel | `Unit(key, …)` made with `Unit.new`; the loop gives each its number | The panel tracks sites by key and shows the number. Before a loop it numbers them in order, as the loop will. |
| The `PAUSED_UNITS` input and the `RUN_NOW` request | One `UNIT_COMMAND` request carrying add / edit / remove / pause / resume / run now | Every change during a loop is now a command. |
| Editing `UNITS` while the loop runs | `UNITS` is only what a loop starts on | Before a loop the panel edits `UNITS`; during one it sends commands; after one it is read-only. |
| `units_file.py` writing `units.json` | Panels never write files; the loop's `UnitsOutput` is the record | Deleted `units_file.py`. |
| `PlacedImageOutput` for the overview | `SurveyOutput.overview()` | Used it. |
| `LoopView` with untyped events and a string state | Typed events, a `UnitState` enum, `Series` | The panel words each event and state itself. |
| Polling the library every 500 ms for the newest loop | `ProgramOutput.subscribe` and `Library.subscribe` | The panel follows "the loop that read this survey" by subscription. |
| Curves assumed the monitor's measurement | `Series(routine, name, points)` | Curves lets you pick a measurement. |

## Leaks I found, and how I resolved them
- **Writing files from a panel** (`units.json`): removed; #68 owns the record of a loop's units.
- **Keeping its own unit numbers**: removed; numbers come from the loop.
- **Showing internal keys**: never shown; the screen shows only the number.
- **Finding "the" loop by type alone** (the newest loop anywhere): replaced by "the loop whose
  recording read this survey", using the recording's public list of what it read. Two experiments
  open side by side no longer confuse it.

## Gaps in #68's public interface, and the smallest additions
1. **"Has this survey finished making its overview?"** `SurveyOutput.overview()` raises an error
   until then, and a panel shouldn't use errors to ask a question. Added `SurveyOutput.has_overview()`
   (four lines).
2. **"Has the loop stopped?"** When a run ends, the engine flips `recording.finished` but sends no
   change notice to panels. The only notice is the requests collection saying what is accepted has
   changed. The panel listens there. No addition was needed, but it is subtle: the engine could one
   day tell the library when a recording finishes. Listed under "Found but not fixed".

## Checks
- ruff, format, pyright and import-linter pass on the whole project. Pyright shows 0 errors when
  checking for Windows (CI's platform); on Linux it reports 8 Windows-only driver calls, the same 8
  as on #68.
- The repository has no automated tests. The full pipeline was run in the app: devices,
  calibration, survey, drawing 3 sites, the loop, pause, run now, a site drawn mid-loop, Curves,
  Stop, then reopening the experiment. Every step worked.
- The app starts and runs (`pyrpoc --simulated`).

## Decisions made on your behalf (Phase 0)
- **Drawn sites are still lost when the app restarts** (`UNITS` isn't saved with the session, as
  #68 chose). Kept: #68 made this choice so autosave doesn't rewrite thousands of outlines.
- **One panel to draw and watch**, not two. The map is the same for both, and splitting it would
  make a user switch panels mid-experiment.
- **Before a loop, sites are numbered in drawing order**, which matches what the loop will give them.
