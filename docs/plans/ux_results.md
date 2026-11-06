# Results: the multisite screens, before and after

What changed in PR #67, for someone without a technical background. The audit behind it, with the
reasons for each change, is `ux_audit.md`. Screenshots are in `ux_audit_assets/`: `before_*.png`
and `after_*.png`.

Words used here:
- **panel**: one of the boxes inside the app window.
- **site**: a region of the dish that the loop keeps coming back to. The code calls it a *unit*.
- **loop**: the Longitudinal Monitor program, which visits each site again and again.

---

## 1. Before and after, job by job

### The window
| | Before | After |
|---|---|---|
| First screen | Data Library, saying "Nothing open" (`before_first_launch.png`) | Acquisition, where every job starts (`after_first_launch.png`) |
| Panels you add | Open as loose windows on top of the app, covering the menu and each other (`before_loop_running_whole_screen.png`) | Open inside the window: displays on the right as tabs, device settings beside Devices (`after_loop_running_whole_screen.png`) |
| Start / Stop | Two small unlabelled icons, ▶ and ■ (`before_calibration_ready.png`) | "▶ Start" and "■ Stop" (`after_calibration_ready.png`) |
| After a run ends by itself | "stopped" | "finished" ("stopped" only if you pressed Stop) |

### Calibrate, then survey
| | Before | After |
|---|---|---|
| Missing calibration | "needs a calibration (GalvoCalibrationOutput) saved in this folder" | "needs a calibration saved in this folder" |
| Folder not clean, or a retry after a failed calibration | "needs a folder holding only what it reads (move out acquisition)" (`before_retry_refused.png`) | "needs an empty folder (choose a new folder, or move out acquisition)" (`after_retry_refused.png`) |

### Draw sites (the Sites panel, called Autonomy before)
| | Before | After |
|---|---|---|
| No survey yet | Blank axes, "0 units" (`before_empty_panels.png`) | "Run a Survey first: its map of the dish is where you draw sites." (`after_empty_sites.png`) |
| Survey, no sites | "0 units"; how to draw only in a tooltip (`before_sites_no_sites.png`) | "Click Draw site, then click the corners of a region on the map to follow." (`after_sites_no_sites.png`) |
| While drawing | No guidance (`before_sites_drawing.png`) | "Click each corner of the site on the map. Click the first corner again, or double-click, to close it. Press Esc to stop drawing." (`after_sites_drawing.png`) |
| Sites drawn | "3 units" (`before_sites_chosen.png`) | "3 sites drawn. Next: start Longitudinal Monitor in Acquisition to follow them." (`after_sites_chosen.png`) |
| Opening the panel | 0.9–1.4 s, app frozen | about 0.5 s |

### Watch and steer the loop
| | Before | After |
|---|---|---|
| Layout beside the map (`before_sites_loop_running.png` / `after_sites_loop_running.png`) | Heading, two greyed-out buttons, "Queue, overdue first" (a repeat of the table), "Recent" (every routine step) | Heading, "Click a site on the map or in the list to pause it or visit it next.", then Activity |
| With a site chosen | "Run now" / "Pause" | "Pause site 2" / "Visit site 2 next" |
| After clicking Pause | Nothing for 0.3–0.7 s, then the row changes | "Pausing site 2…" at once (3–5 ms), then "Site 2 paused." (`after_sites_paused.png`) |
| A change the loop refuses | Only a line in the event list; the status line said "4 units" | "The loop could not pause site 9: …" on the status line |
| Removing a site mid-loop | Instant, permanent, no question | Asks first, naming the site, defaulting to "Keep it" (`after_remove_confirm.png`) |
| What Activity shows | "unit 3 monitor due", "unit 3 intensity measured", "unit 3 scanned, found", over and over | Only what matters: added, changed, removed, paused, resumed, lost track, found again, finished, failed, refused |
| A paused site's next visit | "now" | blank |
| Reopening a stopped loop | "Loop finished, 12.5 min", stale "in 58 s" times (`before_reopened_stopped_loop.png`) | "The loop stopped early, 13 min in. Start Longitudinal Monitor again with this folder to carry it on." Next visits are blank (`after_reopened_stopped_loop.png`) |

### Read the results (Site Curves, called Curves before)
| | Before | After |
|---|---|---|
| Nothing to show | Bare axes from 0 to 1 | "No measurements yet: each group's measurements appear here once a loop has measured its sites." (`after_empty_curves.png`) |
| Each site's own line | Barely visible (`before_curves.png`) | Readable behind the group's mean (`after_curves.png`) |
| Names | "monitor: intensity", "ungrouped", "Intensity (x0.001)" | "intensity", "no group", plain values |

### Clicks per job

| Job | Before | After |
|---|---|---|
| 1 Set up devices | 7 | 7 |
| 2 Calibrate | 8 | 8 |
| 3 Survey | 3 | 3 |
| 4 Draw 3 sites | 22 | 22 |
| 5 Start the loop | 3 | 3 |
| 6 Steer: pause one, visit one next, add one | 11 | 11 |
| 7 Curves | 4 | 4 |
| **Total** | **58** | **58** |

**The click counts did not change**, and I won't claim they did. One click moved rather than
disappeared: you no longer click past the Data Library at launch, but adding the dish now shows
its settings tab, so getting back to Acquisition is still one click. What did change is the cost
the counts don't show:
- no loose windows to drag out of the way after each panel opens;
- every screen in the multisite job says what to do next;
- every change during a loop is acknowledged at once.

The biggest remaining saving would be drawing a site by dragging a rectangle: 1 action instead
of 5, which would take job 4 from 22 clicks to 10. See "Not fixed".

---

## 2. Speed

Measured in the app the same way before and after. Panel opening includes a fixed 0.2 s wait in
the test script, both times.

| What | Before | After |
|---|---|---|
| App start | 0.64–0.87 s | 0.63–0.71 s |
| Open the Sites panel | **0.9–1.44 s** (app frozen) | **0.47–0.58 s** |
| Colour the survey map (part of the above) | 303 ms | 35 ms |
| Something on screen after Pause / Visit next | **0.3–0.7 s** (nothing until the loop acts) | **3–5 ms** |
| The loop's summary at 100,000 events | 54 ms, growing with the loop's length | **0.9 ms**, flat |
| Each Sites redraw during a short loop | 23 ms (median) | 21–25 ms (unchanged) |
| Switch program | 31–77 ms | 29–57 ms |
| Start → status changes | 25–31 ms | 15–22 ms |

The loop itself still takes 0.3–0.75 s to carry out a pause: that's when it gets to the command
between steps, not the screen. The difference is that you now see "Pausing site 2…" while it
does.

---

## 3. What I renamed

Only what the screen says. The code keeps its own names, and the panels keep their internal keys,
so saved layouts still open them.

| Before | Where | After |
|---|---|---|
| Autonomy | panel name | **Sites** |
| Autonomous | Add Display heading | **Multisite** |
| Curves | panel name | **Site Curves** |
| unit, Draw unit, "Unit" column, "3 units" | Sites panel | **site**, **Draw site**, "Site", "3 sites" |
| Delete | Sites panel | **Remove** |
| Run now | Sites panel | **Visit site N next** |
| Pause / Resume | Sites panel | **Pause site N** / **Resume site N** |
| waiting, paused, lost, done | Status column | **Waiting, Paused, Lost track, Finished** |
| State, Next | column headings | **Status, Next visit** |
| Queue, overdue first | list | removed |
| Recent | list | **Activity** |
| "unit 3 monitor due", "… measured", "… scanned, found" | Recent list | hidden; **"site 3 lost track"**, **"site 3 found again"** shown instead |
| Loop running, 4.2 min | heading | **Loop running, 4 min so far** |
| Loop finished (after Stop) | heading | **Loop stopped early, 13 min in** |
| No running loop takes unit commands | status line | **No loop is running, so nothing was changed.** |
| monitor: intensity | Curves menu | **intensity** (routine added only if two share a name) |
| ungrouped | Curves legend | **no group** |
| stopped (after a normal end) | Acquisition status | **finished** |
| a calibration (GalvoCalibrationOutput) | Acquisition "needs …" | **a calibration** |
| a folder holding only what it reads (move out …) | Acquisition "needs …" | **an empty folder (choose a new folder, or move out …)** |

---

## 4. Decisions I made on your behalf

Each one can be overruled; most are a single commit to revert.

1. **"Site" is the screen's word for a unit.** It's the plain word, and it matches "multisite".
2. **One Sites panel, for drawing and for watching**, not two. Same map for both; no switching
   panels mid-experiment.
3. **Added panels open inside the window**: displays as tabs on the right, taking a little more
   than half the width; device settings as tabs beside Devices. Floating them covered the very
   panels they sit beside. They can still be dragged out.
4. **Drawn sites are still forgotten when the app restarts.** That's #68's choice, so autosave
   doesn't rewrite thousands of outlines. Once a loop starts, its sites are saved with it.
5. **Before a loop, sites are numbered in drawing order**, matching the numbers the loop then gives.
6. **Removing a site asks first only during a loop.** Before a loop, removing costs nothing.
7. **The "clean folder" rule stays.** It traps you after a failed calibration, but changing it is
   #68's design. I only made the message say what to do.
8. **Activity hides routine-level events** (each visit, measurement and focus check). They are
   still in the saved log.
9. **"finished" vs "stopped"** depends on whether you pressed Stop.
10. **Words on every run button, app-wide**, not only in multisite programs: Start, Continuous,
    "Park at point…" and so on.
11. **Three small additions inside #68's code:** `SurveyOutput.has_overview()` (has the survey made
    its map yet?), `LoopLogOutput.events_after()` (only the new events), and the loop log keeping
    its summary as it goes (same results, no longer slower in long loops). No change to #68's
    design.
12. **Tests and a test step in CI.** The repo had none. I added pytest (the standard Python test
    runner) and 21 tests. The lock file was rewritten in uv's newer format; no package versions
    changed apart from pytest and the three small packages it needs.
13. **#67's description rewritten** for what it now is: the multisite UI on top of #68.

---

## 5. Things I found but did not fix

Most important first.

1. **The simulated calibration fails about 1 run in 4 at the default speed** ("the measured shifts do
   not fall on a line: try another spot"). Not a screen problem; #68 already lists it. It's the
   first thing a new user runs, so it matters.
2. **A failed calibration blocks its own folder.** The message now says what to do, but the trap
   remains. Recommendation: let a failed run of the same step not count against a retry in that
   folder. That's a change to #68's design, so it's yours to decide.
3. **Drawing a site takes 5 clicks.** A drag-to-draw rectangle would make it 1 (job 4: 22 clicks
   → 10). It's a new feature, so I left it.
4. **Every recording is named "acquisition", "acquisition_2", "acquisition_3"**, so lists and
   messages can't tell calibration from survey from loop. Recommendation: default the name to the
   program's name. It changes how every run is named, so I left it.
5. **Keyboard focus is invisible**, and placeholder and greyed-out text is low-contrast (2.3–2.9:1).
   Both live in the theme, a compiled stylesheet shared by the whole app.
6. **The loop's refusal reasons still say "unit"** ("the loop has no such unit"). The words come
   from inside #68's loop.
7. **The engine doesn't announce when a recording finishes.** The Sites panel learns of it
   another way (the run stops accepting commands), which works but is subtle. A proper notice
   would help every panel.
8. **A site drawn outside the surveyed map is accepted**, and the loop takes it. A warning while
   drawing would help.
9. **Sites often show "Lost track" on their first visit** in the simulation, then "found again" a
   minute later. That's tracking, not the screen, but a new user will wonder.
10. **The group average jumps when a site joins mid-loop**, because the average then covers a
    different set of sites. That's a choice about the analysis.
11. **Smaller:** the Sites panel is still four menu levels deep (Panels › Add Display ›
    Multisite › Sites). "Source: Latest" on every data panel doesn't say what it follows. The
    settings cards summarise as one dense line. The square map leaves black space in a tall panel.
12. **Not tested on Windows itself.** The app targets Windows; I ran it on Linux with stand-ins for
    three Windows-only driver calls. Folder dialogs and a real window manager's placement weren't
    exercised.
