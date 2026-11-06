# Handoff: PR #67, the Autonomy and Curves panels

## The goal
#67 is **UI/UX only**: the Autonomy and Curves panels for the multisite pipeline. The engine and
the non-display plugins were settled in #68. Change them only where a UI decision needs it, and
say so in the PR.

## Where things stand
- **#68**, branch `claude/autonomy-part1`, head `f2bd845`: the multisite pipeline (calibrate →
  survey → loop), unit identity and commands, the loop's scratch and resume, the run inbox, and
  recordings in parts. Clean, and every check passes.
- **#67**, branch `claude/autonomy-port`: the panels, written against an **older API**. It merged
  an early part 1 (`ff6a8a4`), not the current #68. Start by rebasing or merging it onto
  `claude/autonomy-part1`, then port the panels to the API below.
- Panel files on #67: `plugins/panels/autonomy/{panel,canvas,loop_watch,picture,units_file,units_table}.py`
  and `plugins/panels/curves/panel.py`. It also has small edits to
  `plugins/panels/mosaic/{mosaic_view,stack_view}.py`, and moves `ColourImage` to
  `utils/widgets/colour_image.py`.

## Read first
- `.claude/CLAUDE.md`: the layers, the checks, and the code style.
- `pyrpoc/core/README.md`: the pieces, now with the run inbox and recording parts.
- `pyrpoc/plugins/programs/building_blocks/multisite/README.md`: units, commands, the scratch.
- `docs/diagrams/`: `core` and `nouns` (the engine), `multisite` (the pipeline) and
  `multisite_loop` (inside a loop run). Each `.d2` is rendered to an `.svg`.

## What the user cares about
- **Encapsulation above all.** `program_io/` is inputs, requests and outputs, nothing else: no
  folder for one feature's things. Behaviour belongs to whoever owns it; a loop's behaviour
  belongs to multisite.
- **Be cautious with changes to `core/` and `engine/`.** Propose them before making them.
- **Never silence a check.** Ask instead.
- **Name things so a non-engineer reads them right**, and keep docstrings to one or two plain
  sentences.
- **Explain plainly.** The user wants to understand each decision, not just approve it.

## What changed under #67 (the API it must move to)

| #67 used | Now |
|---|---|
| `Unit(id: int, outline, group)`, numbered by the panel | `Unit(key, outline, group)`. Make units with `Unit.new(outline, group)`, which gives a key no other unit ever has. The **loop** gives each unit its visible number as it joins, and never reuses one. |
| `PAUSED_UNITS` input, `RUN_NOW` request | One request, `UNIT_COMMAND` (`program_io/requests/units.py`), carrying `AddUnit`, `EditUnit`, `RemoveUnit`, `PauseUnit`, `ResumeUnit` or `RunUnitNow`, each naming its unit by key |
| `UNITS` edited live during a loop | `UNITS` is only the units a loop **starts** on (`restored=False`). While a loop runs, every change is a `UNIT_COMMAND`. A command the loop can't carry out is logged as `CommandRefused`. |
| `units_file.py` writing `units.json` into the folder | Gone: a panel never writes files. The record of a loop's units is its `UnitsOutput` (`records`: a `UnitRecord` per unit, holding its number, unit, join and leave times, and whether it is paused). Drawn units that no loop has started on yet live only in the `UNITS` input. |
| `PlacedImageOutput` overview | `SurveyOutput.overview()` returns `(planes, Placement)` |
| `LoopView` with `UnitView.id`, a `state` string, an untyped `series`, and `recent` as screen text | `UnitView.number` and `.key`; `.state` is the `UnitState` enum; `.series` is a tuple of `Series(routine, name, points)`; `LoopView.recent` holds typed `LoopEvent`s (the panel formats them); there is also `LoopView.started_s` |
| Untyped `LoopEvent` | A class per kind, in `program_io/outputs/loop_events.py` |
| `recording.parameters` | `recording.part_of(output_name).parameters`. A continuous run edited mid-run fills `intensity`, then `intensity 2`, and so on. |

## How the loop behaves now
- Deleting a unit and drawing it again makes a new unit with a new number.
- An `EditUnit` with a new outline means "the unit is here now": tracking starts again there.
- The loop keeps `loop_scratch.jsonl` in its folder until every routine has ended. A stopped or
  crashed loop, started again in the same folder, carries on from it: `CleanFolder(carries_on=True)`
  allows its earlier runs to stay, and the units currently drawn are ignored. The log of the run
  that finishes holds the whole loop.
- The save target can't change while any run is going; the acquisition panel greys it out.

## Open UI decisions for #67
1. **Finding the loop.** The old Autonomy panel scanned the library for the newest `LoopLogOutput`
   every 500 ms. Better options: `ProgramOutput.subscribe` on the loop's log, and
   `Library.subscribe` to notice a new recording or a new part (the executor calls
   `library.notify()` when a part begins). Decide whether this is one panel or a "draw" panel
   plus a "watch" panel.
2. **Units to show before and during a loop.** Before: the `UNITS` input. During and after: the
   loop's `UnitsOutput` and its log's `LoopView` (outlines as tracked).
3. **Following new parts.** A panel showing `intensity` does not move to `intensity 2` by itself.
   This applies to every panel, not just these two.
4. **Curves.** Choose a `Series` by routine and name, not by assuming the monitor routine.
5. **Drawn units are lost on restart** (`UNITS` is `restored=False`). Decide whether that is fine.

## Known issues, not UI
- The simulated calibration sometimes fails at clock speed 20 with "shifts do not fall on a line";
  at 5 it passes.
- The simulated dish can't be carried on across an app restart: a restart creates a new dish with
  its own clock.
- A device edit applies to the hardware at once, so it can land mid-pass. The new part records it
  from the next pass.

## Test harnesses
These are scratch scripts outside the repo (session scratchpad `…\c8c57b00-…\scratchpad\`):
- `smoke_units.py`: headless calibrate → survey → steered loop → stop and resume → reload.
- `smoke_parts.py`: headless continuous run edited mid-run, giving a recording in parts.
- `smoke_gui_parts.py`: the same, in the app run off-screen.

The full scratchpad path is
`C:\Users\ishaa\AppData\Local\Temp\claude\6e59ddb06fbd7688\c8c57b00-1710-4530-8efb-ea8d65f86759\scratchpad\`.
An older off-screen GUI script for the panels, `smoke_gui_autonomy.py`, is in
`…\540dff24-f1d7-4356-8a53-9d070741883e\scratchpad\`. It uses the **old** API, but it shows how
to drive the Autonomy and Curves panels off-screen.

## Deferred (in `docs/todo.md`, agent section)
A run has two doors, its starting values and its inbox. They could become one. Not for now.
