# multisite

Multisite imaging of a dish in three stages:
1. **Calibration** measures how far the scanner moves per volt, using the stage as the ruler.
2. **Survey** maps a region tile by tile. It keeps focus, measures how fast things move, and places the tiles into one overview.
3. **Loop** runs routines on chosen regions of the dish (*units*), tracking each unit and keeping it in focus.

Units can be drawn by a person or found by software; the stages don't care which.

Programs import only this package's root (`__init__.py`). Everything below it is internal.

## Programs
A multisite program subclasses one of three bases in `bindings/programs.py`:

| Base | Reads | Fills | Needs before it starts | The program gives |
|---|---|---|---|---|
| `CalibrationProgram` | | `calibration`, `view`, `log` | `clean_folder` | `microscope` |
| `SurveyProgram` | `calibration` | `survey`, `log` | `clean_folder` | `microscope`, `overscan` |
| `LoopProgram` | `survey`, `calibration` | `log`, `units`, `frames` | both below | `microscope`, `overscan`, `routines` |

"Reads" are program inputs too (`CALIBRATION`, `SURVEY`), with no card: the engine sets each to the newest output of its type in the save folder whose run completed, so a step stopped partway is never read. Each base also declares the inputs its stage reads, with their cards, and `run`s its stage. A program adds its devices, its own inputs (extending the base's `inputs`, `cards` and, for a loop, `live`), and where it is listed. Two programs of a kind therefore always read, fill and refuse the same things, so a panel can rely on what every multisite program publishes.

## Layers
Each layer imports only the layers below it. `uv run lint-imports` enforces this.

| Layer | What it is | Knows about |
|---|---|---|
| `bindings/` | Each stage bound to a pyrpoc run: its program inputs, what it reads and emits, adapters from outputs to sinks, its `run_*`, and the program base that calls it. Also `RigMicroscope`. | everything below, plus `RunContext`, devices and ProgramOutputs |
| `routines/` | What a loop does to one unit, such as `monitor` | the routine contract in `stages/loop/routine.py` |
| `stages/` | `calibration/`, `survey/` and `loop/`, which never import each other | `interfaces/`, `analysis/`, `building_blocks/focus`, and value types from `program_io` |
| `interfaces/` | What a stage is given: `Microscope`, `DigitalLine`, `FieldOfView`, `Overview`, `Cancellation` | value types from `program_io` |
| `analysis/` | Maths on arrays: noise, features, shifts, content, stitching, motion, plane fits | numpy and OpenCV only |

## How a stage works
A stage is given:
- a `Microscope`;
- a `Cancellation` (a `RunContext` already satisfies it);
- frozen settings;
- a sink protocol to write into.

| Stage | Writes through | Also given |
|---|---|---|
| Calibration | `CalibrationSink` (and it returns a `GalvoCalibration`) | |
| Survey | `SurveySink` | |
| Loop | `LoopSink` | a `Steering` protocol and a `SurveyMap` |

A stage never sees a run, a device or a ProgramOutput. It can be driven by fakes, and the outputs can change without touching the stage.

## What a stage says while it works
Every stage logs typed events through its sink as it works, and each binding writes them to the run's `log` output. Each event's `text()` also goes to the run's status line. The three logs are daughters of `LogOutput`, so a panel that shows a `LogOutput` shows any of them.

| Stage | Log | Saying what it is doing now |
|---|---|---|
| Calibration | `CalibrationLogOutput` | `FeatureFound`, `AxisMeasuring`, a `StageMoved` per move, then `AxisMeasured` |
| Survey | `SurveyLogOutput` | `TileVisiting`, `TileFocusing` and `TileTaken` per tile; `MotionRevisiting`, `MotionMeasured`, `Stitching`, `SurveyEnded` |
| Loop | `LoopLogOutput` | `MovingToUnit`, `FocusingAtUnit` and `ScanningUnit` before each scan, ended by its `UnitScanned` |

A loop is doing the latest `LoopActivity` until the `UnitScanned` after it; with none since, it is waiting for the next step due. `LoopLogOutput.view().doing` works this out.

## Before a stage starts
Each base's `preflight` asks what its stage needs (`bindings/preflight.py`), so a start that cannot work is refused before its recording opens, in the same list as everything else it needs:
- `clean_folder`: the folder holds only the earlier steps it reads (and, for a loop, the runs of a loop it carries on).
- `starting_units`: a fresh loop has units drawn, each fitting in one field of the survey; a loop carried on has none drawn, since it starts from its scratch.

## Units in a loop
- **Identity.** A `Unit` has a key it keeps for life, made by `Unit.new` wherever the unit comes from: drawn by a person or found by software. The loop gives each unit a number as it joins (`Roster`) and never gives a number twice, so a unit deleted and drawn again is a new unit with a new number.
- **The units are a live input.** A loop starts on `UNITS` and reads it again between steps through `Steering`. Each `Unit` carries whether it is paused and how many times it was asked to run now. The loop compares the units it is given with those it took last:
  - a unit no longer given leaves;
  - a new one joins;
  - a changed one takes its new outline or group, is paused or resumed, or runs ahead of everything due when its count rises.

  A new outline says where the unit is now, so tracking starts again from there. A change the loop cannot make is logged as `UnitRefused`. Only a unit given before and not now leaves, so a loop carried on, which starts with no units given, loses none until a panel gives it its units back from the `units` output.
- **What it publishes.** `UnitsOutput` holds every unit the loop has had, as a `UnitRecord`; `LoopLogOutput` holds every event, one typed class per kind; `UnitFramesOutput` holds every frame.

## The scratch
While a loop runs, it keeps `loop_scratch.jsonl` in its folder: the name of each recording that ran it, and a copy of every event. When every routine has ended, the scratch goes.

If a loop is stopped or cut short, the scratch stays. Starting the loop again in that folder carries it on:
- `clean_folder(carries_on=True)` lets the recordings the scratch names stay in the folder;
- `resumption()` replays the events into each unit, its routines' progress and history, and the focus offset;
- the new run's log starts with every earlier event, so the run that finishes the loop tells all of it.

A loop carried on starts from the scratch, not from `UNITS`, which `starting_units` makes sure is empty. A routine carries on from its `RoutineHistory`, and each unit is taken to be where it was last found, its look learnt again from the next scan.

## Adding to it
- **A routine:** one file in `routines/` that yields `Scan`, `Wait` and `Record` steps for one unit, starting from its `RoutineHistory` (empty unless the loop is carried on). Add its binding in `bindings/` for the program inputs it reads, and a `LoopProgram` whose `routines` returns it.
- **An instrument:** a class with the `Microscope` methods. The simulated one lives with the simulated programs, so this package never knows about the simulation.
- **A stage:** a new folder in `stages/` with its own sink protocol and its events in `program_io/outputs`, logged through the sink. Add a binding, a program base that runs it, and export the base here.
