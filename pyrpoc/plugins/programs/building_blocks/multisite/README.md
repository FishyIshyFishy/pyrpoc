# multisite

Multisite imaging of a dish in three stages:
1. **Calibration** measures how far the scanner moves per volt, using the stage as the ruler.
2. **Survey** maps a region tile by tile. It keeps focus, measures how fast things move, and places the tiles into one overview.
3. **Loop** runs routines on chosen regions of the dish (*units*), tracking each unit and keeping it in focus.

Units can be drawn by a person or found by software; the stages don't care which.

Programs import only this package's root (`__init__.py`). Everything below it is internal.

## Layers
Each layer imports only the layers below it. `uv run lint-imports` enforces this.

| Layer | What it is | Knows about |
|---|---|---|
| `bindings/` | Each stage bound to a pyrpoc run: its program inputs, what it reads and emits, adapters from outputs to sinks, and the one `run_*` a program calls. Also `RigMicroscope`. | everything below, plus `RunContext`, devices and ProgramOutputs |
| `routines/` | What a loop does to one unit, such as `monitor` | the routine contract in `stages/loop/routine.py` |
| `stages/` | `calibration/`, `survey/` and `loop/`, which never import each other | `interfaces/`, `analysis/`, `building_blocks/focus`, and value types from `program_io` |
| `interfaces/` | What a stage is given: `Microscope`, `DigitalLine`, `FieldOfView`, `Overview`, `Progress` | value types from `program_io` |
| `analysis/` | Maths on arrays: noise, features, shifts, content, stitching, motion, plane fits | numpy and OpenCV only |

## How a stage works
A stage is given:
- a `Microscope`;
- a `Progress` (a `RunContext` already satisfies it);
- frozen settings;
- a sink protocol to write into.

| Stage | Writes through | Also given |
|---|---|---|
| Calibration | (nothing: it returns a `GalvoCalibration`) | a `show(frame)` callback |
| Survey | `SurveySink` | |
| Loop | `LoopSink` | a `Steering` protocol and a `SurveyMap` |

A stage never sees a run, a device or a ProgramOutput. It can be driven by fakes, and the outputs can change without touching the stage.

## Before a stage starts
Each program declares the checks its stage needs, so a start that cannot work is refused before its recording opens:
- `CleanFolder`: the folder holds only the earlier steps it reads (and, for a loop, the runs of a loop it carries on).
- `CalibrationMeasured` and `SurveyFinished`: the steps it reads ran to their end, not stopped partway.
- `StartingUnits`: a fresh loop has units drawn, each fitting in one field of the survey; a loop carried on has none drawn, since it starts from its scratch.

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
- `CleanFolder(carries_on=True)` lets the recordings the scratch names stay in the folder;
- `resumption()` replays the events into each unit, its routines' progress and history, and the focus offset;
- the new run's log starts with every earlier event, so the run that finishes the loop tells all of it.

A loop carried on starts from the scratch, not from `UNITS`, which `StartingUnits` makes sure is empty. A routine carries on from its `RoutineHistory`, and each unit is taken to be where it was last found, its look learnt again from the next scan.

## Adding to it
- **A routine:** one file in `routines/` that yields `Scan`, `Wait` and `Record` steps for one unit, starting from its `RoutineHistory` (empty unless the loop is carried on). Add its binding in `bindings/` for the program inputs it reads, and a program that hands it to `run_loop`.
- **An instrument:** a class with the `Microscope` methods. The simulated one lives with the simulated programs, so this package never knows about the simulation.
- **A stage:** a new folder in `stages/` with its own sink protocol. Add a binding and export its `run_*` here.
