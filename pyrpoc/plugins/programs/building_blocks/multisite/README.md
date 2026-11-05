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

## Adding to it
- **A routine:** one file in `routines/` that yields `Scan`, `Wait` and `Record` steps for one unit. Add its binding in `bindings/` for the program inputs it reads, and a program that hands it to `run_loop`.
- **An instrument:** a class with the `Microscope` methods. The simulated one lives with the simulated programs, so this package never knows about the simulation.
- **A stage:** a new folder in `stages/` with its own sink protocol. Add a binding and export its `run_*` here.
