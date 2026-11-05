# core: how pyrpoc fits together

`core/` defines the abstract pieces of pyrpoc and how they connect. Nothing
else lives here: no specific device, program, input, program output, value rule
or panel. Those are all built from these pieces somewhere else, and register
themselves so nothing in core has to list them.

Core should rarely change. Adding hardware, an experiment, a setting, a type
of program output or a panel never touches it. If core has to change, the
architecture is changing.

## In plain words

A **program** is one kind of experiment, such as a confocal scan or reading a
spectrum. It says which **devices** it needs and which settings it reads.
Those settings are its **program inputs**.

Every setting has one current value, kept in one place. The Acquisition panel
lets you edit the inputs of the program you've chosen. Other panels can set
inputs too: drawing masks in the Mask Editor sets the masks, and clicking an
image sets the picked pixel. A setting that two programs share keeps its value
when you switch between them.

Pressing run starts a **run**, and every run opens one **recording**. The run
takes a snapshot of the program's input values, claims its devices so no other
run can use them, and fills the recording's **program outputs**, such as an
image or a spectrum. Single does the program's work once. Continuous does it
again and again into the same recording until you stop it, so a long
continuous run is still one recording. Editing an input while it runs ends
that recording and starts a new one with the new values.

A program may declare some inputs **live**: a run reads their current value
whenever it chooses, between its own steps, so edits made while it runs reach
it without starting a new recording. A program may also accept **requests**,
such as "visit this unit now": a panel posts one and the running program takes
it once. The recording logs every live change and request it was given, and
when.

A program output is one named part of a recording, of one type: an
`ImageOutput`, a `SpectrumOutput`, a `MosaicOutput`. Its type decides
everything about its content: what it holds, how a program adds to it, how a
panel reads it, and how it is written to files and read back. Core knows only
that it has a name, belongs to a recording, can say it changed, and holds some
memory. Everything stays in memory while the recording is open.

The recordings saved into one folder form an **Experiment**: choosing where
runs save is choosing the Experiment. With no folder set, nothing is saved. A
note written before a run is saved with its recording. It is not a program
input: no program reads it.

A program can read the outputs of an earlier recording in the same folder. It
names the type of output it needs, not the program that made it, so programs
never depend on each other.

Open recordings collect in the **library**. Panels show what's in the library
and let you act on it, which sets more program inputs, and the loop repeats.

```
 ProgramInputs ──▶ Run ──▶ Recording ──▶ Library ──▶ panels show it
  Requests    ──▶          (its outputs)               │
       ▲                                               │
       └──── panels set program inputs, post requests ◀┘
```

## The collections

A **collection** is something the app keeps in memory while it runs, holding
the current set of one kind of thing. There are four:
- **Device inventory:** the hardware.
- **ProgramInputs:** every setting's current value.
- **Requests:** commands posted for running programs, each taken once.
- **Library:** the open recordings.

Any part of the app can look in a collection, change it through its own
operations, and be told when it changes. Collections are the only place parts
of the app meet.

## The pieces

### Running

| Piece | What it is |
|---|---|
| **Device** | A piece of hardware the microscope has: its identity, its settings (a `DeviceConfig`, whose fields each carry a `Field`) and maybe a connection. |
| **Program** | One kind of experiment: the devices it needs, the program inputs it reads, the outputs it fills, the earlier outputs it reads, the requests it accepts, and the runners that can start it. Its `run` does the work. |
| **RunContext** | What a running program is handed: its input values, the current value of its live inputs, its requests, the earlier outputs it reads, its devices, its outputs, status and cancel. |
| **Runner** | A way to start a program, such as once, continuously, or armed until a pixel is picked. The program declares it. The runner adds controls through its `RunnerContext`, and whichever panel shows the program draws them. A runner may watch program inputs. |

A run is not a piece of core: it is the engine executing one recording.

### Program inputs: user → program

| Piece | What it is |
|---|---|
| **ProgramInput** | The declaration of one value a program reads: a key, a default, and a `Field`. **One key means one value everywhere.** The key names the value itself, never a group or a program, and is declared once with `declare`. A second declaration of a key fails. Every program and panel that uses the value imports that one declaration. |
| **Field** | The rules a value follows: how it is checked (`coerce`), saved (`encode`) and loaded (`decode`). Bad values raise `ParameterError`. How a value is edited on screen is not part of its rules. |
| **ProgramInputs** | The collection of current values, by key. Any part of the app may read a value, set one (checked by its `Field`) or subscribe. It is saved with the session, except inputs declared `restored=False`, which start at their default each launch. |
| **ProgramInputValues** | The values one run was given. A program reads them as `ctx.inputs[SOME_INPUT]`. Every one is saved with the recording. |

A program lists its inputs flat. It may also lay them out in `cards` for the
Acquisition panel, which is screen layout only: a card never appears in a
key, a session or a recording.

### Program outputs: program → user

| Piece | What it is |
|---|---|
| **Recording** | What one run produced: what made it (program, name, start time, input values, devices, the earlier recordings it read), where it is saved (or nothing, when unsaved), its note, whether it is finished, and its program outputs. Two outputs are of the same recording when their `recording` is the same object. Its folder is its Experiment. |
| **ProgramOutput** | One named part of a recording, of one type. Each type is a subclass that owns its content, how programs add to it, how panels read it, and how it is written to files and read back. Many programs may fill the same type. Types register in `program_output_registry`, which is how a recording on disk finds them again. |
| **Library** | The recordings that are open: acquired now, or loaded from disk. Nothing else. You can add one, close one, list them, and subscribe. |
| **SaveTarget** | What a run is called and where it saves. Its folder is the Experiment. |

### Views

| Piece | What it is |
|---|---|
| **Panel** | A dock widget: something the user sees and acts in. It reads the collections and calls their operations. The Acquisition panel is one like any other. What a panel shows, and how it picks what to show, is its own business. |

## The rules

1. **Data flows one way.** Only a run, or loading a recording, creates a
   `Recording`.
2. **Panels and programs never call each other.** They meet only through the
   collections.
3. **Every panel is equal.** None is special in code. The app hands each
   panel the collections it needs.
4. **Working state and screen state are separate.** Working state is what the
   app is: the collections, and where runs save. Screen state is only how it's
   shown: dock layout, zoom, which program a panel shows. The session saves
   them separately.

## Where the specific things live

Each kind of thing registers through the registry in the file that defines
its base, in core or, for panels, in `utils/widgets/`. Nothing else lists them.

| To add | Put it in | Registers with |
|---|---|---|
| A device | `plugins/devices/` | `device_registry` |
| A program | `plugins/programs/` | `program_registry` |
| A program input, shared by programs and panels | `plugins/program_io/inputs/` | `declare` |
| A type of program output, with its files | `plugins/program_io/outputs/`, one file each | `program_output_registry` |
| A panel showing program outputs | `plugins/panels/` | `data_panel_registry` (`utils/widgets/data_panel.py`) |
| A device's own panel | beside its device in `plugins/devices/` | `device_panel_registry` (`utils/widgets/device_panel.py`) |
| A common value rule, like a whole number or a choice | `utils/value_rules/` | (none: inputs and device settings use them directly) |

The bases panels build on live in `utils/widgets/`: `DataPanel` with its
`SourcePicker`, and `DevicePanel`, given its device and a `DeviceLink` to it.
Each keeps its own registry, since each is built from different collections.

## The files

| File | Holds |
|---|---|
| `registry.py` | `Registry` |
| `device.py` | `Device` + `device_registry` |
| `device_config.py` | `DeviceConfig`, and how its fields carry their `Field` |
| `program.py` | `Program`, `RunContext` + `program_registry` |
| `runner.py` | `Runner`, `RunnerContext`, `Control`, `Button`, `Toggle` |
| `program_input.py` | `ProgramInput` + `declare`, `Field`, `ProgramInputValues`, `ProgramInputs` |
| `requests.py` | `Requests` |
| `recording.py` | `Recording` |
| `program_output.py` | `ProgramOutput` + `program_output_registry` |
| `library.py` | `Library` |
| `saving.py` | `SaveTarget` |
| `panels.py` | `Panel`. The only file here that uses Qt. |
