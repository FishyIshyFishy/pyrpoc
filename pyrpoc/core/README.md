# core: how pyrpoc fits together

`core/` defines the abstract pieces of pyrpoc and how they connect. Nothing
else lives here: no specific device, program, input, kind of data, value rule
or panel. Those are all built from these pieces somewhere else, and register
themselves so nothing in core has to list them.

Core should rarely change. Adding hardware, an experiment, a setting, a kind
of data or a panel never touches it. If core has to change, the architecture
is changing.

## In plain words

A **program** is one kind of experiment, such as a confocal scan or reading a
spectrum. It says which **devices** it needs and which settings it reads.
Those settings are its **program inputs**.

Every setting has one current value, kept in one place. The Acquisition panel
lets you edit the inputs of the program you've chosen. Other panels can set
inputs too: drawing masks in the Mask Editor sets the masks, and clicking an
image sets the picked pixel. A setting that two programs share keeps its value
when you switch between them.

Pressing run starts a **run**. It takes a snapshot of the program's input
values, claims its devices so no other run can use them, and does the work.
Whatever it measures, it **publishes** into its **program outputs**, such as an
image or a spectrum. Each published value is saved to disk as it arrives, if
saving is on.

Program outputs collect in the **library**. Panels show what's in the library
and let you act on it, which sets more program inputs, and the loop repeats.

```
 ProgramInputs ──▶ Run ──▶ ProgramOutputs ──▶ Library ──▶ panels show them
       ▲                                                        │
       └────────────── panels set program inputs ◀──────────────┘
```

## The collections

A **collection** is something the app keeps in memory while it runs, holding
the current set of one kind of thing. There are three:
- **Device inventory:** the hardware.
- **ProgramInputs:** every setting's current value.
- **Library:** the open program outputs.

Any part of the app can look in a collection, change it through its own
operations, and be told when it changes. Collections are the only place parts
of the app meet.

## The pieces

### Running

| Piece | What it is |
|---|---|
| **Device** | A piece of hardware the microscope has. Its settings are a `DeviceConfig`, whose fields each carry a `Field`. |
| **Program** | One kind of experiment: the devices it needs, the program inputs it reads (laid out in `InputSection`s), the outputs it publishes, and the runners that can start it. Its `run` does the work. |
| **Run** | One execution of a program. It gets a frozen copy of the program's input values, claims its devices, and publishes into its program outputs. |
| **RunContext** | What a running program is handed: its input values, its devices, its outputs, status and cancel. |
| **Runner** | A way to start a program, such as once, continuously, or armed until a pixel is picked. The program declares it. The runner adds controls through its `RunnerContext`, and whichever panel shows the program draws them. A runner may watch program inputs. |

### Program inputs: user → program

| Piece | What it is |
|---|---|
| **ProgramInput** | The declaration of one value a program reads: a key, a default, and a `Field`. **One key means one value everywhere.** The key names the value itself, never a section or a program, and is declared once with `declare`. A second declaration of a key fails. Every program and panel that uses the value imports that one declaration. |
| **Field** | The rules a value follows: how it is checked (`coerce`), saved (`encode`) and loaded (`decode`). It can also supply its own `Editor` widget. Bad values raise `ParameterError`. |
| **ProgramInputs** | The collection of current values, by key. Any part of the app may read a value, set one (checked by its `Field`) or subscribe. It is saved with the session. |
| **ProgramInputValues** | The values one run was given. A program reads them as `ctx.inputs[SOME_INPUT]`, and they go into the run's recording. |
| **InputSection** | A heading and the inputs under it. It is layout only: each program writes its own to group its inputs on screen. A section never appears in a key, a session or a recording. |

### Program outputs: program → user

| Piece | What it is |
|---|---|
| **ProgramOutput** | One named output of one run. Every value published into it is one `Data` of its declared kind. Memory holds the latest value, where it came from (`Provenance`), and its writer. |
| **Data** | One value a program produced. It is immutable, and **only programs make it**. Each kind registers in `data_kind_registry` under its name, which is how a recording on disk finds its class again. |
| **Library** | The program outputs that are open: acquired now, or loaded from disk. Nothing else. |
| **SaveTarget, Writer** | Where a run saves, and how one kind of `Data` is written to files and read back. Writers register in `writer_registry`. |

### Views

| Piece | What it is |
|---|---|
| **Panel** | A dock widget: something the user sees and acts in. It reads the collections and calls their operations. The Acquisition panel is one like any other. |
| **DataPanel** | A panel that shows one program output of the kind it `renders`, chosen with a `SourcePicker`. |
| **DevicePanel** | One device's controls and configuration, in a dock of its own. |

## The rules

1. **Data flows one way.** Only a run, or loading a recording, creates a
   `ProgramOutput`.
2. **Panels and programs never call each other.** They meet only through the
   ProgramInputs and the Library.
3. **Every panel is equal.** None is special in code. The app hands each
   panel the collections it needs.
4. **Working state and screen state are separate.** Working state is what the
   app is: the three collections, and where runs save. Screen state is only
   how it's shown: dock layout, zoom, which program a panel shows. The session
   saves them separately.

## Where the specific things live

Each kind of thing registers through the registry in the core file that
defines it. Nothing else lists them.

| To add | Put it in | Registers with |
|---|---|---|
| A device | `plugins/devices/` | `device_registry` |
| A program | `plugins/programs/` | `program_registry` |
| A program input, shared by programs and panels | `plugins/program_io/inputs/` | `declare` |
| A kind of data, and how it is saved | `plugins/program_io/outputs/` | `data_kind_registry`, `writer_registry` |
| A panel | `plugins/panels/` | `data_panel_registry`, `device_panel_registry` |
| A common value rule, like a whole number or a choice | `utils/value_rules/` | (none: inputs and device settings use them directly) |

## The files

| File | Holds |
|---|---|
| `registry.py` | `Registry` |
| `device.py` | `Device` + `device_registry` |
| `device_config.py` | `DeviceConfig`, and how its fields carry their `Field` |
| `program.py` | `Program`, `RunContext` + `program_registry` |
| `runner.py` | `Runner`, `RunnerContext`, `Control`, `Button`, `Toggle` |
| `program_input.py` | `ProgramInput` + `declare`, `Field`, `Editor`, `ProgramInputValues`, `ProgramInputs`, `InputSection` |
| `program_output.py` | `ProgramOutput`, `Data` + `data_kind_registry`, `Provenance`, `Origin` |
| `library.py` | `Library` |
| `saving.py` | `SaveTarget`, `Writer` + `writer_registry` |
| `panels.py` | `Panel`, `DataPanel`, `SourcePicker`, `DevicePanel` + their registries. The only file here that uses Qt. |
