# core: the nouns

This folder defines what each piece of pyrpoc is. Everything else in the
package builds on these nouns.

## The flow

```
 ProgramInputs ──▶ Run ──▶ ProgramOutputs ──▶ Library ──▶ panels show them
       ▲                                                        │
       └──────────── panels set program inputs ◀───────────────────┘
```

Everything the software does is this loop:
1. Program inputs feed a run.
2. A run publishes program outputs.
3. Outputs sit in the Library, where panels show them.
4. What the user does in panels sets program inputs.

Collections are the only place parts of the app meet. Any panel may read or
set a program input, and a runner may watch them. Nothing is routed through
the Acquisition panel: it is one view of the program inputs among others.

**Two words used throughout:**
- **Collection.** Something the app keeps in memory while it runs, holding
  the current set of one kind of thing. There are three: the **device
  inventory** (the hardware), the **ProgramInputs** (every setting's current
  value) and the **Library** (the open program outputs). Any part of the app
  can look in a collection, change it through its own operations, and be told
  when it changes.
- **Working state vs screen state.** Working state is what the app is: the
  three collections, and where runs save. Screen state is only how it's
  shown: which program the Acquisition panel shows, dock layout, zoom.

## The nouns

### Running

| Noun | Definition |
|---|---|
| **Device** | A piece of hardware the microscope has. Found through `device_registry`. |
| **Program** | A declaration of one experiment: the devices it needs, the program inputs it reads (laid out in input sections), the program outputs it emits, and the runners that can start it. Its `run` does the work. |
| **Run** | One execution of a program. It gets a frozen copy of the program's inputs, claims its devices, and publishes into its program outputs. |
| **RunContext** | What a running program is handed: its input values, its devices, its program outputs, and cancel. It reaches an output with `output_for`. |
| **Runner** | A way to start a program (once, continuously, arm and run). It's declared by the program, and its controls are drawn by whichever panel shows the program. |

### Program inputs: user → program

| Noun | Definition |
|---|---|
| **ProgramInput** | The declaration of one value a program reads, like dwell time or a mask. It has a key, a label, a default and rules for valid values. **The same key means the same value in every program that declares it.** That's how a setting carries across modality changes. Declared once, next to the code that shares it, e.g. `DWELL_TIME_US = float_input("scan.dwell_time_us", ...)`; a second declaration of a key with other rules fails. |
| **InputSection** | A labelled list of program inputs. Purely layout: how a panel groups a program's inputs on screen. |
| **ProgramInputs** | The collection of current values for every program input, by key. Any panel may read it, set a value (validated by the input's rules) or subscribe. It's saved in the session, and a run's values go into its recording. The app's one collection is `ProgramInputsStore` (`engine/acquisition/program_inputs.py`). |
| **InputTarget** | One place a panel can put a value it makes: one slot of one program input. A program input's rules say what kind of value it `accepts` (a `Mask`, a `PixelPick`), the `slots` it has for one (the whole value, or e.g. one Modulation line), and how to `place` one there. `targets_for(kind, inputs)` lists them; a panel lets the user choose one, which is its screen state, and writes through it. |
| **PixelPick** | A click on one pixel of an image: the program output clicked on, plus `x` and `y` in its pixels. The Target point accepts one and turns it into galvo volts. |
| **Mask** | A drawn region, as a program input value. A Mask Editor draws it straight into the ROI or Modulation input chosen in it. It carries a label saying what it was drawn on. Held inline: in the session and in recordings it is its label, its shape and its pixels, packed one bit each and base64-encoded. |

### Program outputs: program → user

| Noun | Definition |
|---|---|
| **Data** | One value a program produced: an image, a spectrum, a mosaic. Immutable. **Only programs make it.** |
| **ProgramOutput** | One named output of one run. It holds the latest `Data`, where it came from (`Provenance`), and its writer. Every published value reaches its files; memory holds only the latest. |
| **Library** | The program outputs that are open: acquired now, or loaded from disk. Nothing else. |
| **Recording** | One run's program outputs on disk, opened through its `_meta.json`. |
| **SaveTarget, Writer** | Where a run saves, and how each kind of `Data` is written and read. |

### Views

| Noun | Definition |
|---|---|
| **Panel** | A dock widget: something the user sees and acts in. It reads the collections and calls their operations, such as setting a program input through an `InputTarget`. **The Acquisition panel is one.** |
| **DataPanel** | A panel that shows one program output of the kind it `renders`, chosen with a source dropdown. |

## The rules

1. **Data flows one way.** Only a run or loading a recording creates a
   `ProgramOutput`.
2. **Panels never call programs, and programs never call panels.** They meet
   only through ProgramInputs and the Library.
3. **Every panel is equal.** None is special in code. The app hands each
   panel the collections it needs.
4. **Working state and screen state are separate.** The session saves them
   separately.

## The files

| File | Holds |
|---|---|
| `registry.py` | `Registry` |
| `device.py` | `Device` + `device_registry` |
| `device_config.py` | `DeviceConfig`: a device's settings, a dataclass whose fields carry their value rules |
| `program.py` | `Program`, `RunContext` + `program_registry` |
| `runner.py` | `Runner`, `RunnerContext`, `Control`, `Button`, `Toggle` |
| `program_inputs.py` | `ProgramInput` + `program_input_registry`, the value rules (int, float, choice, ...), `ProgramInputValues` (a run's values), `ProgramInputs` (protocol), `Editor`, `InputTarget`, `targets_for` |
| `input_section.py` | `InputSection` |
| `pixel_pick.py` | `PixelPick` |
| `mask.py` | `Mask`, as a program input value |
| `data.py` | `Data`, `ArrayData`, `Image`, `DecayHistogram`, `RawSamples`, `Spectrum`, `Mosaic` and its `MosaicTile`, `DATA_KINDS` |
| `program_output.py` | `ProgramOutput`, `Provenance`, `Origin` |
| `library.py` | `Library` (protocol) |
| `saving.py` | `SaveTarget`, `Writer` + `writer_registry` |
| `panels.py` | `Panel`, `DataPanel`, `SourcePicker`, `DevicePanel` + their registries. The only file here that uses Qt. |
