# core: the nouns

This folder defines what each piece of pyrpoc is. Everything else in the
package builds on these nouns.

## The flow

```
 ProgramInputs ──▶ Run ──▶ ProgramOutputs ──▶ Library ──▶ panels show them
       ▲                                                        │
       └────────────── panels set program inputs ◀──────────────┘
```

Everything the software does is this loop:
1. Program inputs feed a run.
2. A run publishes program outputs.
3. Outputs sit in the Library, where panels show them.
4. What the user does in panels sets program inputs.

Collections are the only place parts of the app meet. The Acquisition panel
is a view of the selected program's inputs; other panels never depend on
which program is selected. A panel that makes something programs use, like
the Mask Editor's masks or a click on an image, writes it into one program
input declared in core beside its type. How a program uses it is that
program's own inputs, like the line each mask drives. Runners may watch the
program inputs, as arm and run waits for the next pick.

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
| **PixelPick** | The last click on one pixel of an image: the pixel, and the image's id, label and the inputs it was acquired with. Image panels write it into `PICK`; no program shows it, and arm and run turns the next one into what its program reads, such as galvo volts. |
| **Mask** | A drawn region, with a label and an id that stays the same through edits. `MASKS` holds every drawn mask, in order. Only a Mask Editor edits it; programs that use masks show it read-only and say what each mask does with their own inputs. Held inline: in the session and in recordings each is its id, label, shape and pixels, packed one bit each and base64-encoded. |

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
| **Panel** | A dock widget: something the user sees and acts in. It reads the collections and calls their operations. **The Acquisition panel is one.** |
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
| `program_inputs.py` | `ProgramInput` + `program_input_registry`, the value rules (int, float, choice, ...), `ProgramInputValues` (a run's values), `ProgramInputs` (protocol), `Editor`, `FieldValues` |
| `input_section.py` | `InputSection` |
| `pixel_pick.py` | `PixelPick`, and `PICK`, the program input holding the last one |
| `mask.py` | `Mask`, and `MASKS`, the program input holding every drawn mask |
| `data.py` | `Data`, `ArrayData`, `Image`, `DecayHistogram`, `RawSamples`, `Spectrum`, `Mosaic` and its `MosaicTile`, `DATA_KINDS` |
| `program_output.py` | `ProgramOutput`, `Provenance`, `Origin` |
| `library.py` | `Library` (protocol) |
| `saving.py` | `SaveTarget`, `Writer` + `writer_registry` |
| `panels.py` | `Panel`, `DataPanel`, `SourcePicker`, `DevicePanel` + their registries. The only file here that uses Qt. |
