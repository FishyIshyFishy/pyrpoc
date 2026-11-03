# core: the nouns

This folder defines what each piece of pyrpoc is. Everything else in the
package builds on these nouns.

The architecture is still being reached in phases. A line marked
*(from phase N)* describes how it will be once that phase is done, and says
what the code does today.

## The flow

```
 ProgramInputs ──▶ Run ──▶ ProgramOutputs ──▶ Library ──▶ panels show them
       ▲                                                        │
       └──── panels set program inputs, or answer input requests ◀┘
```

Everything the software does is this loop:
1. Program inputs feed a run.
2. A run publishes program outputs.
3. Outputs sit in the Library, where panels show them.
4. What the user does in panels sets program inputs.

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
| **Program** | A declaration of one experiment: the devices it needs, the program inputs it reads (laid out in input sections), the program outputs it emits, and the runners that can start it. Its `run` does the work. *(Devices from phase 2: today `uses`. Program inputs and input sections from phase 5: today parameter blocks in `params`.)* |
| **Run** | One execution of a program. It gets a frozen copy of the program's inputs, claims its devices, and publishes into its program outputs. |
| **RunContext** | What a running program is handed: its input values, its devices, its program outputs, and cancel. *(From phase 2 its outputs are reached with `output_for`: today `dataset_for`.)* |
| **Runner** | A way to start a program (once, continuously, arm and run). It's declared by the program, and its controls are drawn by whichever panel shows the program. |

### Program inputs: user → program

| Noun | Definition |
|---|---|
| **ProgramInput** | The declaration of one value a program reads, like dwell time or a mask. It has a key, a label, a default and rules for valid values. **The same key means the same value in every program that declares it.** That's how a setting carries across modality changes. *(From phase 5: today one field of a parameter block, keyed by its dotted path such as `ScanGroup.dwell_us`.)* |
| **InputSection** | A labelled list of program inputs. Purely layout: how a panel groups a program's inputs on screen. *(From phase 5: today `Section`, one per parameter block.)* |
| **ProgramInputs** | The collection of current values for every program input, by key. Any panel may read it, set a value (validated by the input's rules) or subscribe. It's saved in the session, and a run's values go into its recording. *(From phase 4: today `BlockStore`, held by the acquisition.)* |
| **InputRequest** | A panel or runner asking for a value of one type that another panel can supply. Whichever supplying panel the user acts in answers it. There's no storage: it's a question with one answer. *(From phase 3: today `RunnerContext.request_pixel`.)* |
| **PixelPick** | The value a click on an image supplies: the program output clicked on, plus `x` and `y` in its pixels. *(From phase 3: today a `Pixel` dataset.)* |
| **Mask** | A drawn region, as a program input value. The Mask Editor supplies it; ROI and Modulation inputs hold it. *(From phase 6: today a `Mask` is data in the Library.)* |

### Program outputs: program → user

| Noun | Definition |
|---|---|
| **Data** | One value a program produced: an image, a spectrum, a mosaic. Immutable. **Only programs make it.** *(From phase 6: today panels also make `Mask` and `Pixel` data.)* |
| **ProgramOutput** | One named output of one run. It holds the latest `Data`, where it came from (`Provenance`), and its writer. Every published value reaches its files; memory holds only the latest. *(From phase 2: today `Dataset`.)* |
| **Library** | The program outputs that are open: acquired now, or loaded from disk. Nothing else. *(From phase 6: today it also holds masks and picked pixels.)* |
| **Recording** | One run's program outputs on disk, opened through its `_meta.json`. |
| **SaveTarget, Writer** | Where a run saves, and how each kind of `Data` is written and read. |

### Views

| Noun | Definition |
|---|---|
| **Panel** | A dock widget: something the user sees and acts in. It reads the collections, calls their operations, and may answer input requests for the types it `supplies`. **The Acquisition panel is one.** *(Answering input requests from phase 3. The Acquisition panel becomes an ordinary panel in phase 4.)* |
| **DataPanel** | A panel that shows one program output of the kind it `renders`, chosen with a source dropdown. |

## The rules

1. **Data flows one way.** Only a run or loading a recording creates a
   `ProgramOutput`. *(From phase 6: today the Mask Editor and pixel picks
   also create them.)*
2. **Panels never call programs, and programs never call panels.** They meet
   only through ProgramInputs, InputRequests and the Library. *(From phase 3:
   today the acquisition routes pixel picks between image panels and
   runners.)*
3. **Every panel is equal.** None is special in code. The app hands each
   panel the collections it needs. *(From phase 4: today the Acquisition
   panel is wired in specially.)*
4. **Working state and screen state are separate.** The session saves them
   separately. *(From phase 4.)*

## The files

| File | Holds |
|---|---|
| `registry.py` | `Registry` |
| `device.py` | `Device` + `device_registry` |
| `program.py` | `Program`, `RunContext` + `program_registry` |
| `runner.py` | `Runner`, `RunnerContext`, `Control`, `Button`, `Toggle` |
| `program_inputs.py` | The value rules (int, float, choice, ...), `Editor`, `FieldContext`. *(Today also parameter blocks: `Group`, `BlockStore`, `BlockMap`, `block_registry`. `ProgramInput` from phase 5, the `ProgramInputs` protocol from phase 4.)* |
| `input_section.py` | `Section` *(`InputSection` from phase 5)* |
| `input_requests.py` | `InputRequest`, `InputRequests` (protocol), `PixelPick` *(from phase 3)* |
| `mask.py` | `Mask`, as a program input value *(from phase 6)* |
| `data.py` | `Data`, `ArrayData`, `Image`, `DecayHistogram`, `RawSamples`, `Spectrum`, `Mosaic`, `DATA_KINDS` *(today also `Mask` and `Pixel`)* |
| `program_output.py` | `ProgramOutput`, `Provenance`, `Origin` *(from phase 2: today `dataset.py` with `Dataset`)* |
| `library.py` | `Library` (protocol) |
| `saving.py` | `SaveTarget`, `Writer` + `writer_registry` |
| `panels.py` | `Panel`, `DataPanel`, `SourcePicker`, `DevicePanel` + their registries. The only file here that uses Qt. |
