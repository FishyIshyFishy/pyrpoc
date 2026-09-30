# How pyrpoc is organised

This page is the map. It covers what each folder is for, who is allowed to use
whom, and where a new feature goes. The rules on this page are checked on every
commit by `uv run lint-imports`, so the map can't drift from the code.

## The map

```
pyrpoc/
  structs/            the shared vocabulary: the nouns every other part talks through
  qt_components/      generic widgets that know no feature: tables, cards, parameter form, icons
  plugins/            things you add more of
    devices/            hardware: DAQ, galvo, Prior stage, time tagger
    programs/           experiments, with their parameter groups and runners
    data_panels/        displays of one open dataset: 2D tiled, overlay, spectrum, mask editor
  device_inventory/   the devices this workbench has, and the Devices panel
  data_library/       the open data, recordings on disk, and the Data Library panel
  acquisition/        choosing a program, setting it up and running it, and the Acquisition panel
  app/                builds all of the above, connects them, and shows the window
    session/            remembers your setup between launches
  main.py             starts the app
```

There are four kinds of folder:

- **The vocabulary: `structs/`.** Every file is one noun, like `Dataset`,
  `Device`, `Program` or `DataPanel`. When two parts of pyrpoc need to talk,
  they talk through a noun defined here. It imports nothing else from pyrpoc.
  Its `__init__.py` lists every file and what it holds.
- **Plugins: `plugins/`.** You add more of these over time: a new instrument, a
  new experiment, a new way to look at data.
  - Each plugin subclasses a base class from `structs/` and registers itself in
    that base class's registry. Nothing keeps a list of plugins; registering is
    how the app finds them.
  - A plugin knows only `structs/` and `qt_components/`. A program may also use
    devices.
- **Subsystems: `device_inventory/`, `data_library/`, `acquisition/`.** These
  are the parts of the app itself. Each one is a folder holding everything
  about one job:
  - its logic, with no Qt, so it can be tested without a screen;
  - a Qt model, which is the commands the screen calls and the signals it
    listens to;
  - its panel.

  A subsystem never names a plugin. It looks plugins up in registries.
- **The app: `app/`.** Builds the three subsystems, connects them, and shows the
  window. It is the only place that knows about everything, so it should stay
  small. If it grows, a subsystem is missing.

## Who may use whom

Higher can use lower, never the other way round:

```
app
acquisition                       (uses the inventory and the library)
data_library    device_inventory  (side by side, independent of each other)
plugins                           (programs may use devices; programs and data panels never meet)
qt_components
structs
```

Plus three rules:

- **No subsystem imports a plugin.** They find plugins through registries.
- **The no-Qt cores stay Qt-free.** That's the executor, device claims,
  recordings, the library store, the saver, the recording format, and the
  session file.
- **Subsystem models never import a panel or widget.** Panels use models, never
  the reverse.

## Where does my feature go?

| You want to add… | It goes in… |
|---|---|
| A new instrument | `plugins/devices/<name>/`: a `Device` subclass registered in `device_registry` |
| A new experiment | `plugins/programs/<name>.py`: a `Program` subclass registered in `program_registry` |
| A new way to start a program (like Continuous) | `plugins/programs/components/runners/` |
| A new way to look at data | `plugins/data_panels/<name>/`: a `DataPanel` subclass registered in `data_panel_registry` |
| A new kind of data (like a 3-D volume) | a class in `structs/data.py`, then a writer for it and a data panel that shows it |
| A new file format | `data_library/writers/`: a `Writer` registered in `writer_registry` |
| A feature of the data library (search, tags, streaming off RAM) | `data_library/` |
| A feature of running (queues, run history, parallel runs) | `acquisition/` |
| A new generic widget | `qt_components/`, if it knows no feature; otherwise next to the panel that uses it |
| **A whole new capability** (see below) | **a new subsystem folder** |

**A whole new capability** is something that isn't a plugin and isn't part of
an existing subsystem's job. Examples: sequencing experiments over hours,
controlling pyrpoc remotely or from an agent, or analysis pipelines. It gets
its own top-level folder, shaped like the other subsystems:

1. **Its logic, with no Qt,** in plain Python files you can test.
2. **A Qt model** (`model.py`): the commands the screen calls, and the signals
   it listens to.
3. **Its panel** (`panel.py`), if it has one.
4. **New shared nouns go in `structs/`,** if other parts need to talk to it.
5. **It is built and connected in `app/application.py`.** Its panel is added in
   `app/window.py`.
6. **Its settings are remembered through `app/session/`,** if it has any.
7. **It gets its place in the layers** in `pyproject.toml`. It usually sits
   above what it uses: a sequencer, for example, would sit above `acquisition`.

## What stays true on disk

Folder names are not saved anywhere:
- **Saved sessions** store registry keys like `"image_2d"`.
- **Recordings** follow [data-format.md](data-format.md).

Moving code around never breaks either. `tests/` loads a committed recording of
every format version on every commit.
