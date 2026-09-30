# pyrpoc dev refresher

Where things live, by idea rather than by file. The long version is `docs/architecture.md`.

## Commands

```
uv run pyrpoc          launch
uv run pytest          tests
uv run lint-imports    folder rules
git commit             pre-commit runs ruff, pyright, lint-imports and pytest
```

## The four kinds of folder

- **`structs/`: the vocabulary.** The types every other part talks through, and the
  registries plugins sign up in. It mirrors the rest of the package: the nouns
  implemented in `pyrpoc/X/` are defined in `structs/X/`. Working on a device?
  Its contract is under `structs/plugins/`. Working on the library? Under
  `structs/data_library/`.
- **`plugins/`: things you add more of.** Hardware (`devices/`), experiments and
  the pieces they are built from (`programs/`), and ways to look at one dataset
  (`data_panels/`). Each registers itself; nothing keeps a list.
- **Subsystems: the app's own jobs.** Each folder holds its plain-Python logic,
  a Qt model (commands in, signals out) and its panel.
  - `device_inventory/`: the devices you've added and their connections.
  - `data_library/`: open data, recordings on disk, file formats, loading, limits.
  - `acquisition/`: choosing, setting up and running programs; runners, series.
- **`app/`: wiring.** Builds the subsystems and connects them, plus the window,
  menus, theme and the session that remembers your setup.

`qt_components/` holds generic widgets that know no feature.

## What's in structs/

A noun lives next to whoever **implements or stores** it, not whoever uses it most.

```
structs/
  registry.py            Registry, the lookup-by-key every plugin kind uses
  plugins/               contracts that plugins implement
    devices.py             Device: what hardware must do; device_registry
    programs/              Program + RunContext (what an experiment gets while running);
                           Runner + Button/Toggle (ways to start one); Pick
    data_panels.py         Panel, DataPanel, SourcePicker; data_panel_registry
    params.py              fields, Group (a parameter block), BlockStore; block_registry
  data_library/          what the data library holds and saves
    data.py                kinds of data: Image2D, Spectrum1D, Mask2D, ...
    dataset.py             Dataset: one output's frames + where they came from
    library.py             Library: read-only view of open datasets
    saving.py              SaveTarget (where to save), Writer (a file format); writer_registry
```

The placements that surprise:

- **Writers are data, not programs.** A program never touches a writer. It
  declares its outputs and the `Data` kind each carries (`Image2D`, ...), and
  publishes arrays. When saving is on, the data library's saver picks a writer
  for each output by its data kind from `writer_registry`. Loading uses the same
  writer to read the files back.
- **Data kinds live in `data_library/` even though programs make them** and data
  panels draw them. The library is what stores and saves them.
- **`SaveTarget` lives in `data_library/`,** though acquisition holds the current one
  and the Acquisition panel edits it.
- **`Pick` lives with programs.** A runner asks for one ("click a pixel") and an
  image panel answers, so programs and data panels meet only through it.
- **`params.py` lives under `plugins/`,** though the parameter form, acquisition
  and the session also read it. Plugins are what declare parameters.
- **`Panel` (the base for every dock) is in `plugins/data_panels.py`,** so the
  built-in Acquisition, Data Library and Devices panels import it from there too.
- **`Library` is how plugins see the library.** Data panels and the mask field get
  read access through it, never the real `LibraryModel`.

## Import order

Higher may import lower, never the reverse:

```
app > acquisition > data_library | device_inventory > plugins > qt_components > structs
```

- Subsystems never import plugins; they look them up in registries.
- Programs may use devices. Programs and data panels never meet; they talk
  through types in `structs/`.
- Subsystem models hold no widgets. Panels use models, not the reverse.

## Where a feature goes

| Adding... | Goes in |
|---|---|
| an instrument, experiment or data view | its folder under `plugins/`, registered in the registry from its `structs/plugins/` contract |
| a way to start a program, or a parameter group | the programs' shared components under `plugins/programs/` |
| a kind of data or a file format | its contract under `structs/data_library/`, the format's writer under `data_library/`, and a data panel to show it |
| a feature of storing or browsing data | `data_library/` |
| a feature of running | `acquisition/` |
| a whole new capability | a new subsystem folder, wired in `app/`, given a layer in `pyproject.toml`, with any shared nouns in `structs/<subsystem>/` |

## Don't break on disk

- Sessions store registry keys, not module paths, so moving code is safe.
- Recordings follow `docs/data-format.md`. Changing the format means a new
  version, a decoder for it, and a fixture under `tests/fixtures/recordings/`.
