# pyrpoc dev refresher

The full explanation is `docs/architecture.md`. This page is the quick lookup.

## Commands

```
uv run pyrpoc          launch
uv run pytest          tests (tests/)
uv run lint-imports    folder rules
git commit             runs everything via pre-commit (ruff, pyright, lint-imports, pytest)
```

## Where things are

```
main.py               starts the app
structs/              the shared nouns, one per file
  registry.py           Registry: how plugins are found by key
  data.py               kinds of data (Image2D, Spectrum1D, Mask2D, ...)
  dataset.py            Dataset, Provenance, Origin
  library.py            Library: read access to open datasets
  saving.py             SaveTarget, Writer, writer_registry
  params.py             parameter fields, Group, BlockStore, block_registry
  device.py             Device, device_registry
  program.py            Program, RunContext, program_registry
  runner.py             Runner, RunnerContext, Button/Toggle controls
  picks.py              Pick, PixelPick (runner asks, data panel answers)
  panel.py              Panel, DataPanel, SourcePicker, data_panel_registry (the only Qt file)
qt_components/        generic widgets: table, cards, param_form, icons, colors, levels
plugins/
  devices/              daq, galvo, prior_stage, time_tagger
  programs/             confocal, flim, split_confocal, pinpoint_raman, simulation
    components/           param_groups/, runners/ (single, arm_and_run, continuous), editors
  data_panels/          image_2d, overlay, spectrum, mask_editor
device_inventory/     inventory.py (added devices, sessions), panel.py (Devices panel)
data_library/
  store.py              LibraryStore: open datasets, 1 GiB limit, auto-purge
  model.py              LibraryModel: Qt signals, load, notes, purge
  format.py             recording format (_meta.json, DECODERS by version)
  saving.py             RecordingSaver
  writers/              tiff, npz
  panel.py, details.py  Data Library panel, Details dialog
acquisition/
  model.py              Acquisition: selected program, params, save target, blockers
  host.py               RunnerHost, RunnerSession: runs a program's runners
  events.py             RunEvents: run started/finished, series
  executor.py           runs a program on a worker thread
  claims.py             which devices a program needs
  recording.py          Recording, RecordingKey, Series
  panel.py              Acquisition panel
app/
  application.py        builds and connects the three subsystems
  window.py, menubar.py, theme/
  session/              file.py, restore.py, autosave.py (session.json)
assets/               svg icons, sdk dlls
```

## Import order

Higher may import lower, never the reverse:

```
app > acquisition > data_library | device_inventory > plugins > qt_components > structs
```

- Subsystems never import plugins; they look them up in registries.
- `programs/` may import `devices/`; `programs/` and `data_panels/` never meet.
- Subsystem models (`model.py`, `host.py`, `events.py`, `inventory.py`) hold no widgets.

## Adding things

| New... | Goes in |
|---|---|
| instrument | `plugins/devices/<name>/`, register in `device_registry` |
| experiment | `plugins/programs/<name>.py`, register in `program_registry` |
| runner | `plugins/programs/components/runners/` |
| data view | `plugins/data_panels/<name>/`, register in `data_panel_registry` |
| data kind | `structs/data.py`, plus a writer and a data panel |
| file format | `data_library/writers/`, register in `writer_registry` |
| whole new capability | new top-level subsystem folder, wired in `app/application.py`, layered in `pyproject.toml` |

## Don't break on disk

- Sessions store registry keys, not paths.
- Recordings follow `docs/data-format.md`. A format change means a new version plus a decoder, and a fixture in `tests/fixtures/recordings/`.
