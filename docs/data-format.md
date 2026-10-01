# Recording format

A **recording** is what one acquisition leaves on disk. Format 1 is the first
format with a compatibility promise: every recording saved in format 1 or later
must load in every later version of pyrpoc. Recordings saved before format 1
are not supported.

The code is in `pyrpoc/data_library/format.py`, and the readers and writers
for data files are in `pyrpoc/data_library/writers/`.

## Layout

A recording is one metadata file plus data files, all in the same folder:

```
<stem>_meta.json        the entry point; names everything else
<stem>_<channel>.tiff   Image2D: one multi-page float32 TIFF per channel, a page per frame
<stem>_<output>.npz     Cube3D, Samples4D, Spectrum1D: `data` holds every frame on its leading axis
```

- **The stem.** `<stem>` is the save name. If a recording with that stem
  already exists, the new one takes the first free `<stem>_2`, `<stem>_3`, and
  so on. Nothing is ever overwritten.
- **To open a recording, open its `_meta.json`.** Data file names in the
  metadata are relative to the metadata file's folder, so a recording can be
  moved or copied as a whole folder.
- **What gets rewritten.** Data files are never modified once a recording
  ends. The metadata file is rewritten while a recording is going, and after
  it ends only to change an output's `notes`.

## `_meta.json`, format 1

| Field | Type | Meaning |
|---|---|---|
| `format` | `"pyrpoc-recording"` | Identifies the file. |
| `format_version` | `1` | Picks the decoder. |
| `pyrpoc_version` | string | The version that wrote the recording. |
| `program_key` | string | The program that produced it, which need not still exist. |
| `name` | string | What the recording is called, which is the save name. |
| `started_at`, `ended_at` | ISO 8601 UTC string; `ended_at` is `null` while recording | When it started and ended. |
| `last_error` | string or `null` | Why the recording ended badly, if it did. |
| `runs` | list of `{run_id, started_at, ended_at, error}` | The runs that wrote into it. A continuous series has several. |
| `parameters` | object | The program's parameter blocks, encoded as the session file encodes them. |
| `devices` | object | Each device's configuration, keyed by device class. |
| `outputs` | object: output name → output | One entry per output the program emits. |

Each output has these fields:

| Field | Type | Meaning |
|---|---|---|
| `kind` | string | The `Data` kind: `Image2D`, `Cube3D`, `Samples4D`, `Spectrum1D` or `Mask2D`. |
| `channels` | list of strings | Channel labels, in axis order. |
| `frames` | int | Frames written. |
| `files` | object: key → file name | Where the frames are. The keys are what that kind's reader expects (channel labels for TIFF, the output name for NPZ). |
| `metadata` | object | What the program recorded with `ctx.describe`. |
| `notes` | string | Free text the user added in the Data Library panel. |

## Conventions inside an output's `metadata`

Some programs write structured metadata that a display relies on. These keys
fall under the same promise as the fields above.

### `mosaic`: a tiled acquisition

Written by the Mosaic and Mosaic Simulation programs on their `Image2D` output.
Frame `i` of the output (page `i` of each channel's TIFF) is the tile whose
`index` is `i`. The frames are ordinary images, so any viewer can open them;
this key is what lets the Mosaic panel place them. It is written by
`MosaicGroup.layout_metadata` (`plugins/programs/building_blocks/parameter_groups/mosaic.py`)
and read by `plugins/data_panels/mosaic/layout.py`. The program and the panel
share nothing but this table.

| Field | Type | Meaning |
|---|---|---|
| `path` | string | How the stage walked the grid. `"snake"`: row 0 left to right, row 1 right to left, and so on. |
| `x_tiles`, `y_tiles` | int | Tiles along the stage's x and y. |
| `spacing_um` | number | Stage step between neighbouring tiles, the same in x and y. |
| `tiles` | list of `{index, row, col, x_um, y_um}` | Every tile in acquisition order: its grid cell (`row` along stage y, `col` along stage x) and the stage position it was imaged at. |

Tiles carry their own `row` and `col`, so a reader never re-derives the path
from its name. Pixel size is not recorded: the Mosaic panel finds the overlap
from the images themselves.

## Compatibility rules

1. **Within a version, fields may be added but never renamed, removed, or
   given a new meaning.** Readers ignore fields they don't know, and editing
   notes keeps them. So a file written by a later 1.x still loads in an
   earlier one.
2. **Anything else is a new version.**
   - Add a decoder for it to `DECODERS`, and keep every older decoder
     forever.
   - Add a folder of fixtures, `tests/fixtures/recordings/v<N>/`, written by
     the release that introduces format N.
3. **Fixtures are never regenerated or edited.** `tests/test_recording_format.py`
   loads every fixture of every version and checks its contents. If a change
   breaks one, the change is wrong, not the fixture.
4. **Data file formats are covered by the same promise.**
   - A writer's `read` must keep reading what every earlier version of that
     writer wrote.
   - A new on-disk layout for a kind needs a new writer key, or a new format
     version.
