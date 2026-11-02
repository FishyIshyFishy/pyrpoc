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
<stem>_meta.json            the entry point; names everything else
<stem>_<channel>.tiff       Image: one multi-page float32 TIFF per channel, a page per publish
<stem>_<output>.npz         DecayHistogram, RawSamples, Spectrum: `data` holds every publish on
                            its leading axis, `channels` the channel labels
<stem>_<output>_tiles.json  Mosaic: the grid, beside one TIFF per channel with a page per image
```

Every value a program publishes reaches the files, so a long continuous run is
all on disk while the data library holds only its latest value. Loading a
recording opens each output at the last value it was published.

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
| `kind` | string | The `Data` class name: `Image`, `DecayHistogram`, `RawSamples`, `Spectrum`, `Mosaic`, ... |
| `published` | int | Values published into the output; the files hold every one. |
| `files` | object: key → file name | Where the values are. The keys are what that kind's writer expects back (channel labels for TIFF, the output name for NPZ, channel labels plus `tiles` for a mosaic). |
| `metadata` | object | What the program recorded with `ctx.describe`. |
| `notes` | string | Free text the user added in the Data Library panel. |

## `Mosaic` files

A mosaic's grid is part of the noun, not a metadata convention. Its writer
puts the whole plan in `<stem>_<output>_tiles.json` before the first image, so
a mosaic cut short still says where its images belong, then appends each image
as a page of its channel's TIFF. Page `i` was taken at entry `i` of the list.

| Field of each entry | Type | Meaning |
|---|---|---|
| `row`, `col` | int | Its grid cell: `row` along stage y, `col` along stage x. |
| `x_um`, `y_um` | number | The stage position it was imaged at. |

Pixel size is not recorded: the Mosaic panel finds the overlap from the images
themselves.

## Compatibility rules

1. **Within a version, fields may be added but never renamed, removed, or
   given a new meaning.** Readers ignore fields they don't know, and editing
   notes keeps them. So a file written by a later 1.x still loads in an
   earlier one.
2. **Anything else is a new version.**
   - Add a decoder for it to `DECODERS`, and keep every older decoder
     forever.
3. **Data file formats are covered by the same promise.**
   - A writer's `read` must keep reading what every earlier version of that
     writer wrote.
   - A new on-disk layout for a kind needs a new writer key, or a new format
     version.
