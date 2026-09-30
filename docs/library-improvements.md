# Data library improvements: phased plan

Status as of 2026-09-30, on `feat/library-improvements`. This is a plan, not
yet built. Each phase is its own commit (or a few), and the app works at the
end of every phase.

## Scope

1. **Continuous series.** Consecutive continuous runs append to one library
   entry and one set of files. The design is in
   [continuous-series.md](continuous-series.md); phase 3 below lists what
   changes relative to it.
2. **Size safety.** A fixed library limit, an auto-purge checkbox, and a
   blocked start when the library is over the limit.
3. **Loading saved recordings.** Open a recording from disk into the library.
   Its metadata comes with it, because the file layout is fixed and enforced.
4. **Metadata and notes.** Right-click a library row to see a dataset's
   metadata and edit its notes.

Out of scope: streaming data out of RAM, saving an unsaved library entry after
the fact, and the wider `structs/` reorganization.

Decisions already made:

- **A continuous series keeps every frame.** Auto-purge never touches a live
  series, so a long enough series can push the library past the limit.
- **Changing parameters mid-series starts a new recording.**
- **Over the limit with auto-purge off, starting is blocked** until the library
  is cleared or auto-purge is turned on.
- **Notes are written into the recording's `_meta.json`.** Data files (TIFF,
  NPZ) are never modified after acquisition.
- **This version starts backwards compatibility.** Every recording saved from
  now on must load in every later version. Earlier recordings don't have to.

## Where the library code lives (MVC)

In Qt, a widget's signal handlers are the controller, so a separate controller
class is rarely worth having. The line that matters is between the view (it
draws and forwards the user's intent) and the model (it holds the state and
carries out commands). Today the library is split awkwardly between the two:
`DataLibrary` (runtime) holds the entries, `Runs.release` closes them, and the
panel works out totals itself.

The split this plan uses:

| Layer | Where | Holds |
|---|---|---|
| Model: storage and policy | `app/runtime/library.py` (Qt-free) | `DataLibrary`: the entries, their total size, the limit, the auto-purge flag, and which entries can be purged. |
| Model: the file format | `app/runtime/recording_format.py` (new, Qt-free) | Reading and writing `_meta.json`, the version dispatch, and updating notes. |
| Model: commands for the UI | `app/model/library.py` (new, Qt) | `Library(QObject)`: `close`, `load`, `set_notes` and `set_auto_purge`, with signals for the views. It absorbs `Runs.release` and runs the purge on the GUI thread. |
| View and controller | `panels/data_library/` | The table, the context menu, the details dialog and the checkbox. Every action calls a `Library` command. It does no bookkeeping of its own. |

`Application` owns one `Library`, the same way it owns `Runs` and `Runners`.
Panels already reach `Application` through the one permitted import.

## Phase 1: give the library a home

*No behaviour change, so every later phase has somewhere to put its code.*

- Add `app/model/library.py` with `Library(QObject)`, which wraps `DataLibrary`.
  - Move `Runs.release` (unsubscribe, then remove) into it as `close`.
  - Move the total-size sum from the panel into `DataLibrary.nbytes`.
- Add `Dataset.origin` (`acquired`, `loaded` or `authored`) and
  `Dataset.finished` (set in `Dataset.finalize`).
  - Authored masks get `authored`, instead of relying on `run_id == 0` to mean
    that.
  - `finished` is how the library tells a live entry from one it may purge.
- The data library panel talks only to `app.library`.

## Phase 2: recording format v1

*Fix what a recording on disk is before anything else writes one, so the
compatibility promise starts from a clean definition.*

- Rewrite `RunSaver.write_metadata` to the v1 schema. `<stem>_meta.json` is
  the only entry point, and the data files sit beside it:

  ```json
  {
    "format": "pyrpoc-recording",
    "format_version": 1,
    "pyrpoc_version": "3.2.0",
    "program_key": "confocal",
    "name": "sample_a",
    "started_at": "…",
    "ended_at": "…",
    "last_error": null,
    "runs": [{"run_id": 12, "started_at": "…", "ended_at": "…", "error": null}],
    "parameters": {},
    "devices": {},
    "outputs": {
      "intensity": {
        "kind": "Image2D",
        "channels": ["ai0", "ai1"],
        "frames": 10,
        "files": {"ai0": "sample_a_ai0.tiff", "ai1": "sample_a_ai1.tiff"},
        "metadata": {},
        "notes": ""
      }
    }
  }
  ```

  - **File paths are relative to the metadata file**, so a copied or moved
    folder still loads.
  - **Everything needed to rebuild a `Dataset` is here**: kind, channels,
    provenance and files.
- **Numbered stems.** The saver takes the first of `name`, `name_2`, `name_3`,
  and so on, whose `_meta.json` doesn't exist. Nothing is overwritten.
- **Readers sit next to writers.** `Writer` gains a classmethod
  `read(files: dict[str, Path]) -> list[np.ndarray]` that returns frames in
  publish order:
  - `TiffWriter.read` stacks the per-channel TIFF pages into `(C, H, W)`
    frames.
  - `NpzWriter.read` splits the `data` array along its leading axis.

  The format knowledge stays in `programs/components/writers/`, and
  `writer_registry` already maps each kind to its writer.
- Add `app/runtime/recording_format.py`:
  - `read_recording(meta_path) -> list[Dataset]`, which dispatches on
    `format_version` through a table of loaders (only `1` for now). New
    versions add a loader and never remove one.
  - `write_notes(meta_path, output, notes)`, which rewrites the file
    atomically (write a temporary file, then replace).
  - Validate at this boundary: an unknown format, a missing file or a kind
    with no reader raises a `RecordingError` that names the file.
- **Compatibility tests.** Add `pytest` to the dev group and a pre-commit
  hook, which you'll need to approve since it changes the config. Add
  `tests/fixtures/recordings/v1/`: small recordings from `simulation` (TIFF)
  and `pinpoint_raman` (NPZ), committed once and never regenerated. One test
  loads every fixture of every version. Another round-trips save → load.
- Add `docs/data-format.md`: the v1 schema and the policy.
  - Within v1, fields may be added and readers ignore unknown ones. Fields are
    never renamed or removed.
  - Anything else becomes v2, with the v1 loader and fixtures kept forever.

## Phase 3: continuous series

As in [continuous-series.md](continuous-series.md) (`Recording`, `Series`,
`RunnerContext.open_series` / `close_series`, and finalizing when the
recording closes), with these changes relative to that doc:

- The metadata work (numbered stems, the `runs` list) is already done in
  phase 2. This phase just appends to `runs` and sets `frames` and
  `ended_at`.
- `Recording.finalize` sets `Dataset.finished`, which phase 4 relies on.
- The saver writes each output's `notes` from its `Dataset` when it rewrites
  metadata, so notes typed during a live recording survive finalize (phase 6).

## Phase 4: size limit and auto-purge

- `LIBRARY_LIMIT_BYTES = 1 << 30` (1 GiB) in `app/runtime/library.py`.
- **Auto-purge.** When it's on and the total goes over the limit, `Library`
  closes the oldest purgeable entries, one at a time, until the total is back
  under.
  - *Purgeable* means `finished` and not `authored`. A live series is never
    touched, and neither are masks the user drew.
  - If only live entries remain, purging stops. The size warning (below)
    stays visible.
- **Threading.** Frames arrive on run worker threads, and `dataset_changed`
  already carries them to the GUI thread. The purge runs there, coalesced
  with a zero-delay `QTimer` so a burst of frames triggers one check.
  - It can't block acquisition, because runs never wait on the GUI thread.
  - Closing an entry only drops references.
  - It must run on the GUI thread anyway, because closing notifies the panels
    and source pickers.
- **Blocked start.** `Executor.start` raises a new `LibraryFull` when the
  total is over the limit and auto-purge is off. `Runs.start` catches it
  alongside `DeviceBusy`, and the acquisition panel's existing
  `start_refused` dialog shows it. The message names the size and the limit,
  and says to close entries or turn on auto-purge in the Data Library panel.
  - Only new recordings are checked. A run continuing an open series is not,
    so a series doesn't fail mid-loop (the live entry is left alone, as
    decided).
  - Loading (phase 5) is blocked the same way.
- **Panel.** Add an "Auto-purge oldest" checkbox beside the total, which now
  reads "X of 1 GiB" and turns the warning colour when over the limit. Save
  the flag in the workspace: add `library: LibraryState(auto_purge=False)` to
  `WorkspaceState`.
  - Check whether the loader accepts a missing field with a default before
    bumping `SCHEMA_VERSION`. A bump resets everyone's saved workspace.

## Phase 5: loading saved recordings

- An "Open…" button in the Data Library panel (and File → Open recording…)
  opens a file dialog filtered to `*_meta.json`.
- `Library.load(path)`:
  1. Refuses if over the limit and auto-purge is off, using the same message
     as the blocked start.
  2. Calls `read_recording`.
  3. Adds each output as a `Dataset`, with `origin = loaded`, the provenance
     rebuilt from the metadata, `writer = None`, `finished = True`, and the
     path to its meta file remembered.
- Loaded entries behave like acquired ones in every panel: displays, picks,
  and mask sources.
  - An output whose kind has no panel still shows up in the library.
- Frames load eagerly into RAM. Streaming off RAM is a later project.
- Loading the same recording twice gives two entries. Allowed, and not worth
  guarding against.

## Phase 6: metadata and notes

- Right-click a library row for a menu:
  - **Details…**
  - **Show in folder**, only if the entry is on disk.
  - **Close**, the existing action, moved from the button.
- The **Details dialog** (in `panels/data_library/`) shows:
  - Read-only: name, program, started and ended times, kind, frames, shape,
    size, files, and the runs list.
  - Parameters and devices as a read-only tree, built from the encoded dicts.
  - Notes, editable.
- **Saving notes** calls `Library.set_notes(dataset, text)`, which sets
  `Dataset.notes`. If the entry is on disk and finished, it also calls
  `write_notes`. A live recording picks the notes up when it next rewrites
  its metadata (phase 3).
  - If the write fails (read-only folder, file moved), the dialog says so and
    keeps the text.
- Only notes are editable. Renaming would break the link between the display
  name and the file stem; revisit it if needed.

## Verification

Each phase runs pre-commit on commit. In addition:

1. **Phase 1:** closing an entry from the panel still frees it, and the total
   still updates live.
2. **Phase 2:**
   - The fixture and round-trip tests pass.
   - A saved `simulation` run's `_meta.json` matches the schema, and repeated
     runs with the same name produce `name`, `name_2`.
3. **Phase 3:** the headless checks in `continuous-series.md` (one entry after
   5 loops, `runs` has 5 items, a parameter change starts `name_2`).
4. **Phase 4:** a headless script with a lowered limit (monkeypatched) checks
   that:
   - FIFO purge keeps the total under the limit and skips the live series and
     masks.
   - With auto-purge off, the next Single start is refused with the message,
     and turning it on lets the start through.
5. **Phase 5:** load each fixture in the running app. It displays, and it can
   be picked as a mask source. A missing TIFF gives a clear error.
6. **Phase 6:** edit notes on a saved entry, then reload the recording
   (phase 5) and the notes are there. Notes typed during a live series
   survive its finalize.
