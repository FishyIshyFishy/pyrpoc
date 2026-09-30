# Continuous series: one library entry per recording

Status as of 2026-09-30. This is a plan, not yet built. It builds on the
`Continuous` runner and the display fix that keeps panels from resetting
between runs.

## The problem

Each run opens its own `Dataset` for every output, along with its own
`RunSaver` and writers, in `Executor.start`. `Executor.finalize` closes them
all when the run ends. `Continuous` just calls `ctx.execute()` again, so a
continuous loop adds a new library entry, a new set of TIFFs and a new metadata
file on every pass. After a few minutes the data library is full of near
duplicates.

## Goal

Consecutive runs of a continuous series append to the same library entries and
the same files. **Runners never see datasets or files**: they only say that a
series has started or ended. Programs don't change.

Decisions already made:

- **A series keeps every frame.** Guardrails on library memory come in a
  separate PR. Streaming data out of RAM is out of scope.
- **Parameters stay editable mid-series.** If the parameters, a device's
  config, or the save target change, the next run starts a new recording
  instead of mixing metadata.
- **Nothing is overwritten.** A recording whose save name already has files
  gets a numbered stem: `name`, then `name_2`, `name_3`, and so on. This also
  fixes an existing bug: two Single runs with the same name currently
  overwrite each other.

## Concepts

| Piece | Where | What it is |
|---|---|---|
| Run | `app/runtime/executor.py` | One execution of `program.run` on a worker thread. It keeps its own cancel event and device leases (unchanged). |
| Recording | `app/runtime/executor.py` (new) | The datasets and optional `RunSaver` that one or more runs write into. It stores the key it was opened for: program key, encoded parameters, device state and save target. It is finalized once, when it closes. |
| Series | `app/runtime/executor.py` (new, Qt-free) | Holds the current `Recording` (or none) and whether the series is closed. A run started in a series reuses the series' recording if its key matches. Otherwise it closes that recording and opens a new one. |

A Single run is a run with no series. Its recording closes when the run ends,
which is today's behaviour: the same files and the same entries.

## Changes

### `structs/runner.py`: `RunnerContext`

Two new abstract methods, described without referring to how the host does it:

- `open_series()`: until `close_series()` is called, each run this context
  starts continues the previous run's library entries and files, as long as
  nothing that run depends on has changed.
- `close_series()`: ends the series. The entries stay in the library, and the
  files are finalized once the last run ends.

No dataset or recording types leak into `structs`.

### `programs/components/runners/continuous.py`

- When the toggle turns on, call `ctx.open_series()`, then `start()`.
- Every way the toggle turns off (a refused start, a stopped run, a failed
  run) goes through one `finish()` helper, which does `toggle.set(False)` and
  `ctx.close_series()`.

`Single` and `ArmAndRun` are unchanged.

### `app/runtime/executor.py`

- `Recording` holds:
  - `key`, `datasets` and `saver`;
  - `runs`, the ids of the runs that joined it;
  - `error`, the last error, if any;
  - `active`, the number of runs still in flight.

  Its `finalize()` is today's closer loop (datasets first, then the saver). It
  reports failures through the last run's `on_failed`.
- `Series` holds `recording` and `closed`. `close()` runs under
  `Executor._lock` and marks the series closed. It finalizes the recording
  right away if `active == 0`. Otherwise the last run to end does it.
- `Executor.start(..., series: Series | None)`:
  1. Builds the key from the existing `blocks.to_dict`, `device_state` and the
     `SaveTarget` fields.
  2. If the series has a recording with the same key, reuses it.
  3. Otherwise finalizes the old recording (if idle) and opens a new one with
     the existing `build_saver` and `open_datasets`.

  `Run` gains `recording` and `series` attributes. Provenance stays per run,
  and a dataset keeps its first run's provenance.
- `Executor.finalize(run, error)` splits into two parts:
  - It always releases the leases and sets `run.running = False`.
  - It records the error on the recording and decrements `active`. It
    finalizes the recording only if the run had no series, or its series is
    closed or has moved on to another recording.
- `open_datasets` and `on_dataset` fire only for new recordings, so the library
  gets one entry per output per recording.

### `app/runtime/saving.py`: `RunSaver`

- When it is created, it picks the first free stem out of `name`, `name_2`,
  `name_3`, and so on, where "free" means no `<stem>_meta.json` exists yet.
  Every recording writes its metadata in `prepare`, so that file reliably marks
  a stem as taken. Writers already build their paths from `root`, so they
  follow automatically.
- Metadata gains `"runs": [{run_id, started_at}]`. Each run is appended as it
  joins, and the list is rewritten in `finalize`. The top-level `run_id` and
  `started` stay as the first run's, for compatibility.

### Writers (`programs/components/writers/tiff.py`, `npz.py`)

No change to the writer interface:

- `TiffWriter` already appends pages. Its `unlink` runs only on its first
  write (it is guarded by `if not self.paths`).
- `NpzWriter` already buffers until `finalize`, which now happens when the
  recording ends.

### `app/runtime/runs.py`

`Runs.start(..., series)` passes the series through to `Executor.start`. No new
signals.

### `app/model/runners.py`: `Slot`

- Gains `series: Series | None`.
  - `open_series()` creates a series; a closed slot refuses.
  - `close_series()` closes the series and drops it.
  - `execute()` passes `self.series` to `host.start`.
- `Runners.detach()` closes the slot's series after stopping its runs, so
  switching program mid-loop still finalizes the files.

## Not changing

- **Panels.** Displays already hold their widgets across runs. With one entry
  per recording, the source doesn't even switch.
- **The data library panel.** Its size cell already updates on
  `dataset_changed`.
- **Programs and `RunContext`.**

## Verification

1. Run `uv run pre-commit run --all-files --config .github/pre-commit.yaml`.
2. Write a headless script that builds `MainWindow`, selects `simulation`, and
   drives the `Continuous` toggle with `QTimer`, with saving enabled into a
   temp dir. Check that:
   - After 5 loops, `len(app.library) == 1`, and `len(dataset) == 5 * num_frames`.
     Each `<name>_<channel>.tiff` has that many pages, and `<name>_meta.json`
     lists 5 runs.
   - Changing `FrameGroup.num_frames` through `app.blocks` mid-series produces
     a second entry and `<name>_2_*` files, and leaves the first files
     untouched.
   - Stopping mid-run turns the toggle off, finalizes the metadata and releases
     the leases (a Single run starts right after).
   - Two Single runs with the same name give two entries, with `name_*` and
     `name_2_*` files.
3. Check that `pinpoint_raman` with `ArmAndRun` (no series) still gives one
   entry and one npz per run.
4. Manually: run the app with continuous simulation and the 2D Tiled panel open.
   The data library should show a single growing row.
