# Future: plotting the autofocus curve

Autofocus currently publishes only its region frames. Each step's z and
Tenengrad score go to the status line and into the `autofocus` metadata of the
`intensity` output (`steps: [{z_um, metric, step_um}, ...]`). This page
describes how to also show the score against z live, as a curve. It is a small
change that touches no subsystem.

## Why it is easy

- **One point per frame keeps the shape fixed.** `Dataset.append` stores frames
  as a list, and `NpzWriter` saves them with `np.stack`, so every frame must
  have the same shape. If the curve grew by a point each frame, the shape would
  change. So the program publishes one `(z, metric)` point per step. The curve
  is all of the dataset's frames, not its latest one.
- **A panel can read every frame.** `Dataset.frames()` returns all of them, so a
  data panel's `show_frame` can redraw the full curve on each new point.
- **The program already makes one call per step.** `FocusRun.report` in
  `plugins/programs/autofocus/program.py` is where the extra `ctx.publish` goes.

## Steps

1. **A data kind** in `structs/data_library/data.py`, e.g. `Curve1D`:
   - It is `(C, 2)` float32, one `(x, y)` point per channel.
   - Name the kind for its shape, not for autofocus, so other scans (a power
     sweep, a delay scan) can reuse it.
   - Add it to `DATA_KINDS`.
   - Do not reuse `Spectrum1D`: its axis is a bin index, and pickers filter on
     kind. A spectrum panel would then offer focus curves.
2. **A writer.** Register the kind on `NpzWriter`
   (`data_library/writers/npz.py`) with one more `@writer_registry.register`
   line. The saved `data` is then `(N, C, 2)`.
3. **A panel** in `plugins/data_panels/curve/`, with `renders = [Curve1D]`.
   `show_frame` stacks `dataset.frames()` and plots `x` against `y` for each
   channel.
   - The hill climb visits z out of order, overshooting and coming back. Draw
     the points as a scatter, optionally joined in visit order so the path of
     the climb shows. Do not join them sorted by z.
   - Mark the best point.
4. **The program.**
   - Add `"focus_curve": Curve1D` to `Autofocus.emits`.
   - In `FocusRun.report`, call
     `ctx.publish("focus_curve", np.array([[step.z_um, step.metric]]), channels=["tenengrad"])`.
     It is one channel because the metric is already summed across the
     detectors.
5. **The format.** Add the kind to the layout list in
   [data-format.md](data-format.md). A new kind does not break old recordings.
   The NPZ path is already covered by the committed `spectrum` fixture in
   `tests/fixtures/recordings/v1/`, so a curve fixture is optional.

## What to decide then

- Whether the curve shows each channel's own Tenengrad (`C` points per step)
  or only the summed score that drives the climb. Per channel is free to
  compute and shows whether the detectors agree on focus.
