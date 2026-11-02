# Future: plotting the autofocus curve

Autofocus currently publishes only its region frames. Each step's z and
Tenengrad score go to the status line and into the `autofocus` metadata of the
`intensity` output (`steps: [{z_um, metric, step_um}, ...]`). This page
describes how to also show the score against z live, as a curve. It is a small
change that touches no subsystem.

## Why it is easy

- **The curve is a noun.** A dataset holds one value, so the program publishes
  the whole curve so far after each step, the way the mosaic programs publish
  the mosaic so far. A panel then only ever draws its latest value.
- **The program already makes one call per step.** `FocusRun.report` in
  `plugins/programs/autofocus/program.py` is where the extra `ctx.publish` goes.

## Steps

1. **A data kind** in `structs/data_library/data.py`, e.g. `Curve`:
   - An `ArrayData` with `axes = ("point", "xy")`: one `(x, y)` row per visit,
     in visit order.
   - Name the kind for what it is, not for autofocus, so other scans (a power
     sweep, a delay scan) can reuse it.
   - Add it to `DATA_KINDS`.
   - Do not reuse `Spectrum`: pickers filter on kind, so a spectrum panel
     would then offer focus curves.
2. **A writer.** Its saved file only needs the last value, since each publish
   is the whole curve: a small writer that rewrites one `.npy` on each write,
   or register the kind on `NpzWriter` and accept that it keeps every publish.
3. **A panel** in `plugins/data_panels/curve/`, a `DataPanel[Curve]` with
   `renders = Curve`. `show_value` plots the value's rows.
   - The hill climb visits z out of order, overshooting and coming back. Draw
     the points as a scatter, optionally joined in visit order so the path of
     the climb shows. Do not join them sorted by z.
   - Mark the best point.
4. **The program.**
   - Add `"focus_curve": Curve1D` to `Autofocus.emits`.
   - In `FocusRun.report`, append `(step.z_um, step.metric)` to the points so
     far and call `ctx.publish("focus_curve", Curve(np.array(points)))`. The
     metric is already summed across the detectors, so there are no channels.
5. **The format.** Add the kind to the layout list in
   [data-format.md](data-format.md). A new kind does not break old recordings.

## What to decide then

- Whether the curve shows each channel's own Tenengrad (`C` points per step)
  or only the summed score that drives the climb. Per channel is free to
  compute and shows whether the detectors agree on focus.
