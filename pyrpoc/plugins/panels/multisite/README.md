# multisite panel

A running loop over the whole dish. The survey's overview is the backdrop, with each unit's latest frame laid where it was taken. Hold **Tab** to bring see-through cards up over the live dish; the first card is the routine queue.

It is not registered as a panel yet. To run it on its own, against a fake loop:

```
uv run python -m pyrpoc.plugins.panels.multisite.demo
```

The demo's **Indicators** menu switches each way of showing what the loop is doing, so they can be compared.

## Keys
All keys are in `keys.py`.

| Where | Key | Does |
|---|---|---|
| anywhere | hold Tab | peek at the overlay until released |
| anywhere | tap Tab | keep the overlay up; Tab or Esc closes it |
| dish | wheel / right or middle drag | zoom round the cursor / pan |
| dish | Shift + drag | draw a unit freehand |
| dish | click / Del | choose a unit / ask to remove it |
| dish | F, double-click | the whole dish |
| queue | ↑ ↓ | choose a routine; its unit is marked on the dish |
| queue | Shift + ↑ ↓, or drag | reorder a routine that is due |
| queue | Enter, double-click | glide to the unit |
| queue | P / R / Del | pause or resume its unit / run it next / remove its unit, after asking |

## How the queue runs
The queue holds routines (one routine on one unit), in the order they will run:
- **Now**: the routine being run.
- **Run now**: routines asked to run next.
- **Due**: due routines, in the person's order where they gave one, then by due time.
- **Upcoming**: routines not yet due, by due time.
- **Paused**: routines whose unit is paused.

Reordering only breaks ties among routines that are due: nothing runs before it is due. `state/run_order.py` holds the rule, and the fake loop runs by the same function. Pause, run-now and remove act on the whole unit, as the loop does; the queue's hint line says so.

A loop takes changes only between steps, so the panel shows each change as on its way until the loop's log shows it taken:
- A queue row says "pausing…" (or the change sent) until then (`state/pending.py`).
- A drawn unit stays dashed, tagged "joining…", until it joins or is refused (`state/drafts.py`).

## Layers
Each layer imports only those below it. `uv run lint-imports` checks this.

| Layer | What it is |
|---|---|
| `demo/` | The fake dish and loop, and the window that runs them |
| `view.py` | `MultisiteView`: the dish with the overlay over it, fed by one `LoopFeed`. Tab is caught here |
| `overlay/`, `canvas/` | The see-through cards (`Card` is the base, `QueueCard` the first), and the dish with its layers. They never import each other |
| `keys.py`, `look.py`, `feed.py` | Every key, every colour, and `LoopFeed`: what the panel reads from a loop and every change it sends |
| `state/` | No Qt: the queue order, reordering, peek, pending changes, drafts, outlines, wording |

To add a card, subclass `Card`, give it an `Anchor`, and add it to the overlay in `view.py`. To add an indicator, add a layer in `canvas/layers/` and a flag in `canvas/indicators.py`.

## Wiring it in
Everything the panel needs comes through `feed.LoopFeed`. Wiring in is:
1. **A feed built on a run's outputs.**
   - `snapshot()`: fold the run's `LoopLogOutput` with `LoopFold`, then `state.routines.routine_statuses` and `state.snapshot.snapshot_of`. Fold again only when the log grows, not on every frame.
   - `backdrop()`: from `SurveyOutput.overview()`.
   - `frames()`: each unit's newest frame in `UnitFramesOutput`.
   - The changes it sends go to the live `UNITS` input:
     - pause: set `Unit.paused`;
     - run now: raise `run_now_asked`;
     - remove: drop the unit;
     - add: append the `Unit` the panel made.
2. **A thin `DataPanel`** that holds a `MultisiteView` over that feed.
3. **Reordering.** The loop has no notion of the person's order yet. It needs a live input of routine keys that `Steering` reads, and `Queue.runnable` sorting due threads by it, as in `state/run_order.py`.

## Still to decide
- Which of the four indicators to keep.
- Whether pause, run-now and remove should act per routine (a loop change).
- How groups are given.
- The selected-routine, recent-events and loop-summary cards.

A unit's largest size (`state/outline.py`) is a stand-in for the loop's own rule, which allows for a tracking margin.
