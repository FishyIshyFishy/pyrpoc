# Multisite Loop panel

Shows a survey's dish and the newest loop run over it. The source picker chooses a **survey**, not a loop, so units can be drawn before any loop has run: a fresh loop starts only on units drawn.

## What it shows
- **The dish**: the survey's overview, with each unit's latest frame laid where it was taken. Each unit is outlined where it was last tracked, and drawn by where it stands. A dashed box one field across marks where the loop is working, and turns solid while it scans.
- **The status line**: what the loop is doing now and what it does next.
- **The overlay**: Tab, or the Queue button, holds two see-through glass cards up over the live dish, and Tab again puts them away. So do Esc and a click on the dish between the cards: the overlay is a glance at the loop, not a pane that stays. The cards fade in and out, and a faint veil over the dish says it is not taking input meanwhile.
  - **Queue** (left): what the loop runs, top first. Row *now* is the routine running. Then come any routines asked to run next, then the rest in the order they fall due. A paused unit's routines are not queued. This is the loop's own order (`LoopView.queue`), not one the panel works out.
  - **Units** (right): every unit, where it stands, and what it does next, with that routine's place in the queue.

Choosing a unit in either card or on the dish chooses it in all three. Double-clicking a row brings its unit to the dish between the cards.

## Steering the loop
Right-click a unit in either card or on the dish:
- **Run next**: put it at the front of the queue.
- **Pause** / **Resume**: take it out of the queue, or put it back.
- **Remove from loop...**: stop it for good, after asking.

Each acts on the whole unit, every routine of it, because the loop does. Every change goes through the live `UNITS` input, which the loop reads between its steps. Until the loop logs a change taken, the unit's state reads *Pausing…*, *Removing…* and so on (`state/pending.py`).

**Draw unit** (or right-click the empty dish) draws one unit: drag round a region. The dotted square is one field, and a shape larger than it is not added. A drawn unit is *Drawn* until the loop joins it, or *Refused* with the loop's reason.

On the dish the wheel zooms, a left or middle drag pans, and a double-click on the empty dish shows all of it.

## How it reads the loop
`feed.LoopFeed` finds the newest recording that read the chosen survey and holds a `LoopLogOutput`. On a clock it folds the log's new events with `LoopFold`, takes new frames from the same recording's `UnitFramesOutput`, and learns the instrument's clock from when events arrive (`state/clock.py`), so countdowns run right even on a simulated dish that runs faster than wall time. It finds every output by its type, never by a program's name for it.

## Layers
Each layer imports only those below it; `uv run lint-imports` checks this.

| Layer | What it is |
|---|---|
| `panel.py` | `MultisitePanel`: the buttons, the status line, the dish and the overlay, wired to one `LoopFeed` |
| `overlay.py` | `Overlay`: the cards held up over the dish, with the fade, the veil and closing on a click between them |
| `dish.py`, `cards.py`, `unit_menu.py` | The dish view; the queue and units glass cards; the menu a right-click on a unit opens |
| `backdrop.py`, `outlines.py`, `lasso.py` | What the dish draws: the overview and frames, the units and the working field, a unit being drawn |
| `feed.py`, `look.py`, `render.py` | The loop as the panel reads it and steers it; the colours of the dish and the glass; channels to colour |
| `state/` | No Qt: the queue and units rows, their wording, pending changes, the clock, outlines |

## Not done yet
- A loop carried on is given no units at first. Pausing or running one next gives it back, but removing a unit the loop was never given does nothing until then.
- Groups are not drawn: every unit drawn takes the plain group.
- Measurements and frames of a chosen unit over time are not shown.
