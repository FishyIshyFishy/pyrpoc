# Runs and runners: where they stand, and how they grow

Status as of 2026-09-29; paths updated for the reorganization in [architecture.md](architecture.md). This is a plan, not a spec: each section below says
what to build when a feature first needs it, and nothing here should be built
before then.

## Goals this has to reach

- **Several programs per session.** A side program runs while an
  acquisition runs, and sequences run one program after another.
- **Talking to a running program.** Feedback reaches a program while it runs,
  such as a new target, a retune, or a stop condition.
- **Long autonomous loops.** Experiments run for hours or days: acquire,
  analyse, decide, act, and repeat, without someone at the keyboard.

## What exists now

| Piece | File | What it is |
|---|---|---|
| `Run` | `acquisition/executor.py` | One execution of one program. It is the handle that everything outside the program uses to reach it. It holds its datasets, devices, cancel event and `running` flag. `stop()` asks the program to stop. |
| `Executor` | `acquisition/executor.py` | Starts runs, each on its own worker thread, and has no Qt. It opens datasets, sets up saving, and hands each run a **copy** of its params. Runs may overlap as long as they don't share a device. |
| `Leases` / `DeviceBusy` | `acquisition/claims.py` | Record which run holds each device. A start that needs a held device is refused before anything opens. |
| `RunEvents` | `acquisition/events.py` | The only Qt adapter for runs. It starts them, re-emits run events on the GUI thread with each signal carrying its `Run`, and subscribes to datasets. It knows nothing of the selected program. |
| `RunnerHost` | `acquisition/host.py` | Attaches the selected program's runners, holds their controls, and routes picks through `pick_mode_changed`. |
| `RunnerSession` | `acquisition/host.py` | The `RunnerContext` that one program's runners were attached with. It also tracks the runs those runners started. |
| `Runner` / `RunnerContext` | `structs/runner.py` | A frozen declaration on a program, and the host-side surface it gets. That surface is `execute()`, params, controls, status, blockers and `request` for a `Pick`. |
| `RunContext` | `structs/program.py` | The surface a *running* program gets: params, devices, `publish`, `status`, `check_cancel` and `sleep`. |

Changes in this round:

- **`RunContext.frames`, `RunContext.continuous` and the `Continuous` runner are removed.** The number of frames is an ordinary acquisition parameter, and each program loops over it with `ctx.check_cancel()`. If a program ever wants "until stopped", that is that program's own parameter.
- **`run_bridge.py`, `runner_host.py` and `Application.start_run`/`stop_run` are merged into one Qt adapter,** since split by job: running in `Runs` (`app/runtime/`), the selected program's runners in `Runners` (`app/model/`). Nothing reaches through `app.bridge` any more.
- **A run's params are a snapshot** (`resolve_block` always copies). Edits in the form no longer reach a run already going. Two overlapping runs that declare the same block never share one instance.
- **`RunnerContext.stop` and `.running` are removed.** No runner used them. The host still stops the slot's runs.

## Limits that remain, on purpose

The core allows overlapping runs, but the GUI still gives you one at a time.
Each of these limits is a place a later feature will change:

1. **One slot.** The acquisition panel shows the selected program's runners.
   Its Stop button stops only the runs those runners started.
2. **Switching programs stops the old slot's runs.** Once a run's slot is gone, no UI can reach it to stop it. This stays until a runs panel exists (see A).
3. **Busy devices are not blockers.** A start that hits a held device is refused with a message. It isn't greyed out in advance, and it isn't queued.
4. **One `SaveTarget` per session.** Every run saves under the same name and folder.
5. **Status is unscoped.** The acquisition panel shows the status of any run, including one from a previous selection that is still winding down.
6. **No live parameter edits.** Changing a field mid-run no longer affects that run. The inbox (C) is the supported replacement.
7. **A pick is one-shot and happens before the run.** `request` is answered once, by a dataset panel, before `execute`.

## Features, and the change each one needs

The features are ordered by how much they need. Each one names the smallest change that makes it possible.

### A. See and stop every run (runs panel)

**Needs:** `RunEvents` to keep a list of active runs, and a panel that lists them and has a Stop button for each. All the events a panel needs already carry the `Run`.

**Unblocks:** removing limit 2 (switching programs no longer has to stop anything) and limit 5 (each row shows its own run's status).

### B. Runners that drive runs (orchestration)

A runner is a *policy that drives runs*, not just "a way to start the selected program". Each item below is a runner:

- **`Single`:** start one run.
- **`ArmAndRun`:** wait for a pick, then start.
- **A time-lapse:** start, wait until finished, sleep, repeat.
- **A sequence:** run a confocal scan, then Raman at the points it found.
- **An autonomous loop:** acquire, analyse, decide, act, repeat.

**Needs:**
- `RunnerContext.start(program, params) -> RunHandle`. `RunHandle` is an abstract type in `structs/`, and `app.runtime.executor.Run` implements it. `structs/` can't import `app/`, and runners only need `id`, `running`, `stop()`, `datasets` and `on_finished`. `execute()` then becomes shorthand for starting the attached program.
- `RunHandle.on_finished(callback)`, so a runner can chain runs. This is what a time-lapse needs.
- A way to declare runners that belong to no program. Register "experiments" through a registry in `structs/`, with the same `Runner` type, and let the GUI offer them next to programs. `program.runners` stays as the runners offered when a program is selected.
- The runner's params are not the program's params. An experiment declares its own blocks, such as interval, repeats or thresholds, the same way a program does.

**Rule to keep:** a runner never touches Qt. Its callbacks arrive on one thread (the GUI thread today, via `RunEvents`), so runner code needs no locks.

### C. Talking to a running program (inbox)

**Needs:**
- A `Message` base class in `structs/`. `Pick` could become one of its subclasses, or sit beside it.
- A class attribute `Program.accepts: list[type[Message]]` next to `emits`. This lets routing be checked before a run starts, the same way outputs are.
- `RunHandle.send(message)` from outside, and `RunContext.receive(kind)` / `RunContext.poll(kind)` inside. It's a thread-safe queue per run. The program decides when to look, typically between frames next to `check_cancel()`.
- Sending a kind the program doesn't `accept` is an error at the sender.

**Unblocks:**
- Retargeting or FRAP during a scan (a display sends a `PixelPick` to a running run)
- Live retuning (the form sends a "block changed" message, restoring limit 6 deliberately)
- Feedback from an analysis loop (a runner from B sends targets it computed)

**Rule to keep:** the running program never learns who sent a message. Displays, runners, other programs and scripts all look the same to it.

### D. Picks that grow

These build on what exists and need no redesign.

| Feature | Change |
|---|---|
| Box, line or ROI | A new `Pick` subclass, plus a drag interaction on the panel that can answer it. |
| Pick on a spectrum | A new subclass, plus `set_pick_mode` on the spectrum panel. |
| Pick N points, then run | `request(kind, callback, count)`, or a runner that re-requests. Add a "done" gesture on the display so picking mode doesn't flicker between points. |
| Cancel from the display (Esc or right-click) | A `cancelled` signal on `DataPanel`, routed like `picked`. |
| Show the picked point afterwards | Params-to-display: a panel draws markers from any block field of a known kind (such as `Point`) in the selected program's params. |
| Pick without a display (typed, stage readout) | Loosen `Pick.dataset` into a subclass concern, and let a non-panel source answer requests. |
| Refuse unsuitable data before the click | Give a `Pick` kind a `accepts(dataset) -> bool` that the panel asks before it arms. |
| Typed `request` | Make `request[T: Pick](kind: type[T], callback: Callable[[T | None], None])`. This removes the `isinstance` in `aim_at_pick`. |

### E. Queuing and scheduling

**Needs:** a queue in `Executor` or `RunEvents` that holds a start refused with `DeviceBusy`, and retries when a lease is released (`finalize` already releases leases under the executor lock). Show held devices in the device panel.

**Unblocks:** limit 3. It also lets an autonomous loop queue its next step behind a side program instead of failing.

### F. Per-run saving and provenance

**Needs:**
- `start` takes the `SaveTarget` per run (it already does at the executor), so a runner from B can name and place each run it starts. The workspace-wide `SaveTarget` becomes the default for runs started by hand.
- `Provenance` gains `started_by` (runner or experiment name), `parent_run` (the run whose output led to this one) and a log of received messages (C).

**Why:** a days-long autonomous experiment has to be reconstructable afterwards. You need to know why the loop did what it did, not only what each run produced.

### G. Headless and long-lived operation

**Needs:** a `RunnerContext` implementation without Qt, a sibling of `RunnerSession` that dispatches callbacks on a plain event queue. `Executor` already needs no Qt, and after B, experiments don't either.

**Unblocks:** scripting, remote control, and loops that survive the GUI being closed or restarted. It also allows resuming after a crash later: persist the experiment's state and the ids of runs in flight.

## Suggested order

1. **A (runs panel).** Small, and removes the most awkward current limit.
2. **B (orchestration) with `on_finished`.** This is the first real use of overlapping runs.
3. **C (inbox),** when the first in-run feedback feature arrives.
4. **F (provenance),** together with the first autonomous loop.
5. **D, E and G** as specific features call for them.

## Invariants

- `structs/` stays abstract and has no Qt. `RunHandle` and `Message` belong there. `Run` and queues belong in `app/`.
- Programs and panels never import each other. Anything passed between them is a type in `structs/` (a pick, a message, a dataset).
- A running program talks to the world only through its `RunContext`. It never sees the GUI, a runner or another run.
- Contention is about devices, not about runs. What stops two runs is a lease, never a global "is running".
- A run's inputs are fixed when it starts: a params snapshot plus whatever arrives through its inbox. There is no shared mutable state between the form and a worker thread.
