# Autonomous microscopy, part 1: the engine

This is the first of two implementation PRs for autonomous microscopy. It
changes `core/`, `engine/` and `app/` only, and adds no autonomous programs or
panels. Those come in part 2 (`plugins_plan.md`), which builds on this one.

It is a plan to review, not a spec. An agent turns each phase into code. Words
come from `pyrpoc/core/README.md`.

**Ground rules:**
- **No backwards compatibility.** Old sessions and recordings need no support.
  The format-1 key conversion in `engine/data_library/format.py` can go.
- **No speculative options.** Every input, field and argument exists because
  something reads it.
- **Core changes on purpose.** An Experiment and live program inputs are
  architecture, so `core/README.md` is updated in the same phase as the code
  that adds them.
- **One commit per phase.** Each phase leaves the app working and the checks
  passing.

---

## What part 2 needs from this PR

| Part 2 needs | Built in |
|---|---|
| Runs that belong to an Experiment, saved into its folder | Phase 1 |
| A Library that shows experiments clearly, including older ones | Phase 2 |
| Survey refuses to run without a galvo calibration in the Experiment, and the loop refuses to run without a survey | Phase 3 |
| The loop reads "the survey" from the Experiment: the latest by default, changeable, and recorded | Phase 3 |
| You steer a loop that's already running | Phase 4 |
| Units, which may number in the thousands later, kept out of the session file | Phase 5 |

---

## Phase 1: The Experiment

An **Experiment** is something you open, the way you open a project. It is a
folder on disk, and the app knows which one is open.

- [ ] **Experiment is a noun in `core/`.** It's a folder with a small
  `experiment.json`, holding the name and when it was created. It is
  described in `core/README.md`.
- [ ] **Open, create and close** from the menu bar. You can open at most one
  at a time.
- [ ] **The open Experiment is working state.** The session remembers it and
  reopens it at startup.
- [ ] **Saving goes into the open Experiment.**
  - With no Experiment open, ordinary programs still run, and their outputs go
    to the Library as **Unsaved**.
  - Turning saving on with no Experiment open asks you to open or create one.
  - What `SaveTarget` holds now (folder plus name) becomes the Experiment plus
    the run's name. The folder is the Experiment's, not something you set per
    run.
- [ ] **Each recording knows its Experiment,** which is the folder it sits in.
- [ ] **Done when:** you create an Experiment, run a confocal frame and a
  spectrum with saving on, and both recordings land in that folder. A run with
  no Experiment open shows as Unsaved.

## Phase 2: The Data Library, organized by Experiment

The Library is where you go to find things. It should read as experiments
holding runs, not as a flat list of outputs.

- [ ] **Groups:** the open Experiment first, then any other Experiments you've
  loaded, then Unsaved.
- [ ] **Within an Experiment,** runs are listed in the order they happened,
  each with its program's display name, its name and its time. A run's outputs
  sit under it.
- [ ] **Load another Experiment** to view it without opening it. Runs still
  save only into the open one.
- [ ] **Source pickers group the same way.** Each data panel still offers only
  the kinds it renders.
- [ ] **A visual pass on the Library panel:** cards and empty states that
  match the rest of the app (see #59).
- [ ] **Done when:** with two Experiments loaded and some unsaved outputs, the
  Library and every source picker show three clear groups.

## Phase 3: What a program needs from its Experiment

Some programs depend on earlier runs in the same Experiment. Survey needs a
galvo calibration, and the loop needs a survey. Most programs need nothing,
and are unchanged.

- [ ] **A program declares the earlier runs it reads.**
  - Each is named by the program it came from (its `program_key`), never by
    filename.
  - It's declared like a program input, and appears as one in the Acquisition
    panel: a dropdown of the matching runs in the open Experiment.
  - The latest is chosen by default, and you can pick an earlier one.
- [ ] **A program that's missing one refuses to start, and says why.**
  - For example: "Survey needs a galvo calibration in this Experiment."
  - This goes through the existing `Runs.blockers`, so Play and arm-and-run
    are disabled with the reason shown.
- [ ] **The run is handed what it reads.**
  - When the run starts, the chosen recording is loaded and given to the run
    through `RunContext`, alongside its input values.
  - A program never reaches into the Library or the folder itself.
- [ ] **The recording records it.** Each recording lists the recordings it
  read (`inputs` in `_meta.json`), so calibration → survey → loop can be
  followed on disk.
- [ ] **Done when:** a test program that reads another program's recording
  refuses to start without one, uses the latest by default, can be pointed at
  an earlier one, and its recording names the one it used.

## Phase 4: Live program inputs

Today a run gets a frozen copy of its input values when it starts, so nothing
you do afterwards reaches it. The loop needs your edits while it runs:
- run an action now;
- add or remove a unit;
- pause or resume a unit;
- change a routine's settings.

The edits stay program inputs. Panels set them exactly as they set any other
input, and there's no new channel between panels and programs.

- [ ] **A program declares which of its inputs are live.** Everything else
  stays frozen at start, exactly as today.
- [ ] **A run reads a live input's current value when it chooses,** through
  `RunContext`. A program reads between its own steps, so a value never changes
  under it mid-step.
- [ ] **No leaks:**
  - The program sees only values: no `ProgramInputs`, no subscriptions, no Qt,
    and no knowledge of which panel set a value.
  - The panel sees only program inputs: it never knows a run is reading.
  - The engine owns the hand-off across threads.
- [ ] **Each change a run reads is logged in its recording,** with the time it
  was read. A recording then shows what the run was actually given, and when.
- [ ] **Room for later:** an experiment's own logic changing the plan mid-run
  is not in this MVP, but it should be possible later. Nothing here should
  assume that only a panel ever changes a live input.
- [ ] **Done when:** a test program that loops over steps picks up a live
  input changed from a panel at its next step, and its recording logs the
  change.

## Phase 5: Program input values kept beside the session

Units are a program input, set in a panel the way masks are. Hand-drawn units
are few, but identified units can number in the thousands. That's too many to
store inline in the session file and in every recording's parameters.

- [ ] **A field can mark its values as large.** A large value is saved as its
  own file beside the session, and beside each recording that used it. The
  session and the parameters hold only a reference to that file.
- [ ] Loading a session or a recording loads the file back. The program and
  the panels never see the difference.
- [ ] **Done when:** a test input with a large value round-trips through
  session save and restore and through a recording, and neither JSON file
  contains the value.

---

## Out of scope for this PR
- Every autonomous program, data kind, writer and panel (part 2).
- Linking selection between panels.
- Resuming a crashed run.
