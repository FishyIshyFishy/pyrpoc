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
- **Modularity first.** Each rule has one owner, and file and folder
  knowledge stays out of `core/` and `plugins/`. A change that would need a
  new kind of thing to be listed in more than one place is a sign the plan is
  wrong.
- **Core changes on purpose.** Live program inputs and runs that read earlier
  runs are architecture, so `core/README.md` is updated in the same phase as the code
  that adds them.
- **One commit per phase.** Each phase leaves the app working and the checks
  passing.

---

## What part 2 needs from this PR

| Part 2 needs | Built in |
|---|---|
| Runs saved together in one folder, the Experiment | Phase 1 |
| A Library that opens whole Experiments and browses them like folders | Phase 2 |
| Survey refuses to run without a galvo calibration in the Experiment, and the loop refuses to run without a survey | Phase 3 |
| The loop reads "the survey" from the Experiment: the latest by default, changeable, and recorded | Phase 3 |
| You steer a loop that's already running | Phase 4 |
| Units, which may number in the thousands later, kept out of the session file | Phase 5 |

---

## Where phases 1 to 2.5 put things

These phases are about files and folders. Knowledge of files and folders stays
in `engine/data_library/`, and `core/` doesn't learn it.

| Piece | Lives in | Knows about |
|---|---|---|
| The `Library` protocol | `core/library.py` | open program outputs, plus one group label for each. Not folders, recordings or notes. |
| `SourcePicker` | `core/panels.py` | the group labels the `Library` gives it, nothing more |
| How outputs group into recordings and experiments, the tree, sorting, the notes store, `notes.txt` | `engine/data_library/` | `meta_path`, folders, `_meta.json` |
| The save folder, the run's name, the notes box before a run | `engine/acquisition/`, next to `SaveTarget` | what the next run is called, where it saves, its note |

- **Each rule has one owner.**
  - What counts as an Experiment (the folder of `meta_path`) is decided in one
    function, in `engine/data_library/`.
  - `notes.txt` is written by one function. Everything else calls these
    two.
- **Nothing a program reads changes.**
  - Notes are not a program input.
  - Grouping is not part of `Data` or `Provenance`.

## Phase 1: The Experiment is the save folder

To you, choosing an Experiment is just choosing the folder where things get
saved. There is no menu, no open or close, and no new state. "Experiment" is a
way of reading the data that's already there: the recordings in one folder
belong together.

- [ ] **Setting the save folder is setting the Experiment.** It's set where
  the save folder is set today, with the same button beside the name.
  "Experiment" is a word for developers: to the user it is just a folder, and
  no screen text says "Experiment".
- [ ] **No new noun in the app's state.**
  - There's nothing to create or open. A recording's Experiment is the folder
    it sits in.
  - The folder may hold Experiment-wide files the app writes, such as the
    notes file from phase 2.5.
  - The save folder is still the only thing remembered, exactly as today.
- [ ] **No folder set means no Experiment.**
  - Today `SaveTarget.folder` falls back to the working directory. That
    fallback goes.
  - Turning saving on with no folder set is a blocker ("a folder to save
    into"), like the missing name already is.
- [ ] **Saving off works as today.** Outputs stay in the Library as
  **Unsaved**.
- [ ] `core/README.md` says, in a line or two, that the recordings in one
  folder form an Experiment. Nothing in `core/` code changes in this phase.
- [ ] **Done when:**
  - with the save folder set, a confocal frame and a spectrum both land in it
    as recordings of that Experiment;
  - saving with no folder set is refused, and says why.

## Phase 2: The Data Library, organized like a file explorer

The Library is where you go to find things, and it should feel like browsing
folders.

**Words:**
- A **recording** is one run saved on disk: its `_meta.json` and the files
  holding each of its `ProgramOutput`s.
- An **Experiment** is a folder of recordings.

**Neither word becomes a class in `core/`.** The core `Library` still holds
only program outputs. Recordings and experiments are a view that
`engine/data_library/` builds from each output's `meta_path`.

- [ ] **Open a recording:** loads that one run's outputs, as today.
- [ ] **Open an Experiment:** loads every recording in the folder at once.
  - It loads eagerly, under the Library's existing memory limit, which already
    refuses what doesn't fit.
  - Lazy loading is not in this PR. It would change what a `ProgramOutput`
    holds in core, and is only worth it if part 2's loop frames need it.
- [ ] **Shown as a tree,** built in `engine/data_library/`:
  - experiments (folders), with recordings inside them, and each recording's
    outputs under it;
  - a recording opened on its own still appears under its folder;
  - **Unsaved** is its own group.
- [ ] **Sorting works like a file explorer:** by name or by date, ascending or
  descending, from column headers you click.
- [ ] **Each recording shows** its name, its program's display name and when
  it was taken.
- [ ] **New recordings appear in their Experiment as they're saved.**
- [ ] **Source pickers group the same way, without knowing about folders.**
  - The core `Library` protocol gains one method: the group label for an
    output. `engine/data_library/` decides the label (the Experiment folder's
    name, or "Unsaved").
  - `SourcePicker` only shows the groups. Plugin panels are unchanged.
  - Each data panel still offers only the kinds it renders.
- [ ] **A visual pass on the Library panel:** cards and empty states that
  match the rest of the app (see #59).
- [ ] **Done when:**
  - opening an old Experiment folder loads all its recordings under one group;
  - sorting by name and by date both work;
  - unsaved outputs sit in their own group;
  - no file under `core/` or `plugins/` mentions folders, recordings or
    `meta_path` for grouping.

## Phase 2.5: Notes, before and after acquiring

You take notes on what you acquire, and read them back in two ways:
- per recording, in its details, when analysing;
- all together, to remind yourself how yesterday's data collection went.

Notes live in each recording's `_meta.json`.

- [ ] **A notes box in the Acquisition panel.** What you write there before
  pressing run goes into that run's `_meta.json`.
  - **It's run metadata, not a program input.** It sits beside `SaveTarget`
    in `engine/acquisition/`.
    - No program reads it.
    - It's never in a recording's parameters.
    - It never shows up as an input of any program.
  - It clears once a run has taken it, so one note is never copied onto runs
    it wasn't written for. That is the same code that hands the run its save
    settings.
- [ ] **Notes belong to a run, not to each output.**
  - `notes` comes off `ProgramOutput` in core. Today each output carries its
    own copy.
  - `engine/data_library/` keeps one note per run, keyed by the run's id. That
    works for unsaved runs too, which have no `_meta.json`.
  - Every output's details show and edit that one note.
- [ ] **Notes stay editable afterwards** in the Library's details.
- [ ] **`notes.txt` in the Experiment folder:** every note in the
  Experiment in one readable file. Each entry has:
  - when the run was taken;
  - its name;
  - its program's display name;
  - the note.
- [ ] **Only useful entries:**
  - Only saved recordings with a non-empty note get an entry.
  - A run made with saving off, while a folder was set, gets none. Its note
    still shows in its Library details.
- [ ] **`notes.txt` has one writer, and is built from the recordings.**
  - One function in `engine/data_library/format.py` rebuilds it from the
    folder's `_meta.json` notes, in time order, and writes it atomically.
  - Called after a recording saves a note, and after a note is edited in the
    Library. Nothing else writes the file.
  - Because it's rebuilt from disk, it never holds duplicates, never falls out
    of step, and is safe when two runs save into one folder.
  - Its first line says it's generated, and that edits belong in the app.
- [ ] **Done when:**
  - a note written before a run shows in that recording's details and in
    `notes.txt`;
  - editing it in the Library updates both;
  - an unsaved run and a saved run without a note add nothing to
    `notes.txt`;
  - `ProgramOutput` no longer has `notes`, and no program input holds notes.

## Phase 3: What a program needs from its Experiment

Some programs depend on earlier runs in the same Experiment. Survey needs a
galvo calibration, and the loop needs a survey. Most programs need nothing,
and are unchanged.

- [ ] **A program declares the earlier runs it reads.**
  - Each is named by the program it came from (its `program_key`), never by
    filename.
  - It's declared like a program input, and appears as one in the Acquisition
    panel: a dropdown of the matching recordings in the save folder.
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
