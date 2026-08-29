# pyrpoc rebuild: implementation plan

Companion to `260827-refactor_plan.md`, which describes the destination. This one describes how to get there without breaking the working application along the way.

## The overall approach

The new code is written next to the old code, not on top of it. Most of the new structure (`core/`, `devices/`, `operations/`, `data/`, `run/`, `programs/`) contains no user interface code at all, so it can be built and tested while the existing application keeps running untouched. Nothing imports the new folders until we deliberately connect them.

That means there is exactly one dangerous moment in the whole migration: the point where the interface stops calling the old acquisition system and starts calling the new one. Everything before it carries no risk to the working program, because the working program does not know the new code exists. Everything after it is cleanup.

The alternative — rewriting each folder in place — would leave the application broken or half-converted for weeks. The cost of building alongside is that some hardware code exists in two copies for a while. Since one person is doing this over a few weeks, that is a reasonable trade, but it does mean any bug fix during the migration has to be applied in both places until the old copy is deleted.

## How we check for regressions

A regression is any change in behaviour we did not intend. Three different checks cover three different kinds of it, and none of them substitutes for the others.

### Comparing old and new on identical inputs

Most of the code that could break silently is arithmetic: generating the voltage waveform that steers the galvo, converting a stream of samples into pixels, turning a mask image into a pattern of digital pulses. These functions take numbers in and give numbers out, with no hardware involved, which means we can call the old version and the new version with the same arguments and require the results to match exactly.

This is the strongest check available, because if the arithmetic is identical then the hardware cannot behave differently. It is also the cheapest to write. The relevant functions today are:

- `generate_raster_waveform` in `modalities/helpers/daq.py`
- `extract_kept_samples` and `reshape_to_frame` in `modalities/confocal/acquisition_core.py`
- the equivalents in `modalities/split_confocal/acquisition_core.py`, including the t0/t2 channel splitting
- `resize_mask_nearest`, `preprocess_mask_to_scan_grid`, `generate_mask_ttl_signals` in both scan modules
- `read_flim_frame` and `flim_intensity` in `modalities/flim/acquisition_core.py`
- the toy data generators in `modalities/helpers/toy_data.py`

### Running the whole application without hardware

The codebase already falls back to simulated data when the DAQ is missing. In the new structure that fallback becomes a simulated device, which means the entire path from pressing play to a saved file can run on a laptop with nothing plugged in. Tests can start a program, let it produce a few frames, and check the resulting arrays and files.

"Headless" below means running this way: no microscope, and also no windows on screen, so it can run automatically. Everything outside `views/` and `shell/` is written so it can run headless, which is what makes this check possible at all.

### Checking the structure rules still hold

Section 12 of the design document lists rules such as "`views/` imports nothing from `run/` or `programs/`". These can be checked by a small script that reads the import statements in each file and compares them against the allowed list. This catches a different kind of regression: not wrong behaviour, but the structure quietly reverting to the tangle we are leaving behind. Worth writing once, early, and running with the other tests.

### What none of these cover

Anything that only happens on the real instrument: whether the analog input actually stays aligned with the sample clock, whether the digital pulses arrive at the right pixel, whether the time tagger sees the frame trigger. No test on a laptop can confirm these. The plan therefore includes a manual comparison on the microscope, and it happens in the middle rather than at the end, so that if something is wrong we find out before building more on top of it.

## Phases

### Phase 0 — Record what the current code does

Before changing anything, write tests that call the functions listed above with fixed inputs and store the results. These tests import the old code and pass today; their purpose is to fail later if the new code computes something different.

Also capture end-to-end output: run each of the three modalities with saving enabled and the DAQ absent, so the simulated path is used, and keep the resulting TIFF and npz files as reference.

This phase changes no production code. It is the phase most likely to be skipped and the one that makes every later phase checkable, so it is worth the day it takes.

Done when: the reference tests exist, pass, and are committed.

### Phase 1 — Vocabulary and hardware arithmetic

Create `core/` (stream shape definitions, parameter field types, the mask binding description) and `operations/`, moving the arithmetic across from the three `acquisition_core.py` files. This is a move, not a rewrite: the goal is that the functions compute exactly what they computed before, with clearer arguments.

Nothing in the running application imports these yet.

Done when: the phase 0 comparison tests, pointed at the new functions, produce identical arrays.

### Phase 2 — Devices

Create `devices/` with the DAQ, galvo, time tagger and stages, each with its settings panel moved next to its driver, and each with a simulated version. The galvo becomes a real object here rather than four loose parameters repeated in every modality.

Still not imported by the running application.

Done when: the simulated DAQ produces the same arrays the toy data generators produce today, checked against the phase 0 references.

### Phase 3 — Datasets

Create `data/`: the dataset object that holds acquired arrays, the collection of open datasets, and the saving code. The saving code is written to produce files that match what `modalities/*/storage.py` produces now, so that existing analysis scripts keep working.

Done when: writing frames into a dataset and saving produces files identical to the phase 0 reference files.

### Phase 4 — The runner and the first program

Create `run/` (the worker thread, cancellation, status reporting, and the context object handed to a program) and `programs/confocal.py`.

At the end of this phase there are two complete acquisition paths in the repository: the old one, which the application uses, and the new one, which only tests use. They should produce the same results.

Done when: running confocal through the new path with the simulated device produces the same frames and the same saved files as the phase 0 reference, for several parameter combinations including masks enabled.

### Phase 5 — The other two programs

Port split confocal and FLIM. Two behaviours change deliberately here; see the section on intended differences below.

Done when: the same comparison passes for both, allowing for the intended differences.

### Phase 6 — Connect the interface to the new runner

This is the only phase that can break the working application.

The acquisition panel stops calling `ModalityService` and calls the runner instead. To keep this phase as small as possible, two things are deliberately left alone. Parameters are still collected from the form widgets as they are today and converted into a parameter object at the point of hand-off. Displays still receive frames pushed at them, by way of a small piece of temporary code that watches datasets for new data and calls each display's existing `render` method. That temporary code is deleted in phase 8; its only job is to keep displays working while acquisition changes underneath them.

After this phase the old modality classes are still present but nothing reaches them.

Done when: each of the three modalities can be run from the interface against the simulated device and produces the same results as before, and the hardware check below has passed.

### Hardware check — between phases 6 and 7

Stop here and run the application on the microscope. For each of the three modalities, acquire from a stable sample with the old code and with the new code using the same settings, and compare the images. Include one run with a mask enabled, since the digital output path is the part least covered by the automatic tests. Confirm the time tagger is created once per run rather than once per frame, which is one of the intended changes.

If anything is wrong, it is almost certainly in phase 1 or 2, and the fix belongs there rather than in a patch further up. Do not start phase 7 until this passes.

### Phase 7 — Parameters stop living in widgets

The form is generated from the parameter model and writes back into it, instead of being read out with `collect_values`. The shared groups arrive here too, so the scan and DAQ settings are defined once rather than repeated in three files.

The session file format changes, because parameters are stored differently. Either write a conversion for old session files or accept that saved sessions reset once, and say which in advance.

Done when: parameter values survive a save and reload, switching between programs preserves each one's settings as it does today, and the programs receive the same values they received in phase 6.

### Phase 8 — Displays become views

Delete the temporary bridging code from phase 6. Displays read from datasets instead of being handed frames. Remove `export_rpoc_input` and `accepted_kinds` from the display classes and `allowed_displays` from the programs. Merge the two mask editors — `rpoc/editor.py` and `gui/main_widgets/opto_control_mgr/mask_editor.py` — into one view that reads an acquired dataset and writes a mask file.

Done when: displays render correctly during and after a run, a display can be closed and reopened without losing data, and two displays can show the same run.

### Phase 9 — Move folders and delete the old code

Rename `displays/` to `views/` and `gui/` to `shell/`, then delete `modalities/`, `services/`, `backend_utils/`, `optocontrols/`, and `domain/stores.py`. Add the import rule check to the test suite.

Done when: the old folders are gone, the full test suite passes, and the import rule check passes.

## Differences that are intended

Three things will differ between old and new output. They are improvements, not regressions, and listing them in advance keeps them from being investigated as bugs.

FLIM creates and frees the time tagger once per run instead of once per frame. Image data should be unchanged; timing between frames will be different, and a multi-frame FLIM run should be noticeably faster.

Split confocal's raw pixel stream moves from a separate npz file written at the end of the run into a normal saved output alongside the intensity images. Any script reading `*_raw_pixel_stream.npz` needs updating.

The session file format changes in phase 7, as described there.

## When something breaks

Through phase 5, the working application is untouched, so recovering from a mistake means deleting or fixing the new folder. Nothing is at risk.

From phase 6 onward, work on a branch per phase and merge only when that phase's check passes. Because each phase has a specific check attached, a failure points at a specific phase rather than at the migration in general.

The one case to be careful about is a problem that only appears on the microscope. If the hardware check between phases 6 and 7 fails, the cause is nearly always in the arithmetic moved during phase 1 or the device configuration written in phase 2, because those are the only places where the instructions sent to the instrument are produced. Resist fixing it at the point where it was noticed.

## Rough effort

Phases 0 through 3 are mechanical and low risk; expect them to go quickly. Phase 4 is where the design is actually tested, since it is the first time a program owns its own loop. Phase 6 is short but needs care. Phases 7 and 8 are the largest interface changes. Phase 9 is deletion, which is fast.

Section 13 of the design document lists what was deliberately left out. Read it again before considering `run/` finished, because one of those items — running one program inside another, needed later for z-stacks and mosaics — would change what the runner has to be, and it is much cheaper to allow for it in phase 4 than to retrofit it.
