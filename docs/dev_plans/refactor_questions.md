# Initial refactor — decision questionnaire

Answer these and I can build the first cut without further round-trips. Every question has a
**recommended default marked `▶`**. Fastest path: reply *"defaults, except A1=b, G1=pydantic, …"*.
Questions marked **★** are the consequential ones actually worth your attention; the rest are
mostly "confirm the obvious".

Fill inline on the `Answer:` line if you prefer (blank = default).

---

## A. Scope & process

**★ A1. What does the *first* refactor cover?**
- (a) ▶ **Vertical slice:** confocal only, end-to-end — `structs` + NI-DAQ instrument (with sim mode)
  + the runner + confocal modality + mask modifier + intensity displays + a minimal GUI to drive it,
  runnable with no hardware. Then port split/FLIM/stages after the shape is proven.
- (b) All three modalities + full backend, minimal GUI.
- (c) Everything including the full block-based routine editor.
Answer: Everything including the full block-based routine editor. It can be minimal in this first edition, but it should be something we can at least iterate on, since functionality relies on it.

**A2. Build strategy on this branch**
- (a) ▶ Build the new structure **in place**, deleting old `pyrpoc/` modules as they're replaced;
  the app may be temporarily broken mid-refactor.
- (b) Build alongside in a parallel package, swap at the end.
Answer: Build alongside in a parallel package, and swap at the end.

**A3. Must the slice run with no hardware attached (sim mode)?** ▶ **Yes** (non-negotiable for dev).
Answer: Yes. If necessary, create a pretend instrument, pretend modalities, pretend modifiers, etc.., and test as you go. Indeed, we will need a new test suite, so making tests as you go for this refactor will be important.

**★ A4. Commits** — (a) ▶ I make checkpoint commits at milestones (skeleton, structs, each
subsystem) / (b) one commit at the end / (c) no commits, you review the working tree.
Answer: Commits at all milestones

**A5. Keep the three planning docs (`refactor_routines`, `structs_candidates`, this) in the repo?**
▶ Yes. And delete `structs_candidates.md` now that it's folded in? ▶ Keep for now.
Answer: Yes, and delete structs_candidates.md

---

## B. `structs/`

**B1. Base `Parcel` shared fields** — (a) ▶ minimal, nothing shared (each subtype carries its own) /
(b) shared save-flag + source reference / (c) shared channel labels.
Answer: Minimal, nothing shared for now (we will figure out shareable stuff later)

**★ B2. Partial vs complete frames** — (a) ▶ a distinct `PartialFrameParcel` type / (b) a
`complete: bool` flag on `ImageFrameParcel`.
Answer: Whatever you think is best, we can change later.

**B3. Initial parcel types** — ▶ `ImageFrameParcel` `[C,H,W]`, `PartialFrameParcel`,
`HistogramCubeParcel` `[H,W,bins]` (FLIM). Add/remove any?
Answer: Whatever you think is best, we can change later.

**B4. Parcel richness** — (a) ▶ ndarray + channel labels only / (b) also carry `Axis`/units metadata
now (needed later for spectra).
Answer: No metadata, just what is necessary for now.

**B5. Typed-key implementation** — (a) `enum` per plugin-kind / (b) ▶ `NewType(str)` wrappers /
(c) frozen dataclass wrappers.
Answer: Enums are good

**B6. Parcel registry key** — (a) ▶ the parcel class object itself / (b) a typed key.
Answer: Whatever you think is best

**B7. Parameter declaration types** — ▶ port the existing six (Number/Text/Path/Choice/Checkbox/
ChannelSelection), Qt widget-building stripped out to `gui/`. Any additions?
Answer: Just the minimal set of widgets for now to retain all previous functionality

**★ B8. Config representation** — (a) ▶ per-modality typed dataclass built via `from_dict` (like
`main`) / (b) a single generic coerced values object.
Answer: whatever you think is best. 

**B9. Routine sequencing in the slice** — (a) ▶ single-block only, but schema designed to allow
multi-block / (b) full multi-block (confocal → FLIM) now.
Answer: single block for now, to simplify the GUI implementation, but with the backend designed to accomodate complexity in the future.

**B10. Routine persistence** — (a) ▶ one in-memory current routine / (b) named saved routine files
(create/load/list) now.
Answer: we can figure out persistence here when we figure out whole software persistence - for now, one in-memory is fine.

---

## C. `instruments/`

**C1. Base instrument mandated interface** — (a) ▶ minimal (identity + optional `connect`/`test`);
each device adds its own methods / (b) mandate `connect`/`test`.
Answer: minimal

**★ C2. Simulation strategy** (how the app runs hardware-free) —
- (a) ▶ a **simulated NI-DAQ instrument** (real vs sim decided at connect time) that returns
  toy waveforms/frames.
- (b) a `sim=True` flag on the one DAQ instrument.
- (c) per-modality toy fallback on `DaqUnavailableError`, like `main`.
Answer: dont have simulation or simulation fall back be a thing in the real path. for simulation, just make an entirely new modality and entirely new instrument and stuff. we don't want to polute real functionality with simulation functionality.

**C3. `ni_daq`** — ▶ port the existing DAQ primitives (AO/AI/DO/counter, waveform, run_daq) into an
`NIDAQ` instrument. Confirm?
Answer: confirm

**C4. TimeTagger** — ▶ port existing `time_tagger.py` as an instrument only if FLIM is in the initial
scope (see A1); otherwise leave for the FLIM port.
Answer: FLIM is in scope, do the port.

**C5. Stages (prior/zaber)** — (a) ▶ defer entirely (they're empty stubs, no modality uses them) /
(b) implement basic move/position now.
Answer: implement placeholder

**C6. Instrument live-state hook** (e.g. stage position streaming) — (a) ▶ defer / (b) design now.
Answer:  whatever you think is best

---

## D. `acquisition/`

**D1. Runner threading model** — ▶ keep `main`'s shape: a daemon worker thread, `should_stop` event,
`on_frame`/`on_error`/`on_finished` callbacks, with `emit` provided by the runner. Confirm?
Answer: whatever you think is best

**D2. Modality contract** — ▶ `run(emit, should_stop, feedback)` + a declared `manifest`. Confirm the
signature?
Answer: whatever you think is best

**D3. Feedback channel** — (a) ▶ a `queue` the modality polls (like `main`'s `PointClickSource`) /
(b) defer feedback entirely from the slice.
Answer: defer feedback, but note the possibilities and make sure we aren't shooting ourselves in the foot to implement it soon.

**★ D4. Modifier-realization dispatch inside a modality** —
- (a) ▶ a `match` on modifier type in one `realize_modifier(modifier, ...)` method.
- (b) a `dict {ModifierType: handler_method}` on the modality.
- (c) a naming convention (`realize_mask`, `realize_reacquire`, …).
Answer: whatever you think is best

**D5. Modifiers implemented in the slice** — ▶ `MaskModifier` only; the reacquire-region feedback
modifier deferred. Include reacquire now instead?
Answer: reacquire deferred

**D6. `ScannerModality` convenience helpers** — ▶ waveform generation, a `run_daq` wrapper,
pixel↔voltage geometry, channel labeling. Confirm scope?
Answer: just whatever code is needed to help with DRY

**D7. Split-confocal raw-sample NPZ stream** — (a) ▶ port it / (b) drop for now. (Only relevant once
split is ported.)
Answer: drop

---

## E. `core/`

**★ E1. Compatibility-check strictness in the slice** —
- (a) ▶ manifest checks: required instruments present, a display accepts the emitted parcel type,
  every available modifier is realizable by the block's modality — warn/halt with reasons.
- (b) also verify the manifest matches reality (derived-from-implementation, see E2).
- (c) minimal (required instruments only).
Answer: the manifest checks occur at runtime, and make sure that there's no confusing inconsistencies, like no displays that can receive the modality being played being able to actually receive the data the modality emits. the compatibility check for now should be blocking, and in a dialog, so that the user has to close the dialog, make some changes, then try again. it should be informative as well.

**E2. "Derived from implementation" mechanism** (the anti-lying-manifest goal) —
- (a) ▶ derive a modality's realizable-modifier set by introspecting which modifier types its
  dispatch actually handles (so the manifest can't claim one it doesn't).
- (b) keep it hand-declared, guard with a test.
- (c) defer to a later pass.
Answer: for now, let's just have the manifests be hand declared. we can figure out automation later.

**E3. Parcel routing** — ▶ port the `acquisition_interpreter` idea: fan each emitted parcel to
attached displays whose manifest accepts that parcel type. Confirm?
Answer: confirm

**E4. Storage format** — ▶ keep multi-page TIFF-per-channel + `_meta.json` (+ NPZ aux for split).
Change anything?
Answer: no changes

**E5. Runtime app-state object** — ▶ a plain `AppState` holding the live instrument/display objects +
the current routine + run status. Confirm?
Answer: yes

**E6. `core` ↔ `gui` mechanism** — ▶ `core` exposes a plain API, accepts an `emit` callback, and
exposes observable run status; `gui` adapts these to Qt signals (no Qt in core). Confirm?
Answer: yes

---

## F. `gui/`

**★ F1. GUI scope in the slice** — (a) ▶ minimal functional UI: pick/configure the (single) routine
block, play/stop, dockable displays, sim-mode warnings / (b) the full block-based, LabVIEW-like
routine editor + settings menu now.
Answer: yes. for the playing, the thing that gets played is whichever block within the routine the user selects as active. the groups of things in the settings menu should be marked by which block they apply to, for now just have two groups of blocks for the two different blocks (or N) that get put into a routine, we will figure out a more elegant way to do this later.

**F2. Toolkit** — ▶ keep PyQt6 + PyQt6Ads docking. Confirm?
Answer: yes

**F3. Displays ported in the slice** — ▶ the intensity displays (streamed, tiled, multichan) for
confocal; the FLIM display when FLIM is ported. Adjust?
Answer: port all displays

**F4. Settings menu (availability vs enablement)** — (a) ▶ defer until after the slice runs /
(b) build it now.
Answer: build now. this is basically what the acquisition tab is currently (where we enter all the acquisition parameters), but bigger in scope, to accomodate everything the routine deems as necessary.

**F5. Theme** — ▶ keep the Breeze dark theme. Confirm?
Answer: i have a separate branch with themes that i will integrate later. for now, breeze dark is fine. 

---

## G. Cross-cutting

**★ G1. Serialization library** (for routine/config/session; replaces the hand-rolled codec) —
- (a) ▶ **attrs + cattrs** (lightweight, plays well with plain typed structs and numpy-free config).
- (b) **pydantic v2** (batteries-included validation + JSON, heavier; could also absorb parameter
  validation).
- (c) plain dataclasses + a thin JSON helper.
Answer: attrs and cattrs is fine

**G2. Session persistence in the slice** — (a) ▶ defer (focus on the run path first) / (b) include
save/load now.
Answer: defer, but make sure we have a persistence mechanism in mind

**G3. Tests** — (a) ▶ unit tests for `structs`/manifest/compatibility + a sim-mode smoke test that
runs confocal and asserts parcels flow / (b) minimal / (c) comprehensive.
Answer: comprehensive

**G4. Keep the `pyright` + `pytest` config?** ▶ Yes.
Answer: Yes

**G5. Old code** — ▶ delete replaced `main`-era modules as their replacements land (git history keeps
them). Confirm?
Answer: No