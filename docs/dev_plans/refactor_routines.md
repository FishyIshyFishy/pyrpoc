# Refactor: routines — directory structure spec

**Branch:** `refactor/routines` (from `main`)
**Phase:** directory structure only. Per-subdirectory detail is planned separately, after this is agreed.

This document fixes the **folder-level** shape of the refactor and the invariants every
folder must obey. It deliberately does **not** design the internals of each subdirectory —
that happens in a follow-up pass, one subdirectory at a time.

---

## 1. What this refactor is fixing

The `main` architecture has the right *ideas* (registries, a declarative parameter system,
per-modality separation) but three structural problems:

1. **`BaseModality` is a god object** — it fuses the run engine, the threading loop, storage,
   optocontrol handling, and parameters into one class. Every acquisition concern lives in it.
2. **Arbitrary modularity** — a single concept (e.g. "confocal") is scattered across
   `modalities/confocal/{confocal,acquisition_core,parameters,storage}.py`, and the shared scan
   code is *duplicated* between confocal and split-confocal rather than shared. Modular in form,
   tangled in practice.
3. **Declarations that can silently lie** — e.g. FLIM declares `allowed_optocontrols = [Mask]`
   but never applies masks. Compatibility is asserted on paper, not derived from behavior.

The refactor's goal: a self-contained **acquisition subsystem** with a shared async engine; a thin
**orchestration core** above it; **modifiers as inert data** that each modality realizes in its own
way; **routines** as inert user-facing data; and **compatibility derived from implementation** and
checked before a run.

---

## 2. Vocabulary (plain language)

- **Parcel** — the unit of acquired data that flows out of acquisition. **Not one envelope type —
  a base `Parcel` with concrete, *structural* subtypes** (e.g. an image-frame parcel, a
  partial/streaming frame, a histogram cube). Different parcels legitimately carry different fields
  (a complete frame has no sequence index; a streaming chunk does), so they are **distinct types**
  rather than one fat optional-field blob — anything else is implicit duck-typing. Parcel types are
  **registered by type** (not by string) so the manifest layer can reference them. The subtypes are
  **structural and shared across modalities** (`ImageFrameParcel`, `HistogramCubeParcel`) — never
  per-modality identity (there is no `ConfocalParcel`). Replaces `AcquiredData` + the `DataKind` enum.
- **Instrument** — a piece of hardware the software controls (DAQ card, TimeTagger, stage). An
  ordinary object that owns the device connection and exposes control/read methods. Knows nothing
  about acquisition or the GUI. This is where a vendor C++/Python SDK gets wrapped and given
  convenience functions. A modality that uses an instrument just *uses its methods* — there is no
  abstract "capability" layer.
- **Modality** — developer-written **code** for *a way of acquiring data* (confocal, FLIM).
  Produces parcels; orchestrates instruments directly. Its manifest declares the **instruments it
  requires**, the **parcel type(s) it emits**, and the **modifiers it can realize**; its code
  contains the realization of each compatible modifier. Users never author modalities.
- **Modifier** — a **dataclass**: inert configuration, no behavior of its own, living in
  `acquisition/modifiers/`. It describes *what the user wants* (e.g. a mask + which DAQ line; a
  "reacquire the region I draw" toggle). The **modality** reads the modifier dataclass and decides
  what it means for *that* modality. The same `MaskModifier` is realized differently by confocal
  (full-dwell TTL) and split (TTL gated to `t0`) — that difference lives in the modality's code, not
  in the modifier.
- **Routine** — user-authored **data** (never code): an ordered sequence of modality "blocks"
  plus, per block, which modifiers are *available*. Defines what the user sees and can do in a
  session. Inert — the engine interprets it; it holds no logic and is never a code pinch point.
- **Manifest** — a machine-readable declaration, **derived from the implementation**, of what a
  plugin needs and produces. A **modality's** manifest declares its **required instruments** (the
  concrete instrument types it uses), its **emitted parcel type(s)**, and the **modifier types it
  can realize**. A **display's** manifest declares the parcel type(s) it accepts. The compatibility
  checker reads manifests; nobody hand-maintains a separate wishlist.
- **Run engine** (in `acquisition/runner/`) — the shared async machinery that runs *one* configured
  modality on a worker thread: it provides `emit(parcel)`, `should_stop()`, and a feedback channel,
  and owns start/stop/error/finalize. Modalities never reimplement threading.
- **Core / orchestration** (in `core/`) — the application layer *above* acquisition: it interprets
  the routine, runs the compatibility check, drives the acquisition subsystem block by block, and
  routes emitted parcels to displays and storage. Knows nothing about *how* any modality acquires.
- **Display** — a GUI widget that consumes parcels and draws them; declares (via manifest) which
  parcel type(s) it accepts.
- **`ScannerModality`** — an optional **DRY helper**, not an architectural mandate: a base with
  galvo/DAQ convenience functions a scanning modality *may* inherit. It must never constrain what a
  modality can do.

---

## 3. Invariants (every folder obeys these)

1. **Core and acquisition are Qt-free.** `gui/` is the only package that imports PyQt. The engine
   drives the UI through a plain `emit(parcel)` callback + plain APIs; the GUI adapts those to Qt
   signals.
2. **Dependency direction is one-way, bottom → top:**
   `structs → instruments → acquisition → core → gui`.
   Nobody imports upward. Plugins (modalities, displays, instruments) and parcel types are
   discovered through **registries keyed by type / typed key — not bare strings.** Avoid stringy
   keys everywhere; a stable string identifier is tolerated **only** at the persistence boundary,
   where a live class reference can't be serialized.
3. **Parcels are typed by *structure*, not identity.** A base `Parcel` with concrete structural
   subtypes (image-frame, histogram-cube, …), registered by type. Many modalities emit the *same*
   parcel type; displays accept parcel *types*. Never a per-modality parcel (`ImageFrameParcel`
   shared by confocal and split is right; `ConfocalParcel` is wrong) — that is how "structure not
   identity" survives even though parcels are now distinct types.
4. **Modifiers are data; modalities are behavior.** A modifier is an inert dataclass; the modality
   that lists it as compatible contains the code that realizes it. There is **no generic
   `for modifier in modifiers: apply(modifier)` loop** — realization is modality-specific by design.
5. **Compatibility is manifest-driven and derived from implementation**, checked at "play" time.
   A modality's manifest states which modifiers it can realize; the checker validates a routine
   against that and **warns or halts with a plain-language reason** (no display enabled; a modifier
   a modality can't realize; a streaming-only display with a non-streaming modality; …).
6. **Routines are inert user data**, never a code pinch point. Sequencing = data (the routine);
   step behavior = code (a modality); interpretation = the orchestration core.
7. **Confined if/else is acceptable inside a modality's modifier realization.** This is the one
   place we explicitly allow branching ("if this modality is doing X, apply the modifier this way").
   It is isolated per modality so it can be refactored on its own later.
8. **`structs/` carries no Qt, no hardware, no logic** — only data types. (Parameter *declarations*
   live here; the widgets that render them live in `gui/`.)
9. **DRY bases are conveniences, not frameworks.** `ScannerModality` helps avoid copy-paste; it
   must never dictate a modality's shape.
10. **No modality dropdown.** Routines are the single entry point for choosing and configuring what
    happens. Modality change *is* high-stakes and is wrapped inside routine change; frictionless
    back-and-forth is a later UX nicety, not a structural driver.
11. **We hold objects, not ids.** Instruments/displays/modifiers are stored as the live objects
    themselves. There is no `instance_id` field — identity is object identity.

---

## 4. Proposed top-level directory structure

```
pyrpoc/
  structs/           # universal data types: parcel types, manifests, parameter declarations, routine schema, typed keys.
  instruments/       # hardware control: SDK wrappers + convenience control functions.
  acquisition/       # the acquisition subsystem: how data actually gets acquired.
    runner/          #   the shared async run engine (threading, lifecycle, emit/feedback).
    modalities/      #   developer code: ways of acquiring. + ScannerModality DRY base.
    modifiers/       #   modifier dataclasses (inert config): mask, reacquire-region, ...
  core/              # orchestration: routine interpretation, compatibility, parcel routing, storage.
  gui/               # all UI: routine editor, settings menu, displays, instrument widgets, docking.
  main.py
```

`modalities/` and `modifiers/` live **inside `acquisition/`**, never at the top level. The
dependency layering is strictly bottom → top:

```
structs                         (no internal deps)
  └─ instruments                (deps: structs)
       └─ acquisition           (deps: structs, instruments)
            └─ core             (deps: structs, instruments, acquisition)
                 └─ gui         (deps: all, via core's API + registries)
```

`acquisition/` is ignorant of `core/` and of routines — it only knows how to run *one configured
modality with its enabled modifier dataclasses*. `core/` is the only thing that knows about routines
and drives `acquisition/` block by block.

---

## 5. Per-folder responsibilities and boundaries

### `structs/`
The shared vocabulary. Pure data types with no behavior, no Qt, no hardware imports. The chosen set
(to be refined during the detailed pass):

- **Parcels:** a base `Parcel` + concrete **structural** subtypes (`ImageFrameParcel`,
  `PartialFrameParcel`, `HistogramCubeParcel`, …), each carrying only the fields that make sense for
  it; a **parcel registry keyed by type**. Supporting: `Axis` (named/units axis, for parcels that
  need labeled coordinates), light `Channel` info if needed.
- **Parameters (declarations, Qt-free):** `Parameter` base + `Number` / `Text` / `Path` / `Choice`
  / `Checkbox` / `ChannelSelection`; `ParameterValue`; `ParameterGroup(s)`; a resolved/validated
  `Config`; `ValidationResult`.
- **Manifests:** `Manifest` base; `ModalityManifest` (required instrument types, emitted parcel
  type(s), realizable modifier types, parameter groups); `ModifierManifest` (what a modifier needs —
  a DAQ line, a source display); `DisplayManifest` (accepted parcel types).
- **Typed keys (anti-stringy):** `ModalityKey`, `ModifierKey`, `DisplayKey`, `InstrumentKey` — typed
  identifiers used by the routine and by registries in place of bare strings.
- **Routine:** `Routine`; `RoutineBlock` (a modality key + its parameter values + its modifier
  slots); `ModifierSlot` (modifier key + available? + enabled? + config values). Note: the routine
  references modifiers by **typed key + config values**, because the concrete modifier *dataclass*
  lives up in `acquisition/modifiers/` (structs can't import upward) — the acquisition layer
  rebuilds the concrete dataclass from the key + values.
- **Feedback / geometry:** `Region` (used for a partial parcel's coverage and for draw-a-box
  reacquire); `FeedbackEvent` (a deliberately open base — no fixed "selection" type).
- **Status / results:** `ConnectionStatus` (enum); `CompatibilityReport` (the checker's WARN/HALT
  issues with plain-language reasons); `RunContext` (the fully-resolved spec for one run, handed
  from `core` to the runner).
- **Explicitly dropped:** `Capability` / `InstrumentManifest` / `CapabilitySet` (modalities use
  instruments directly), `InstanceId` (we hold objects), a central `SessionSnapshot` (objects
  serialize themselves — see §7), a standalone `Provenance` (provenance is fields on the parcel
  subtypes that need it).
- **Explicitly not:** any widget, device call, or acquisition/orchestration logic.
- **Depends on:** nothing internal. Everything else depends on it.

### `instruments/`
One folder/module per physical device — the hardware-control subsystem. This is where a new
instrument's vendor SDK gets wrapped and given convenience functions so it is pleasant to use from
outside this package.
- **Contains:** a base + registry; `ni_daq` (now an instrument — the galvo/AI/AO/DO/counter card),
  `time_tagger`, `prior_stage`, `zaber_stage`, future devices. Each is an ordinary object exposing
  control/read methods and its own observable state (e.g. a stage's live position).
- **Explicitly not:** anything that knows an acquisition or a GUI exists. Instrument *widgets* live
  in `gui/` and only *observe*/command the instrument — they never reimplement control.
- **Depends on:** `structs`.

### `acquisition/` — the acquisition subsystem
Everything about *how data actually gets acquired*. Self-contained: given a configured modality plus
its enabled modifier dataclasses and parameter values, it runs and emits parcels. It does not know
what a routine is.

- **`acquisition/runner/` — the shared async run engine.**
  - **Contains:** the threading/worker machinery; the run lifecycle (start / stop / error /
    finalize); the plumbing that hands a running modality `emit(parcel)`, `should_stop()`, and the
    feedback channel. Written **once** so no modality reimplements threading (this is the piece that
    was fused into `BaseModality`).
  - **Explicitly not:** any modality-specific logic; any routing/storage decisions (it just emits).

- **`acquisition/modalities/` — the ways of acquiring (developer code).**
  - **Contains:** the concrete modalities (confocal, split-confocal, FLIM); `ScannerModality`, a DRY
    base of galvo/DAQ convenience functions a scanning modality may inherit. Each modality: declares
    its manifest (emitted parcel type(s), required instrument types, **which modifiers it can
    realize**); implements its run (producing parcels via the runner's `emit`); and contains the
    code that **realizes each compatible modifier** in its own context (the confined if/else lives
    here).
  - **Explicitly not:** threading (that's the runner); the GUI; storage.

- **`acquisition/modifiers/` — modifier dataclasses (inert config).**
  - **Contains:** one dataclass per modifier type — `MaskModifier(mask, daq_port, daq_line)`,
    a reacquire-region modifier, etc. Pure data: the configured values plus a manifest of what the
    modifier needs. **No behavior** — the realization lives in whichever modality declares
    compatibility. Registered by type so the routine can reference them by typed key.
  - **Explicitly not:** the mask-drawing editor (that's `gui/`); the TTL-generation code (that's the
    realizing modality); any per-modality branching.

- **Depends on:** `structs`, `instruments`.

### `core/` — orchestration
The application layer above acquisition. This is the "conductor": it turns the user's routine and the
current inventory into runs, and moves parcels to where they're shown and saved.
- **Contains:** routine interpretation (read the block sequence + available/enabled modifiers; drive
  the acquisition subsystem block by block on "play"); the **compatibility checker** (derive from
  manifests, validate a routine before a run, warn/halt); **parcel routing** (fan emitted parcels to
  compatible displays and to storage); **data storage** (parcels → disk: TIFF/NPZ/…); app/session
  state and the wiring that connects instruments, acquisition, and (through `gui`) displays.
- **Explicitly not:** the run engine (that's `acquisition/runner/`); any specific modality/modifier
  knowledge (reached via registries + manifests); any Qt; the user-facing session save/load
  *trigger* (that's GUI — see §7).
- **Depends on:** `structs`, `instruments`, `acquisition`.

### `gui/`
Everything Qt. The only package that imports PyQt. Its internal shape is expected to change a lot
post-refactor, so this spec fixes only its *boundary*, not its layout.
- **Contains:** the **routine editor** (the block-ordering, LabVIEW-like config); the always-visible,
  collapsible **settings menu** (shows the modifiers a routine made *available*, and lets the user
  *enable* each one — availability vs. enablement are distinct); displays (parcel-consuming widgets);
  per-instrument widgets; parameter widgets (render the `structs` declarations); docking, theming;
  the session save/load trigger and the thin Qt adapters over `core` (the old "services", now split
  so their business half is in `core`).
- **Explicitly not:** business logic, device control, acquisition logic. The discipline here is
  **UI rendering vs coordination** — keep them separable even as the UI evolves.
- **Depends on:** everything below, via `core`'s API and the registries.

---

## 6. The modifier model (the crux of this structure)

Because it is the part most likely to be misbuilt, the modifier flow is pinned here:

1. A **modifier is a dataclass** in `acquisition/modifiers/` — inert config the user fills in
   (via a `gui/` editor for rich ones like the mask), plus a manifest of what it needs. It is
   registered by type; the routine references it by typed key + config values.
2. A **modality declares, in its manifest, which modifier types it can realize.** The compatibility
   checker (in `core/`) validates that a routine only makes compatible modifiers available.
3. At run time, `core/` hands the running modality its **enabled** modifier dataclasses. The
   **modality's own code realizes each one** — reading the dataclass and doing the
   context-appropriate thing (confocal turns a `MaskModifier` into full-dwell TTL; split gates it to
   `t0`). There is deliberately **no generic apply-loop**; the branching is the modality's, and it's
   confined there.
4. **Availability ≠ enablement.** The routine makes a modifier *available* (it appears in the
   settings menu); the user *enables* it in settings for a given run. Both are data; neither is logic.

Feedback modifiers (e.g. draw-a-box-to-reacquire) follow the same shape: the modifier dataclass holds
config; the runner carries feedback events into the running modality; the modality decides what a
box *means* for it and converts coordinates to its own space. The feedback shape is left open — a
modifier declares what it consumes; there is no fixed "selection" type.

---

## 7. Cross-cutting concerns and where they live

- **Parameters** are split: the *declaration* (a Number/Choice/… as data) is in `structs`; the
  *widget* that renders it is in `gui`. This is what keeps Qt out of `structs`/`acquisition`/`core`.
- **Data storage** (acquired parcels → disk) is a `core/` concern — it consumes the emit stream.
- **Session persistence** (app config save/load) is **user-facing → its trigger lives in `gui`**.
  Mechanism: each plugin serializes *its own* state (its config is already declared parameters), and
  a thin coordinator gathers them. **No home-baked codec / schema-version ladder** — use standard
  dataclass/library serialization. Dock layout is stored as an opaque blob the GUI provides. This is
  also the one place a stable **string** identifier for a plugin type is unavoidable (a live class
  can't be serialized) — isolate it here.
- **The manifest + compatibility checker** (`core/`) is the seam that makes the plugin catalogs safe:
  it derives what each plugin needs/produces from the implementation and validates a routine before
  "play".

---

## 8. Start clean — do NOT port these from `main`

Confirmed dead / orphaned in the inventory; the refactor starts without them:
- `backend_utils/array_contracts.py` (tested, never wired in).
- `domain/stores.py::ObjectStore` (defined + tested, unused at runtime).
- `rpoc/` as a package: `RPOCMaskEditor` (never instantiated) and `local_treatment.py` (emits a
  signal that is never defined — would crash). Keep only the live `RPOCImageInput` idea, folded into
  a parcel type in `structs/`; move the pure segmentation helpers next to the mask modifier if still
  wanted.
- `DataKind.FLIM_PARTIAL_HISTOGRAM` (no producer, no consumer) — and the `DataKind` enum itself,
  replaced by parcel types.
- `instance_id` fields throughout — we hold objects, not ids.
- The empty `prior_stage.py` / stub `zaber_stage.py` — reimplement properly as instruments.
- The duplicated confocal/split acquisition cores — share via `ScannerModality`.
- The parallel unused optocontrol context-collection path.

Known behavior bug to fix in passing: FLIM currently declares mask support but silently ignores it —
under the manifest checker this must become either a real realization or an honest "not supported".

---

## 9. Next steps (per-subdirectory detailed planning)

Plan each in dependency order (later ones build on earlier decisions):
1. `structs/` — the parcel types + parcel registry, the manifest format, parameter declarations,
   typed keys, the routine schema. Everything else references these. **(In progress — §5 lists the
   chosen set.)**
2. `instruments/` — the base + the control/read/observable-state interface; `ni_daq` as an
   instrument.
3. `acquisition/` — the runner (async engine + emit/feedback), the modality contract +
   `ScannerModality` DRY base + the three modalities, and the modifier dataclasses + the realization
   mechanism.
4. `core/` — routine interpretation, the compatibility checker, parcel routing, data storage,
   app/session state.
5. `gui/` — routine editor, settings menu, displays, widgets, session trigger, Qt adapters.

Open questions to resolve during the detailed passes:
- The exact realization mechanism: how a modality's code is dispatched per compatible modifier
  (a method per modifier type? a `match` on modifier type? a small table inside the modality?).
- How the routine expresses a *sequence* (confocal → FLIM), and whether steps that need logic
  between them (autofocus, z-stack) are procedure-plugins the routine references by typed key.
- How a `RoutineBlock`/`ModifierSlot` round-trips modifier config (typed key + parameter values in
  `structs`, rebuilt into the concrete `acquisition/modifiers/` dataclass at run time).
- The stable string identifier used only at the persistence boundary (qualified type name? a
  registered token?) and the serialization library.
- Precise boundary between `core/` routine interpretation and the `acquisition/runner` (does the
  runner execute one modality only, with `core` sequencing blocks — current assumption — or does it
  take a multi-step plan?).
