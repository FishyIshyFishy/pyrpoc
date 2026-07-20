# Incremental migration plan

**Branch:** `refactor/routines`
**Companion docs:** [`refactor_routines.md`](refactor_routines.md) (target anatomy),
[`refactor_questions.md`](refactor_questions.md) (resolved decisions).

## 0. The lesson this plan exists to fix

We built the new architecture from scratch as `pyrpoc_next`. The **backend** came out
clean and well-tested — and importantly it had *nothing to regress*, because there was
no prior backend UI depending on its internals. The **GUI**, however, regressed: we
rebuilt a working, themed, card-based GUI from scratch and lost its look and pieces
(theme, collapsible cards, managers, the mask editor).

The takeaway is a method, not a one-off fix:

> **Evolve working code; never rebuild it. Keep the app running and visually unchanged
> at every step. Change one seam at a time, behind an adapter, always green.**

## 1. Destination and blueprint

- **Destination:** the anatomy in `refactor_routines.md` — `structs / instruments /
  acquisition / core / gui`, with parcels, manifests, routines, and modifiers-as-data.
- **Blueprint:** `pyrpoc_next` is a *proven reference* for the target shapes — every
  layer exists and is unit-tested. We treat it as the answer key, not as the delivered
  app. We reuse its backend and its tests; we do **not** keep its from-scratch GUI.

## 2. Principles (apply to every phase)

1. **Always green.** A phase is done only when: the full test suite passes, the app
   launches, and a GUI screenshot matches the visual baseline (Phase 0). Commit per phase.
2. **Branch by abstraction.** Introduce a seam, put the new implementation behind it,
   cut over, then delete the old. Old and new coexist only during a cutover.
3. **Adapt, don't rebuild — especially the GUI.** Reuse the theme, `instance_card`
   cards, menubar, the four manager panels, the four displays (incl. the FLIM lifetime
   fit), the mask editor, and the DAQ/TimeTagger drivers. Rewire their *backend calls*;
   leave their *widgets* alone.
4. **Characterize before you change.** Capture current behavior in a test before
   touching the code that produces it.
5. **One seam per phase.** Small, reviewable, revertible. No big-bang cutovers.

## 3. Starting point — a decision to confirm

**Option A (recommended): salvage the backend, migrate the original GUI onto it.**
Keep `pyrpoc_next`'s `structs / instruments / acquisition / core` (tested, no
regression). Bring the **original `pyrpoc/gui`** across and strangler-migrate it panel
by panel onto the new `core`. Effort is spent only where the regression actually was.

**Option B: evolve `main` in place.** Start from the original codebase and reshape each
subsystem behind adapters into the new anatomy, GUI last. A single continuous history
from the original, but it redoes backend work that is already done and tested.

**Recommendation: A.** The backend rebuild was not the mistake; the GUI rebuild was.
Don't redo good, tested work — point the original GUI at the new backend and evolve it.

The phases below assume **Option A**. (Under Option B the same method applies, with
extra early phases that reshape each backend subsystem behind its existing service.)

## 4. Phases (Option A)

Each phase ends with: tests green, app launches, screenshot vs baseline, commit.

### Phase 0 — Guardrails
- Add a characterization smoke test of the *original* app: launch the GUI headless,
  run one simulated acquisition through to a display, and save a frame. This is the
  regression detector.
- Capture a **screenshot of the original GUI** as the visual baseline every later phase
  is compared against.
- Exit: baseline screenshot saved; original smoke test green.

### Phase 1 — Original GUI on the new backend (the linchpin)
- Bring the original `pyrpoc/gui` into the tree unchanged (theme, cards, menubar,
  managers, displays, mask editor, docking).
- Implement the **service interfaces the GUI already expects** (`modality_service`,
  `display_service`, `instrument_service`, `opto_control_service`) as **thin adapters
  over the new `core`** — same signals and methods out, new engine underneath.
- Result: the *unchanged* original GUI runs on the new backend.
- Exit: app launches with the original look; a simulated acquisition flows GUI → core →
  display; screenshot matches baseline.

### Phase 2 — Instruments panel
- Reuse the original instrument-manager UI (cards, Test Connection). Rewire it to the
  new `instruments` registry + `core` inventory via the adapter.
- Exit: add/test/remove instruments works; look unchanged.

### Phase 3 — Displays panel + display widgets
- Reuse the four original displays (incl. FLIM lifetime fit) and the display manager.
- Migrate the data type at this boundary: displays consume **parcels** instead of
  `AcquiredData`; routing delivers by parcel type. Contained to producers + these
  consumers.
- Exit: displays render live acquisition; look unchanged.

### Phase 4 — Acquisition tab → routines/blocks (reusing the card form)
- Keep the original collapsible-card parameter form. Evolve it in place:
  - wrap the existing parameter-group cards in **block** groupings;
  - add the **routine editor** as a new Ctrl+R dock (additive — nothing removed yet);
  - drive modality-per-block from the routine editor; remove the modality dropdown only
    once the editor fully replaces it;
  - fold the OptoControls tab into per-block **modifier cards**, porting the original
    **mask editor** as-is; retire the OptoControls tab only once equivalent.
- Do each of these as a small edit to the existing panel, screenshot-checked.
- Exit: routines/blocks UI with the original look; mask editor intact.

### Phase 5 — Reorg, cleanup, rename
- Move files into the final `structs / instruments / acquisition / core / gui` layout.
- Delete confirmed-dead code (`array_contracts`, `ObjectStore`, dead `rpoc` editor /
  `local_treatment`, orphaned enum members).
- Retire the `pyrpoc_next` reference and the legacy package; update `pyproject` scripts.
- Exit: final anatomy; dead code gone; single `pyrpoc` app; tests green.

## 5. Reuse ledger (do not rebuild)

**From the original GUI:** breeze theme, `instance_card` cards, `MainMenuBar`, the four
manager panels' widgets, the four displays incl. FLIM lifetime fit, the mask editor,
NI-DAQ + TimeTagger drivers, PyQt6Ads docking + layout persistence.

**From the new backend (`pyrpoc_next`):** `structs` (parcels/keys/manifests/routine/
parameters), `instruments` (Qt-free + simulated), `acquisition` (runner + thin
modalities + modifiers), `core` (routing/compatibility/storage/app state/controller),
and all of their tests.

## 6. Risks

- **Phase 1 is the linchpin.** The adapters must reproduce the exact signals/methods the
  original GUI relies on. Enumerate them from the original services *before* writing the
  adapters (a small inventory task).
- **Data-type swap (Phase 3)** touches producers and consumers together; keep it scoped
  to the display boundary and land it in one phase.
- **Don't let the reorg (Phase 5) leak earlier.** Files move only once concepts are
  settled, so phases stay small and green.

## 7. Open decision

Confirm **Option A vs B** before Phase 1 — it sets whether we start from the salvaged
backend (A) or from `main` (B). Everything else follows.
