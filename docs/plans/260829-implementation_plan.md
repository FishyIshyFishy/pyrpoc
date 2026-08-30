# pyrpoc v3.1 build: implementation plan

The code is on v3.0.2 with the phase 0 rollbacks applied. This document is the
build specification for v3.1: what to write, where it goes, what it must
compute, and what proves it. It is written to be executed in one pass.

Companion documents:

- `260827-refactor_plan.md` — the destination. Section numbers below refer to it.
- `phase0.md` — the rollback brief.
- `phase0-survey.md` — what the code does now, and what phase 0 removed.

## Status

Phase 0 is complete. The tree is at 111 Python files / ~12,000 lines, the suite
is 162 tests green, `tests/reference/phase0_references.npz` exists and pins
twelve arrays, and the working tree is clean on `refactor-full`.

This plan was first drafted before the rollback. Five things the rollback
changed invalidated parts of it, and the rewrite below is mostly about those
five.

## What the rollback changed about this plan

**The simulated fallback is gone, so "run the whole application without
hardware" is no longer a check.** That was the first draft's main
regression check and it does not exist any more. `modalities/helpers/toy_data.py`
is deleted and every modality now raises rather than producing toy frames.
The replacement is described under "How we check for regressions".

**No simulated device will be built.** Decided for this build: simulation is
not reinstated in any form, not even as a device you must select by hand.
`devices/*/simulated.py` from section 5 of the design document is out of scope.
Anything that can only be observed on the instrument is out of scope too — the
hardware-facing code is moved with the arithmetic pinned, and residual risk is
parked in `operations/` and `devices/`, which is where it can be fixed in
isolation later.

**There is no hardware check phase.** The first draft put a microscope
comparison between phases 6 and 7 and forbade starting phase 7 until it passed.
That gate is removed; all nine phases run in one pass. The justification is
narrow and worth stating plainly: what the microscope check would have covered —
DO pulse alignment to the pixel clock, AI staying locked to the AO sample clock,
the tagger seeing the frame trigger — lives entirely inside
`operations/raster.py` and `operations/tagger.py`. Nothing in phases 6 through 9
can affect it, and a defect found there later is fixed in those two files
without touching anything built on top.

**Three views listed in the design document have no code to port.**
`streamed.py` and `decay.py` were deleted with `streamed_image_display.py` and
`flim_display.py`. `views/` in v3.1 is `image_2d`, `overlay`, `mask_editor`.
FLIM still emits its histogram cube and still saves it; nothing displays it,
exactly as now.

**`devices/stage/` has nothing to port.** `prior_stage.py` and `zaber_stage.py`
were deleted. The folder is not created.

**Two deletions shrink later phases.** `export_rpoc_input` / `get_rpoc_input`
are already gone, so phase 8 only has to remove `accepted_kinds` and
`allowed_displays`. `pyrpoc/rpoc/` is already gone, so there is one mask editor
to move rather than two to merge.

## Decisions taken for this build

| Question | Decision |
|---|---|
| Headless verification | No simulation ships. Tests substitute a fake at the operation seam, so `run/`, `data/`, saving and each program's `run()` are exercised headless. See below. |
| Continuous acquisition | Kept. `RunContext.frames()` yields unbounded when the run was started in continuous mode. |
| Scope of this build | All nine phases. |
| Session file | `schema_version = 7`, no converter. Saved sessions reset once. The store seeds a default DAQ and Galvo so the reset does not leave a dead play button. |

Design calls made while specifying this, each argued where it appears below:

- `active_ai_channels` becomes DAQ device configuration, not galvo.
- The galvo gets `fast_ao` and `slow_ao` only — no per-axis limits, because
  nothing would read them.
- `Trace1D` is not created; nothing emits it.
- Split confocal's raw stream is 4-D, not `Image2D` as section 8.3 states. It
  gets its own contract.
- TimeTagger channel assignments and trigger voltages become device
  configuration; laser frequency, histogram bins and bin width stay run
  parameters.
- The OptoControls panel slot is reused for a Library panel.

## How we check for regressions

Three checks, none of which involves hardware.

### 1. The arithmetic is pinned to the phase 0 golden arrays

`tests/reference/phase0_references.npz` holds twelve arrays recording what
v3.0.2 computes. `generate_references.py` is repointed at `operations/` in
phase 1 and `test_phase0_references.py` must pass unchanged.

**Do not regenerate the npz.** Regenerating it on a changed implementation
silently rebases the thing the tests compare against, which is the one way this
check can fail without anyone noticing. If a reference test fails, the
implementation is wrong.

Two formulas are *not* currently pinned and must be, because they differ between
the raster path and the FLIM path and unifying them by accident would be a real
bug that no existing test catches:

```
raster: pixel_samples = max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))
flim:   pixel_samples = max(2, int(round(dwell_time_us * 1e-6 * sample_rate_hz)))
```

Add `tests/operations/test_pixel_samples.py` with hard-coded expected values
covering the truncate-vs-round difference and both floors. No npz change.

### 2. The layers above hardware run headless with a fake operation

Programs call operations as module-level functions. A test monkeypatches the
function and everything above it runs for real: the runner's thread and
cancellation, dataset creation from `emits`, `ctx.publish` writing into a
dataset, the save policy producing files, status reporting, and the program's
own control flow including its `finally` blocks.

```python
# tests/programs/test_confocal.py
def test_confocal_run_publishes_and_saves(monkeypatch, tmp_path):
    monkeypatch.setattr("pyrpoc.programs.confocal.raster_scan",
                        lambda **kw: fixed_frame)
    ...
    assert len(library.get(run_id, "intensity")) == 3
    assert (tmp_path / "acq_ai0.tiff").exists()
```

This is not hardware testing. It tests our code, with the hardware call
replaced. It is the only automated coverage phases 3 through 8 get, so it is
worth writing properly rather than as an afterthought.

### 3. The structure rules are checked mechanically

`tests/test_import_rules.py` parses every module under `pyrpoc/` and enforces
the section 7 dependency table plus the section 12 invariants. Written in
phase 1 so it guards from the first commit, and tolerant of folders that do not
exist yet.

Additionally `tests/test_headless.py` runs a subprocess that imports `core`,
`devices`, `operations`, `data`, `run`, `programs` and `session`, then asserts
`"PyQt6.QtWidgets" not in sys.modules`. That is what makes rule 4 mean
something: device panels must be imported lazily inside `Device.panel()`, not at
module scope.

### What none of these cover

Everything that only happens on the instrument. Stated once, accepted, and the
reason the hardware-facing code is moved rather than rewritten.

## The target tree

```
pyrpoc/
├── main.py             entry point (unchanged in shape)
│
├── core/               shared vocabulary. no Qt, no hardware, no dataset I/O.
│   ├── streams.py      Image2D, Cube3D, Samples4D
│   ├── params.py       field types, shared groups, sections(), coerce, to/from_dict
│   ├── modulation.py   MaskBinding; load_mask / save_mask
│   └── errors.py       PyrpocError tree incl. DaqError, TaggerError, Cancelled
│
├── devices/
│   ├── base.py         Device; owns_connection / backed_by; config; lazy panel()
│   ├── registry.py
│   ├── daq/            device.py, panel.py
│   ├── galvo/          device.py, panel.py
│   └── time_tagger/    device.py, panel.py
│
├── operations/         plain functions. no self, no loop, no state.
│   ├── raster.py       waveform, run_raster, extract/reshape, raster_scan
│   ├── split_raster.py reshape_to_split_frame, split_raster_scan
│   ├── modulation.py   resize / preprocess / mask_ttl / split_mask_ttl
│   └── tagger.py       run_flim_scan, flim_scan, read/reshape/intensity
│
├── data/
│   ├── dataset.py      arrays + append + change notification + provenance
│   ├── library.py      the open datasets
│   ├── io.py           SavePolicy: tiff per channel, npz for cubes, meta json
│   └── transforms.py   normalize_channels
│
├── run/                pure Python. no Qt.
│   ├── program.py      Program + RunContext
│   ├── runner.py       worker thread, cancellation, status, dataset setup
│   └── claims.py       device resolution along backed_by
│
├── programs/
│   ├── registry.py
│   ├── confocal.py
│   ├── split_confocal.py
│   └── flim.py
│
├── views/              renders datasets. must not import run/ or programs/.
│   ├── base.py
│   ├── registry.py
│   ├── image_2d.py
│   ├── overlay.py
│   └── mask_editor.py
│
├── shell/              may import everything.
│   ├── window.py
│   ├── catalog.py
│   ├── launcher.py
│   ├── param_form.py
│   ├── run_bridge.py
│   ├── devices_panel.py
│   ├── views_panel.py
│   ├── library_panel.py
│   ├── menubar.py
│   └── theme/
│
└── session/
    ├── state.py        SessionState, schema_version = 7
    └── store.py        JSON read/write; seeds default devices
```

`tests/` mirrors it one-to-one, plus `tests/reference/` which stays where it is.

Rough size: about 6,200 new lines, of which ~2,000 is the vendored Breeze theme
carried over untouched, against about 7,000 deleted. Net roughly flat, which is
the expected shape when a redesign is mostly re-siting rather than adding.

---

# Phases

Nine commits, one per phase, each leaving the suite green. Phases 1 through 5
do not touch the running application at all — it keeps using `modalities/` and
does not know the new folders exist. Phase 6 is the switch. Phases 7 and 8 are
the interface. Phase 9 is deletion.

## Phase 1 — Vocabulary and hardware arithmetic

Create `core/` and `operations/`. This is a move, not a rewrite: the functions
must compute exactly what they compute now.

### `core/errors.py`

One exception tree, replacing the three separate `DaqUnavailableError` classes
(one per `acquisition_core.py`, so `except DaqUnavailableError` imported from
one module does not catch the other two — survey open item 1).

```python
class PyrpocError(Exception): ...
class ParameterError(PyrpocError): ...
class Cancelled(PyrpocError): ...
class DeviceError(PyrpocError): ...
class MissingDevice(DeviceError): ...
class DaqError(DeviceError): ...
class TaggerError(DeviceError): ...
```

`DaqError` keeps the current message formats verbatim —
`f"NI-DAQ acquisition failed: {exc}"` and `f"NI-DAQ FLIM scan failed: {exc}"` —
so what the user sees in the error dialog is unchanged.

### `core/streams.py`

Shape contracts. Each is a marker class with an `ndim`, named axes, and a
`validate(array)` that raises `ValueError` on a mismatch.

| contract | shape | dtype | emitted by |
|---|---|---|---|
| `Image2D` | `(C, H, W)` | float32 | confocal, split confocal, FLIM intensity |
| `Cube3D` | `(H, W, B)` | float32 | FLIM histogram |
| `Samples4D` | `(C, H, W, S)` | float32 | split confocal raw pixel stream |

`Samples4D` is a correction to the design document. Section 8.3 declares
`emits = {"intensity": Image2D, "raw_pixel_stream": Image2D}`, but
`reshape_to_split_frame` returns the raw stack as
`(C, total_y, x_pixels, pixel_samples)` — four dimensions, per-pixel samples
still unaveraged. Filing it as `Image2D` would make the contract a lie and
`Image2D.validate` would reject it.

`Trace1D` from section 5 is not created. Nothing emits a trace, and a contract
with no emitter is the same dead weight `DataKind.PARTIAL_FRAME` was.

### `core/params.py`

The parameter model, with no Qt in it. Today `backend_utils/parameter_utils.py`
mixes field definitions with widget construction (`create_widget`, `get_value`,
`set_value`, `connect_changed`); the widget half moves to `shell/param_form.py`
in phase 7. What stays here is label, tooltip, bounds, and coercion.

```python
@dataclass(frozen=True)
class Field:
    label: str
    tooltip: str = ""
    def coerce(self, value: Any) -> Any: ...

@dataclass(frozen=True)
class IntField(Field):      minimum: int|None = None; maximum: int|None = None; step: int = 1
@dataclass(frozen=True)
class FloatField(Field):    minimum: float|None = None; maximum: float|None = None
                            step: float = 0.1; decimals: int = 6
@dataclass(frozen=True)
class TextField(Field):     ...
@dataclass(frozen=True)
class PathField(Field):     dialog_filter: str = "All Files (*)"
@dataclass(frozen=True)
class BoolField(Field):     ...
@dataclass(frozen=True)
class ChoiceField(Field):   choices: tuple[str, ...] = ()
@dataclass(frozen=True)
class ChannelsField(Field): num_channels: int = 9
@dataclass(frozen=True)
class MasksField(Field):    ...
```

Coercion rules carry over from `parameter_utils.py` unchanged, including
`PathField` rejecting empty strings and paths that end in a separator, and
`ChannelsField` coercing to a sorted unique tuple of ints.

Values live in dataclasses whose fields carry their spec in `metadata`:

```python
def int_field(label, default, *, minimum=None, maximum=None, tooltip="") -> Any
def float_field(...) / text_field(...) / path_field(...) / bool_field(...)
def choice_field(...) / channels_field(...) / masks_field(...)
def group(cls, label) -> Any        # a nested group, with its form section label
```

Each returns `dataclasses.field(default=..., metadata={"param": <Field>})`, so
one declaration serves the value, the default, the form and the validation.

```python
class Group:
    """Base for shared groups. Supports ** unpacking into an operation."""
    def keys(self) -> Iterator[str]
    def __getitem__(self, name: str) -> Any
```

`keys`/`__getitem__` are what make section 8.1's `raster_scan(**p.scan, **p.daq)`
work literally, which is what keeps operations on loose keyword arguments
instead of taking group objects.

Module functions:

```python
def sections(obj_or_cls) -> list[Section]   # ordered (label, [(path, Field)]) for the form
def coerce(cls, raw: dict) -> Any           # validate; raises ParameterError
def to_dict(obj) -> dict                    # session + run metadata
def from_dict(cls, raw: dict) -> Any
```

`sections()` walks nested groups first in declaration order, then collects
root-level scalars into a final section labelled "Acquisition".

### Shared groups

One definition each, replacing nine scan fields and five DAQ fields redeclared
across three `parameters.py` files. Defaults are today's defaults.

```python
class ScanGroup(Group):
    x_pixels = 512; y_pixels = 512
    extra_left = 300; extra_right = 20
    fast_axis_offset = 0.0; fast_axis_amplitude = 1.0
    slow_axis_offset = 0.0; slow_axis_amplitude = 1.0
    dwell_time_us = 2.0

class DaqGroup(Group):
    sample_rate_hz = 100_000.0          # FLIM overrides to 1_000_000.0

class SaveGroup(Group):
    save_enabled = False
    save_path = "acquisition"

class ModulationGroup(Group):
    masks: tuple[MaskBinding, ...] = ()

class SplitGroup(Group):
    t0_samples = 1; t1_samples = 0

class TriggerGroup(Group):              # FLIM only today
    frame_trigger_pfi = 0
    pixel_clock_ctr = 0
    pixel_clock_pfi = 1

class HistogramGroup(Group):            # FLIM only today
    laser_frequency_mhz = 80.0
    histogram_bins = 125
    histogram_binwidth_ps = 100
    frame_settle_s = 5e-3
```

`frame_settle_s` is currently a module constant in `modalities/flim/flim.py`.
Section 8.4 reads it off `p`, so it is promoted to a parameter with its current
value as the default — behaviour identical, now visible and adjustable.

What is deliberately *not* here: `device_name`, `fast_axis_ao`, `slow_axis_ao`,
`active_ai_channels`, and the TimeTagger channel/trigger fields. Those are
wiring and calibration, and they become device configuration in phase 2.
`sample_rate_hz` stays a run parameter because it is a per-experiment choice,
not a description of how the rig is cabled.

### `core/modulation.py`

```python
@dataclass(frozen=True)
class MaskBinding:
    path: Path
    port: int
    line: int
    def channel(self, device_name: str) -> str      # f"{device}/port{port}/line{line}"

def load_mask(path: Path) -> np.ndarray    # cv2.IMREAD_GRAYSCALE, uint8, 2-D or raise
def save_mask(path: Path, mask: np.ndarray) -> None
```

This is the one place `core/` does file I/O, which section 5 asks for and
section 6.4's "no I/O" aside argues against. Following section 5: the alternative
is putting mask file loading in `data/io.py`, which is about *dataset* saving and
would muddy it, and `operations/` may not import `data/` anyway (rule 3), so the
load would end up in the program either way. `MaskBinding` itself is imported by
`core/params`, `operations/modulation`, all three programs, `shell/param_form`
and `views/mask_editor` — six call sites across five folders, so it passes the
section 6.4 test comfortably.

### `operations/raster.py`

Everything moves from `modalities/confocal/acquisition_core.py` and
`modalities/helpers/daq.py`, dropping the duplicate copies in
`modalities/split_confocal/acquisition_core.py`.

```python
def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    return max(1, int(dwell_time_us * 1e-6 * sample_rate_hz))

def generate_raster_waveform(...) -> np.ndarray        # verbatim from helpers/daq.py
def extract_kept_samples(...) -> np.ndarray            # verbatim; the two copies collapse
def reshape_to_frame(...) -> np.ndarray                # verbatim from confocal
def run_raster(...) -> tuple[np.ndarray, int, int, int]  # was run_daq; the two copies collapse

def raster_scan(*, x_pixels, y_pixels, extra_left, extra_right,
                fast_axis_offset, fast_axis_amplitude,
                slow_axis_offset, slow_axis_amplitude, dwell_time_us,
                sample_rate_hz, device_name, ai_channels, fast_ao, slow_ao,
                ttl: dict[str, np.ndarray] | None = None) -> np.ndarray
```

`raster_scan` is today's `acquire_frame` with two substitutions:
`mask_contexts` becomes a precomputed `ttl` dict (so the mask work happens once
per run rather than once per frame), and the argument names align with the
device configuration that supplies them — `fast_axis_ao` → `fast_ao`,
`active_ai_channels` → `ai_channels`.

`run_raster` is carried over line for line, including the parts that are easy to
break by tidying: the `/port0/` lowercase test that splits clocked DO lines from
static ones, the single-channel-vs-list `payload` shape, the start order
(AI, then DO, then AO), the `total_samples / sample_rate_hz + 5` timeout, and the
inverted static write in the `finally` block.

### `operations/split_raster.py`

```python
def reshape_to_split_frame(...) -> tuple[np.ndarray, np.ndarray]   # verbatim
def split_raster_scan(..., t0_samples, t1_samples) -> tuple[np.ndarray, np.ndarray]
```

`split_raster_scan` calls `raster.run_raster`; the only genuine differences from
confocal are the DO gating (handled in `modulation.py`) and the t0/t2 sample
split. That is section 8.3's point about where sharing belongs.

### `operations/modulation.py`

```python
def resize_mask_nearest(...) -> np.ndarray               # verbatim, one copy
def preprocess_mask_to_scan_grid(...) -> np.ndarray      # verbatim, one copy

def mask_ttl(masks: Sequence[tuple[MaskBinding, np.ndarray]], *,
             scan: ScanGroup, pixel_samples: int,
             device_name: str) -> dict[str, np.ndarray]

def split_mask_ttl(masks, *, scan, pixel_samples, device_name,
                   t0_samples: int) -> dict[str, np.ndarray]
```

`mask_ttl` is `generate_mask_ttl_signals` taking loaded arrays and bindings
instead of `MaskContext` objects, per section 8.2. `split_mask_ttl` calls it and
then applies the one line that differs — the gate to the first `t0_samples` of
each pixel — by reshaping the flat signal to `(-1, pixel_samples)` and clearing
the tail. Two functions rather than one with a flag, but the shared twenty lines
are shared rather than copied.

Behaviour preserved: masks that are entirely zero after padding are skipped
(no channel appears in the dict), and a preprocessing failure raises
`RuntimeError(f"Failed to preprocess mask for {channel_name}: ...")`.

### `operations/tagger.py`

```python
def pixel_samples(dwell_time_us: float, sample_rate_hz: float) -> int:
    return max(2, int(round(dwell_time_us * 1e-6 * sample_rate_hz)))

def run_flim_scan(...) -> None            # verbatim
def flim_scan(...) -> None                # verbatim
def reshape_flim_frame(...) -> np.ndarray # verbatim
def flim_intensity(...) -> np.ndarray     # verbatim
def read_flim_frame(flim_measurement, ...) -> np.ndarray   # verbatim
```

The two `pixel_samples` functions have the same name in different modules and
different formulas. That is deliberate and pinned by
`tests/operations/test_pixel_samples.py`; do not unify them.

### Also in phase 1

Write `tests/test_import_rules.py` and `tests/test_headless.py` now, so they
guard every subsequent phase rather than being retrofitted in phase 9.

### Done when

- `tests/reference/generate_references.py` imports from `operations/` and
  `core/modulation`, and `test_phase0_references.py` passes with the npz
  untouched.
- `tests/operations/test_pixel_samples.py` passes.
- `tests/core/test_params.py` covers coercion, `sections()` ordering, `**`
  unpacking of a group, and `to_dict`/`from_dict` round-trip.
- `tests/test_import_rules.py` and `tests/test_headless.py` pass.
- Nothing in `pyrpoc/` outside `core/` and `operations/` imports either.

---

## Phase 2 — Devices

Create `devices/`. Driver and panel rejoin: adding a field to a device and a row
to its panel becomes one edit in one folder, which is the whole argument of
section 6.1.

### `devices/base.py`

```python
class Device:
    display_name: str = "Device"
    owns_connection: bool = False
    backed_by: type["Device"] | None = None
    config: type = None                 # a core.params group class

    def __init__(self, instance_id=None, user_label=None)
    self.config: Group                  # instance of cls.config
    def test_connection(self) -> bool   # only meaningful when owns_connection
    def panel(self, parent=None, on_change=None) -> "QWidget"   # lazy import
    def export_state(self) -> dict / def import_state(self, raw) -> None
```

`panel()` imports its widget inside the method body. That is not a style
preference — `tests/test_headless.py` fails if any device module pulls in
`PyQt6.QtWidgets` at import time, which is what makes section 12's rule 4
enforceable rather than aspirational.

The registry key lives at the registration site in `devices/registry.py`, not on
the class, matching how `modality_registry.register("confocal")` works today.

### `devices/daq/`

```python
class DaqConfig(Group):
    device_name = "Dev1"                # was the "DAQ Device" parameter
    ai_channels = tuple(range(9))       # was "Active AI Channels"

class DAQ(Device):
    display_name = "NI-DAQ"
    owns_connection = True
    config = DaqConfig
    def test_connection(self) -> bool   # nidaqmx.system.System.local() names lookup
```

`ai_channels` goes here rather than on the galvo. Section 9's table lumps
"galvo/AI channels" into one row whose point is that they stop being loose
per-modality parameters; analog *input* is not part of the scanner, it is which
detector BNCs are plugged into the card, so it belongs to the card. The
requirement the table is actually making — one definition, not three — is met.

`owns_connection = True` means the device can verify it exists, the way the
TimeTagger's Test Connection button does today. It does not mean the device
holds an open handle: an NI task *is* the clock domain, so tasks are still
created per scan inside `run_raster`.

### `devices/galvo/`

```python
class GalvoConfig(Group):
    fast_ao = 0
    slow_ao = 1

class Galvo(Device):
    display_name = "Galvo"
    backed_by = DAQ                     # no connection of its own
    config = GalvoConfig
```

Two fields, both read by `raster_scan`. No per-axis voltage limits: section 4
mentions them, but nothing would clamp against them in v3.1, and an unused field
is exactly the accumulation section 3's definition test warns about. If limits
are wanted they are a field plus a clamp in `raster_scan`, added when something
needs them.

### `devices/time_tagger/`

```python
class TaggerConfig(Group):
    laser_channel = 1; detector_channel = 2
    pixel_channel = 3; frame_channel = 4
    laser_trigger_v = 0.05; detector_trigger_v = 0.2
    pixel_trigger_v = 0.5; frame_trigger_v = 0.5
    laser_input_delay_ps = 0

class TimeTagger(Device):
    display_name = "Swabian TimeTagger"
    owns_connection = True
    config = TaggerConfig
    def create_tagger / free_tagger / test_connection      # verbatim from today
    def configure_for_flim(self) -> None                   # reads self.config
    def start_flim_measurement(self, *, n_pixels, n_bins, binwidth_ps) -> object
    def stop_flim_measurement(self, flim) -> None
```

Channel numbers and trigger voltages move out of the FLIM parameter form and
into this panel: they describe how the tagger is cabled and thresholded, which
is section 4's "configuration, calibration, a panel, and persistence". What stays
a run parameter is `laser_frequency_mhz`, `histogram_bins`,
`histogram_binwidth_ps` — section 8.4 reads those off `p.histogram`.

`start_flim_measurement` / `stop_flim_measurement` are named for section 8.4 and
wrap today's `create_flim_measurement` and the `flim.stop()` + `free_tagger()`
pair. The `TimeTagger.Flim(...)` call including `n_frame_average=1` is carried
over unchanged.

### Panels

Each `panel.py` builds its form from `core.params.sections(device.config)` using
the same widget factory phase 7 writes for the acquisition form — so there is
one form generator in the codebase, not two. In phase 2 the panels are built
against a small local factory; phase 7 replaces that with the import of
`shell/param_form.py`. The TimeTagger panel additionally keeps its Test
Connection button and status label.

### Done when

- `tests/devices/test_registry.py`, `test_daq.py`, `test_galvo.py`,
  `test_time_tagger.py` pass, covering config defaults, `export_state` /
  `import_state` round-trip, and `backed_by` being declared.
- `tests/test_headless.py` still passes — importing `pyrpoc.devices` must not
  pull in QtWidgets.
- Still nothing in the running application imports `devices/`.

---

## Phase 3 — Datasets

Create `data/`. This is where the arrays stop living in widgets.

### `data/dataset.py`

```python
@dataclass(frozen=True)
class Provenance:
    program_key: str
    parameters: dict
    devices: dict[str, dict]
    started_at: str          # ISO UTC
    run_id: int

class Dataset:
    id: str
    run_id: int
    stream: str                      # "intensity", "histogram", ...
    spec: type[Stream]
    channel_labels: list[str]
    metadata: dict
    provenance: Provenance
    save: SavePolicy | None

    def append(self, array: np.ndarray, **coords) -> int
    def frame(self, index: int) -> np.ndarray
    def latest(self) -> np.ndarray | None
    def __len__(self) -> int
    def subscribe(self, cb) -> None / def unsubscribe(self, cb) -> None
```

`append` validates against `spec`, stores, calls `save.write(...)` if a policy
is attached, then notifies subscribers. Notification is a plain callback list,
not a Qt signal, because `data/` must stay importable headless.

Threading matters here and is easy to get wrong: `append` runs on the worker
thread, so subscriber callbacks do too. No view ever subscribes directly.
`shell/run_bridge.py` (phase 6) is the only subscriber, and it re-emits on a Qt
signal, which Qt queues onto the GUI thread. That reproduces exactly the
thread-safety today's `data_emitted` pyqtSignal provides.

A multi-frame run is one dataset that grows. Frame counting leaves the program
entirely — `get_frame_limit`, `_run_frame_limit`, `_saved_frame_count` and
`_frames_emitted` all disappear.

### `data/library.py`

```python
class DatasetLibrary:
    def add(self, dataset) -> None
    def remove(self, dataset_id) -> None
    def get(self, run_id, stream) -> Dataset | None
    def matching(self, spec: type[Stream]) -> list[Dataset]   # newest first
    def all(self) -> list[Dataset]
    def subscribe(self, cb) -> None       # library membership changes
```

Not persisted. Datasets are run outputs; on-disk backing is section 13's
out-of-scope list, so the library starts empty each launch — the same state
today's displays start in.

### `data/io.py`

One copy of what `modalities/*/storage.py` does three times. The on-disk layout
is preserved so existing analysis scripts keep working, with two named
exceptions recorded under "Things that change for the user".

```python
class SavePolicy:
    def __init__(self, root: Path, stream: str, spec: type[Stream])
    def prepare(self, provenance) -> None
    def write(self, dataset, array, frame_index: int) -> None
    def finalize(self, frame_count: int, error: Exception | None) -> None
```

Per contract:

| contract | file | when |
|---|---|---|
| `Image2D` | `<root>_<channel>.tiff`, appended, float32 | per frame |
| `Cube3D` | `<root>_<stream>.npz` — `frames` (F,H,W,B) float32, `parameters` | at finalize |
| `Samples4D` | `<root>_<stream>.npz` — `frames` (F,C,H,W,S) float32, `parameters`, `frame_indices` | at finalize |

Plus `<root>_meta.json`, rewritten after every frame as it is today, so run
progress remains readable from disk mid-run. Keys carried over unchanged:
`run_id`, `started`, `save_root_path`, `tiff_paths`, `frames_saved`,
`frame_limit`, `parameters`, `last_error`. Added: `program_key`, `streams`.
`modality_key` is written too, with the same value as `program_key` and a
comment marking it a v3.0 alias to remove in 3.2 — one line, and lab scripts
that read it keep working.

Details to carry over exactly: `save_path` is expanded, and a `.tif`/`.tiff`
suffix is stripped to give the root; the parent directory is created; existing
per-channel TIFFs are unlinked before the first frame; frames are written with
`tifffile.TiffWriter(str(path), append=True)`; channel labels fall back to
`channel_{i}` when the program did not supply them.

Note that split confocal's raw file lands at `<root>_raw_pixel_stream.npz` with
keys `frames` / `parameters` / `frame_indices` — byte-for-byte the same layout
as today. The first draft listed this as a breaking change; it is not. What
changes is architectural: the stream is declared in `emits` and travels the
normal path instead of being smuggled through `_pending_auxiliary`,
`append_auxiliary_payload` and `flush_auxiliary_payloads`, all of which are
deleted.

### `data/transforms.py`

```python
def normalize_channels(chw: np.ndarray) -> np.ndarray   # per-channel min/max to [0,1]
```

The `get_normalized_data_3d` body, currently duplicated in both displays. Only
`views/` uses it today, so by section 6.4's test it is the file in `data/` most
likely to be misfiled; section 5 places it here and programs are the intended
second caller. Worth revisiting in 3.2 if nothing outside `views/` has picked it
up.

### Done when

- `tests/data/test_dataset.py`: append validates against the contract, `latest`
  and `frame` return what was appended, subscribers fire once per append,
  unsubscribe stops them.
- `tests/data/test_io.py`: writing three `Image2D` frames produces the same
  files, with the same per-frame TIFF page count and the same `_meta.json` keys
  and values, as `modalities/confocal/storage.py` produces for the same input —
  compared directly against the old code, which is still present.
- `tests/data/test_io.py` also covers the `Cube3D` and `Samples4D` npz payloads
  and the save-disabled path writing nothing.
- `tests/data/test_transforms.py`: `normalize_channels` matches
  `Tiled2DDisplay.get_normalized_data_3d` on the same array.

The old-versus-new comparison is available in phases 3 through 5 precisely
because the old code is still in the tree. Use it — it is the strongest check
available in this build, and it disappears at phase 9.

---

## Phase 4 — The runner and the first program

Create `run/` and `programs/confocal.py`.

### `run/program.py`

```python
class Program:
    uses: list[type[Device]] = []
    params: type = None
    emits: dict[str, type[Stream]] = {}
    def run(self, ctx: "RunContext") -> None: ...
```

Four attributes and nothing else — section 12's last invariant, checked by
`tests/test_program_shape.py`, which walks `Program.__subclasses__()` and fails
on any class attribute outside that set. The registry key lives in
`programs/registry.py`, and the label and grouping live in `shell/catalog.py`,
which is why neither appears here.

```python
class RunContext:
    params: Any
    devices: dict[type[Device], Device]
    def publish(self, stream: str, data: np.ndarray, *, channels=None, **coords) -> None
    def status(self, text: str) -> None
    def frames(self, count: int | None = None) -> Iterator[int]
    def check_cancel(self) -> None            # raises Cancelled
    def sleep(self, seconds: float) -> None   # cancellable
```

One concrete class, written once, never subclassed.

`frames()` is where continuous acquisition lives:

```python
def frames(self, count=None):
    i = 0
    limit = None if self._continuous else count
    while limit is None or i < limit:
        self.check_cancel()
        yield i
        i += 1
```

The program body is identical whether the run is bounded or continuous; the
Continuous button sets a flag on the run, not on the parameters, so it does not
overwrite the user's stored `num_frames`. Checking cancellation at the top of
each iteration reproduces today's `while not should_stop()` exactly.

`sleep()` waits on the cancel event rather than blocking, so stopping during
FLIM's inter-frame settle is prompt.

`Cancelled` propagates out through the program's `run()`. That is the teardown
mechanism: FLIM's `finally: tagger.stop_flim_measurement(flim)` runs on a stop
because the stop *is* an exception. This replaces `FlimModality.stop()` having
to call `teardown_tagger()` defensively from outside.

One behaviour is preserved rather than improved: a stop requested during a
blocking `raster_scan` is not observed until that frame completes, because the
NI read is one blocking call for the whole frame. Today's behaviour is the same.
Making it interruptible needs the incremental read described in section 4, which
is out of scope.

### `run/claims.py`

```python
def resolve(uses: list[type[Device]],
            inventory: list[Device]) -> dict[type[Device], Device]
```

Walks `backed_by` so claiming the galvo claims its DAQ, and raises
`MissingDevice` naming everything absent. This is where today's
`validate_required_instruments` lands. The launcher calls it on every device
inventory change to grey out programs whose devices are missing, which is how
FLIM is gated today and now gates confocal too.

### `run/runner.py`

Pure Python, no Qt, so it is testable with no `QApplication`.

```python
class Runner:
    def __init__(self, library: DatasetLibrary)
    def start(self, program, params, inventory, *, continuous=False,
              on_status, on_dataset, on_finished, on_failed) -> threading.Thread
    def stop(self) -> None
    @property
    def is_running(self) -> bool
```

`start` does, in order: resolve claims; increment the run id; build one
`Dataset` per entry in `emits`, each with provenance (program key, parameters as
a dict, device configurations, UTC timestamp) and a `SavePolicy` derived from
`params.save` when saving is enabled; add them to the library; call
`on_dataset` for each so the shell can open and bind views before any data
arrives; build the `RunContext`; spawn the worker.

The worker calls `program.run(ctx)` and then, in a `finally`, finalizes every
save policy and reports. `Cancelled` is caught and reported as a clean stop, not
an error — matching what the Stop button does today.

This is the section 8.1 point: the program contains no saving and no frame
counting. `prepare_acquisition_storage`, `save_acquired_frame` and
`finalize_acquisition_storage` do not exist on a program.

Before considering `run/` finished, re-read section 13. The item that matters is
nested runs (`ctx.run_sub`), needed by z-stacks and mosaics; it is the one
omission that would change what the runner has to be. `frames()` returning an
iterator rather than the program owning `range()` is the accommodation made for
it here — an outer program can drive an inner one's iterator. No further work
now.

### `programs/confocal.py`

```python
@dataclass
class ConfocalParams:
    scan: ScanGroup = group(ScanGroup, "Scan")
    daq: DaqGroup = group(DaqGroup, "DAQ")
    modulation: ModulationGroup = group(ModulationGroup, "Modulation")
    save: SaveGroup = group(SaveGroup, "Save")
    num_frames: int = int_field("Frames", 1, minimum=1)


@program_registry.register("confocal")
class Confocal(Program):
    uses = [Galvo, DAQ]
    params = ConfocalParams
    emits = {"intensity": Image2D}

    def run(self, ctx):
        p, daq, galvo = ctx.params, ctx.devices[DAQ], ctx.devices[Galvo]
        ttl = mask_ttl(
            [(b, load_mask(b.path)) for b in p.modulation.masks],
            scan=p.scan,
            pixel_samples=raster.pixel_samples(p.scan.dwell_time_us, p.daq.sample_rate_hz),
            device_name=daq.config.device_name,
        )
        labels = [f"ai{n}" for n in daq.config.ai_channels]
        for i in ctx.frames(p.num_frames):
            ctx.status(f"frame {i + 1}")
            frame = raster_scan(**p.scan, **p.daq, **daq.config, **galvo.config, ttl=ttl)
            ctx.publish("intensity", frame, channels=labels)
```

The program loads the mask files, because `operations/` may not import `data/`
or do path work of its own (rule 3), and it does so once before the loop rather
than once per frame as `extract_mask_contexts` effectively does today.

`num_frames` is validated in one place now — the field's `minimum=1` — rather
than in both `get_frame_limit()` and the schema (survey open item 4).

### Done when

- `tests/run/test_runner.py`: a fake program publishing three frames produces a
  dataset of length three; `stop()` mid-run ends cleanly with no error reported;
  a program that raises reports through `on_failed` and still finalizes saving;
  `frames(None)` with `continuous=True` runs until stopped.
- `tests/run/test_claims.py`: `backed_by` propagation, and `MissingDevice`
  naming every absent device.
- `tests/run/test_context.py`: `sleep` returns early on cancel; `publish`
  rejects an array that fails its contract.
- `tests/programs/test_confocal.py`: with `raster_scan` monkeypatched, a
  3-frame run with saving enabled writes the same TIFF pages and `_meta.json`
  content as `ConfocalModality` does for the same parameters, including with
  masks enabled.
- `tests/test_program_shape.py` passes.

Two complete acquisition paths now exist. The application uses the old one.

---

## Phase 5 — The other two programs

### `programs/split_confocal.py`

```python
@dataclass
class SplitConfocalParams:
    scan / daq / split (SplitGroup) / modulation / save / num_frames

class SplitConfocal(Program):
    uses = [Galvo, DAQ]
    params = SplitConfocalParams
    emits = {"intensity": Image2D, "raw_pixel_stream": Samples4D}

    def run(self, ctx):
        ...
        ttl = split_mask_ttl(..., t0_samples=p.split.t0_samples)
        labels = [f"ai{n}_{w}" for n in daq.config.ai_channels for w in ("t0", "t2")]
        for i in ctx.frames(p.num_frames):
            split, raw = split_raster_scan(**p.scan, **p.daq, **daq.config,
                                           **galvo.config, ttl=ttl, **p.split)
            ctx.publish("intensity", split, channels=labels)
            ctx.publish("raw_pixel_stream", raw)
```

The `run()` body duplicates confocal's shape, and that is correct — section 8.3
is explicit that a shared base class with a mode flag is the trap `BaseModality`
fell into. Sharing happens in `operations/`, where the code is genuinely
identical.

`t0_samples >= 1` and `t1_samples >= 0` move from `from_dict`'s hand-written
checks to the field bounds.

### `programs/flim.py`

```python
@dataclass
class FlimParams:
    scan / daq / triggers (TriggerGroup) / histogram (HistogramGroup) / save / num_frames

class FLIM(Program):
    uses = [Galvo, DAQ, TimeTagger]
    params = FlimParams
    emits = {"intensity": Image2D, "histogram": Cube3D}

    def run(self, ctx):
        p = ctx.params
        tagger = ctx.devices[TimeTagger]
        total_x = p.scan.x_pixels + p.scan.extra_left + p.scan.extra_right
        tagger.create_tagger()
        tagger.configure_for_flim()
        flim = tagger.start_flim_measurement(
            n_pixels=total_x * p.scan.y_pixels,
            n_bins=p.histogram.histogram_bins,
            binwidth_ps=p.histogram.histogram_binwidth_ps,
        )
        try:
            for i in ctx.frames(p.num_frames):
                flim_scan(**p.scan, **p.daq, **daq.config, **galvo.config, **p.triggers)
                ctx.sleep(p.histogram.frame_settle_s)
                cube = read_flim_frame(flim, n_bins=..., y_pixels=..., total_x_pixels=total_x,
                                       extra_left=..., x_pixels=...)
                ctx.publish("histogram", cube)
                ctx.publish("intensity", flim_intensity(cube)[np.newaxis], channels=["intensity"])
        finally:
            tagger.stop_flim_measurement(flim)
```

Setup and teardown are outside the loop. Today `acquire_once` calls
`setup_tagger()` at the top and `teardown_tagger()` in a `finally` on **every
frame**, because `acquire_once` must be self-contained. A ten-frame run creates
and frees the TimeTagger ten times. This is the clearest demonstration of
settled statement 1, and the reason to check that the frames themselves are
unchanged while the timing between them is not.

The histogram dataset carries the metadata today's `FLIM_RAW_FRAME` carries:
`laser_period_ps` (still `int(round(1e6 / laser_frequency_mhz))`), `binwidth_ps`
and `n_bins`.

FLIM publishes the histogram before the intensity, so a subscriber that reads
both sees them consistent. Today the intensity is emitted first; the order is
not observable by anything.

### Done when

- `tests/programs/test_split_confocal.py`: with `split_raster_scan`
  monkeypatched, output frames and saved files match `SplitConfocalModality` for
  the same parameters, including the raw npz payload and keys.
- `tests/programs/test_flim.py`: with `flim_scan` and `read_flim_frame`
  monkeypatched and a fake tagger device, frames match `FlimModality`, and the
  fake records exactly one `create_tagger` / `start_flim_measurement` /
  `stop_flim_measurement` triple across a 3-frame run.
- `tests/programs/test_flim.py` also asserts the tagger is stopped when the run
  is cancelled mid-loop.

---

## Phase 6 — Connect the interface to the new runner

The only phase that can break the working application. Kept as small as
possible: the parameter form and the displays are left alone, and two temporary
pieces of code bridge them to the new backend. Both are deleted in phases 7 and
8. They are deliberate throwaway, and they exist so that this commit leaves an
application that runs.

### `shell/run_bridge.py`

A `QObject` wrapping `Runner`. It subscribes to every dataset the runner creates
and re-emits as Qt signals, which is the whole reason `run/` can stay Qt-free:

```python
class RunBridge(QObject):
    run_started = pyqtSignal()
    run_status = pyqtSignal(str)
    dataset_opened = pyqtSignal(object)
    dataset_changed = pyqtSignal(object, int)
    run_finished = pyqtSignal(int)
    run_failed = pyqtSignal(str)
```

Emitting from the worker thread is safe; Qt queues delivery to the GUI thread
because the receivers live there. This reproduces exactly what
`ModalityService.data_emitted` does today.

### What changes in the acquisition panel

`gui/main_widgets/acquisition_mgr/handlers.py` stops calling
`ModalityService.configure` / `.start` and calls `RunBridge` instead. Parameters
are still scraped from the form widgets by `collect_values`, then converted to a
params object at the hand-off point by a temporary adapter that maps today's
labels onto the new model. The Continuous button passes `continuous=True`.

Programs whose devices are missing are greyed out via `run/claims.resolve`,
replacing `validate_required_instruments`.

### Temporary display bridge

About sixty lines in `shell/`: subscribe to `dataset_changed`, and for each open
display whose `accepted_kinds` matches, call its existing `render(AcquiredData)`
with a synthesised `AcquiredData`. Its only job is to keep the two displays
working while acquisition changes underneath them. Deleted in phase 8.

### After this phase

`modalities/`, `services/modality_service.py` and
`services/acquisition_interpreter.py` are still present but nothing reaches
them. Their tests still pass, which is intentional — they remain the reference
implementation the phase 3–5 comparison tests measure against until phase 9.

### Done when

- Each of the three programs can be launched from the interface, drives the new
  runner, and renders into the existing displays.
- Stop, Continuous and the frame counter behave as before.
- With saving enabled, the files produced are the ones phases 3–5 pinned.
- `tests/shell/test_run_bridge.py`: signals arrive on the GUI thread; a run
  started and stopped through the bridge leaves `is_running` false.

---

## Phase 7 — Parameters stop living in widgets

The form is generated from the parameter model and writes back into it. This is
settled statement 3, and it is what lets anything other than the form
parameterise a run.

### `shell/param_form.py`

The Qt half of the old `parameter_utils.py`, as a factory table keyed by field
type:

```python
class FieldWidget(NamedTuple):
    widget: QWidget
    get: Callable[[], Any]
    set: Callable[[Any], None]
    connect: Callable[[Callable[[], None]], None]

WIDGET_FACTORIES: dict[type[Field], Callable[[Field, QWidget], FieldWidget]]

class ParamForm(QWidget):
    def __init__(self, params_obj, on_change)
    def read_into(self, params_obj) -> None    # form -> model, on every change
    def write_from(self, params_obj) -> None   # model -> form
```

Widgets carried over from `parameter_utils.py`: `QSpinBox` / `QDoubleSpinBox`
with the same bounds and six decimals, `QLineEdit`, the path row with its Browse
dialog, `QCheckBox`, `QComboBox`, and the AI channel toggle row from
`ChannelSelectionParameter` including its stylesheet.

New: the `MasksField` widget — the mask table from section 8.2, three columns
(file, port, line) with add/remove rows and a Browse button per row. It is the
replacement for `BaseOptoControl`, `BaseOptoControlWidget`, the optocontrol
registry, the manager panel, `prepare_for_acquisition`, `get_context`,
`MaskContext`, `extract_mask_contexts`, `allowed_optocontrols` and the
`optocontrols` list in `AppState`. Roughly 700 lines become about 150.

`read_into` runs on every widget change, so the model is authoritative at all
times rather than being scraped at play. `collect_values` and the phase 6
adapter are deleted.

The device panels switch from their local factory to this one, so there is a
single form generator.

### Sections

`core.params.sections()` drives the layout, so the shared groups appear once and
in declaration order. The card-per-section presentation with the collapsed
summary line is kept from `acquisition_mgr/forms.py`.

### `session/` at schema 7

```python
schema_version = 7

@dataclass
class SessionState:
    schema_version: int = 7
    theme_mode: str = "system"
    devices: list[DeviceSessionState]
    views: list[ViewSessionState]
    selected_program: str | None
    params_by_program: dict[str, dict]     # core.params.to_dict output
    ads_layout: str | None
```

`store.py` returns defaults on any version mismatch — no converter, per the
decision above. Old sessions reset once, and the reset is total: dock layout,
instruments, displays and parameters all return to defaults on first launch of
v3.1.

To keep that reset from producing an application with a dead play button, the
store seeds one `DAQ` (`Dev1`) and one `Galvo` (ao0/ao1) when the loaded session
contains no devices at all. That matches what confocal can do today with no
instruments configured, and it is the mitigation that makes the no-converter
decision cheap.

`params_by_program` replaces `params_by_modality`, keyed by program key, so
switching programs still preserves each one's settings.

### Done when

- `tests/shell/test_param_form.py`: every field type round-trips model → widget
  → model; a widget change updates the model without a play; the mask table adds,
  edits and removes rows.
- `tests/session/test_store.py`: save/load round-trip at schema 7; a schema 6
  file loads as defaults with no crash; an empty session seeds a DAQ and a
  Galvo.
- Switching between programs preserves each one's parameters, and the programs
  receive the same values they received in phase 6.

---

## Phase 8 — Displays become views

Delete the temporary bridge. Views read datasets instead of being handed frames.

### `views/base.py`

```python
class View(QWidget):
    renders: list[type[Stream]] = []
    def bind(self, dataset: Dataset | None) -> None
    def refresh(self) -> None
    def clear(self) -> None
    def export_state(self) -> dict / def import_state(self, raw) -> None
```

No view holds an array. `self._dataset` plus `dataset.latest()` on refresh
replaces `self._data_chw`, which today *is* the data — closing a display
destroys it and two displays over one run hold two drifting copies. That is
fixed by this change alone.

`accepted_kinds` becomes `renders` carrying shape contracts; `allowed_displays`
on the programs is deleted; `DataKind` is deleted.

### `views/image_2d.py` and `views/overlay.py`

`tiled_2d_display.py` and `multichan_overlay_display.py`, with `set_data`
replaced by a read from the bound dataset. Per-tile name, autoscale and LUT
levels, the persistence of those, the colour maps, the two-column reflow and the
overlay compositing are all unchanged.

Each view gains a source picker: a combo listing library datasets whose contract
it renders, defaulting to the newest. That is what makes "two displays can show
the same run" and "close and reopen without losing data" true, and it replaces
the implicit routing that today pushes the latest frame at every matching
display.

### `views/mask_editor.py`

`gui/main_widgets/opto_control_mgr/mask_editor.py` moved. It opens as a view
like any other, its source picker lists `Image2D` datasets, and it feeds
`transforms.normalize_channels(dataset.latest()) * 255` into the same threshold
and polygon-ROI machinery it uses now.

Save writes a mask file and is the only exit. The `create_mask_requested`
signal, the temp-file write and the push back into a control instance all go
away with the optocontrols — a mask is an authored preset referenced by a path
parameter, and the editor is a view with a save action (section 11, question 3).

### Refresh routing

`shell/window.py` connects `dataset_changed` to the bound views. A hidden view
is not refreshed while hidden and refreshes on show — today a hidden display
misses frames permanently, because the interpreter skips any display that is not
`attached and docked_visible` and there is nothing to catch up from. Reading
from the dataset fixes that as a side effect.

### Auto-opening

When a run starts, for each declared stream with no open compatible view, the
shell opens the default view for that contract (`Image2D` → `image_2d`).
`Cube3D` and `Samples4D` have no view, so nothing opens for FLIM's histogram or
split confocal's raw stream — both are still saved. That is section 8.1's
"an image view opens because the program declares it emits one", with the caveat
that the decay view section 8.4 assumes was deleted in phase 0.

### Library panel

`shell/library_panel.py` lists open datasets — program, stream, frame count,
save path — with a close action. It takes the panel slot the OptoControls
manager vacates, keeping the panel count at four: Acquisition, Devices, Views,
Library.

### Done when

- `tests/views/test_image_2d.py`, `test_overlay.py`: binding a dataset and
  appending renders; unbinding clears; two views bound to one dataset both
  render it; closing one leaves the data intact.
- `tests/views/test_mask_editor.py`: ROI + threshold produce the same mask array
  as today's editor for the same input, and Save writes a loadable file.
- The temporary bridge and `accepted_kinds` / `allowed_displays` / `DataKind`
  are gone.

---

## Phase 9 — Move folders and delete the old code

Rename `gui/` to `shell/` (`gui/styles/` → `shell/theme/`, keeping
`breeze_all.py` and `breeze_resources.py` byte-identical) and finish the
deletions:

```
modalities/            all of it
services/              all of it
backend_utils/         all of it
optocontrols/          all of it
displays/              all of it
instruments/           all of it
domain/                including stores.py (dead; referenced only by its own test)
persistence/           replaced by session/
```

`pyproject.toml`: version `3.1.0`, drop `pyqtdarktheme` (unused — theming runs
through the vendored Breeze module; survey open item 3).

Test tree moves to mirror the source tree. `tests/domain/test_stores.py` and the
`tests/modalities/` old-implementation tests are deleted along with what they
test; `tests/reference/` stays where it is and keeps passing.

### Done when

- The old folders are gone.
- `tests/test_import_rules.py` and `tests/test_headless.py` pass against the
  final tree.
- The full suite passes.
- `pyrpoc` launches, all three programs run, and saving produces the files
  phases 3–5 pinned.

---

# Things that change for the user

Worth reading before reviewing the diff, because several of these will look like
bugs if you are not expecting them.

1. **Devices replace instruments.** The DAQ and the galvo are now devices you add
   and configure, like the TimeTagger. A fresh session seeds one of each, so play
   works immediately; if you remove them, every program greys out.
2. **Wiring leaves the acquisition form.** `DAQ Device`, `Fast Axis AO`,
   `Slow Axis AO` and `Active AI Channels` move to the DAQ and Galvo panels.
   FLIM's tagger channel numbers, trigger voltages and laser input delay move to
   the TimeTagger panel. What is left in the form is what you change between
   experiments.
3. **The OptoControls panel is gone.** Masks are a table in the Modulation
   section of the acquisition form: file, port, line.
4. **Masks stop being globally toggleable.** You lose the enable/disable card.
   You gain that a run's saved metadata records exactly which masks were applied
   on which lines, which today it does not, because `enabled` lived outside the
   modality (section 8.2's stated tradeoff).
5. **The mask editor is a view.** It opens like a display and reads an acquired
   dataset rather than reaching into a display widget's array.
6. **Views have a source picker,** and closing a view no longer destroys its
   data.
7. **A Library panel** lists the runs currently open.
8. **The program selector shows labels,** not registry keys — "Split Confocal"
   rather than `split_confocal`.
9. **Saved sessions reset once.** Schema 7, no converter.
10. **FLIM is faster across frames.** The TimeTagger is created and freed once
    per run instead of once per frame. Image data should be unchanged; the gap
    between frames will not be.
11. **FLIM's histogram file is renamed and re-typed.** `<root>_raw.npz` becomes
    `<root>_histogram.npz`; `frames` becomes a real `(F, H, W, B)` float32 array
    instead of `dtype=object`; the `acquisition_parameters` key becomes
    `parameters`. Scripts reading the old file need updating; the new one is
    easier to read.
12. **The metadata `parameters` block is nested,** following the group structure,
    and gains `program_key` and `streams`. `modality_key` is still written with
    the same value, as a v3.0 alias to be removed in 3.2.
13. **Split confocal's raw file is unchanged** — `<root>_raw_pixel_stream.npz`,
    same keys, same shapes. Only how it gets there changed. (The first draft of
    this plan listed it as a breaking change; that was wrong.)
14. **The Continuous button stays.**

# Deviations from the design document

Each of these departs from `260827-refactor_plan.md`, with the reason:

| Deviation | Reason |
|---|---|
| No `devices/*/simulated.py`; section 10's "simulation is a device variant" is not implemented | Phase 0 removed simulation; reinstating it is explicitly out of scope for v3.1. |
| No `views/streamed.py`, `views/decay.py`; no `devices/stage/` | Nothing to port — deleted in phase 0. |
| `Samples4D` added; split confocal's raw stream is not `Image2D` as section 8.3 says | The array is 4-D. Filing it as `Image2D` would make the contract false. |
| No `Trace1D` | No emitter. Same reason `DataKind.PARTIAL_FRAME` was removed. |
| `active_ai_channels` on the DAQ device, not the galvo as section 9's table implies | Analog input is a property of the card, not the scanner. The table's actual requirement — one definition instead of three — is met. |
| Galvo has no per-axis limits despite section 4 listing them | Nothing would read them; an unused field is the accumulation section 3 warns about. |
| `run/` is Qt-free; `shell/run_bridge.py` does the thread marshalling | Section 12 requires the suite to run with no `QApplication`. A `QObject` runner would make every runner test need one. |
| `core/modulation.py` does file I/O | Section 5 places it there. `operations/` may not import `data/`, so the alternative puts mask loading somewhere worse. |
| The Library panel is new; it is not in section 5's tree | The dataset library needs a home in the UI, and the OptoControls panel slot is freed. |

# Carried-over items from the survey

- Three `DaqUnavailableError` classes → one `core/errors.DaqError` (phase 1).
- `domain/stores.py` deleted (phase 9).
- `pyqtdarktheme` dropped from `pyproject.toml` (phase 9).
- `num_frames` validated in one place, the field bound (phase 4).
- `split_confocal` used to swallow bare `RuntimeError` into simulated data. That
  is gone, so unrelated bugs that were surfacing as toy frames will now surface
  as errors. If something starts failing that used to appear to work, this is
  why.

# Protocol for the build

One commit per phase, message `phase N: <what>`, suite green at every commit.
Nine reviewable commits beat one large one, and the per-phase check makes a
later failure point at a specific phase rather than at the migration.

Through phase 5 nothing is at risk: the working application does not import the
new folders, and recovering from a mistake is deleting or fixing one folder.
From phase 6 onward, a phase that does not pass its check does not get committed.

Three rules for the build, in order of how expensive they are to get wrong:

1. **Do not regenerate `phase0_references.npz`.** If a reference test fails, the
   implementation changed the arithmetic. Fix the implementation.
2. **Do not tidy `run_raster`, `run_flim_scan`, or the TimeTagger calls while
   moving them.** Move them, then stop. The parts that look redundant — the
   lowercase `/port0/` test, the start ordering, the inverted static write on
   teardown, `n_frame_average=1` — are the parts that only fail on the
   instrument, and nothing in this build can catch a change to them.
3. **Use the old code as the oracle while it is still there.** Phases 3 through 5
   can compare new output against `modalities/*` directly for the same inputs.
   That comparison is the strongest check in this build and it is gone after
   phase 9.
