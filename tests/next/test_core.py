import numpy as np
import tifffile

from pyrpoc_next.core import AppState, Controller, FrameStorage, check_routine, route_parcel
from pyrpoc_next.instruments import SimulatedDAQ
from pyrpoc_next.structs.keys import DisplayKey, InstrumentKey, ModalityKey, ModifierKey
from pyrpoc_next.structs.manifest import DisplayManifest
from pyrpoc_next.structs.parameters import ParameterValue
from pyrpoc_next.structs.parcels import (
    HistogramCubeParcel,
    ImageFrameParcel,
    PartialImageParcel,
)
from pyrpoc_next.structs.routine import ModifierSlot, Routine, RoutineBlock


class CollectingDisplay:
    """A display sink that records what it is asked to render."""

    def __init__(self, accepts):
        self.manifest = DisplayManifest(key=DisplayKey.streamed, display_name="test", accepted_parcels=accepts)
        self.received = []

    def render(self, parcel):
        self.received.append(parcel)


def values(**kwargs):
    return [ParameterValue(label, value) for label, value in kwargs.items()]


def simulated_routine(frames=2, modifiers=None):
    block = RoutineBlock(
        modality=ModalityKey.simulated,
        values=values(**{"X Pixels": 8, "Y Pixels": 8, "Active AI Channels": [0], "Frames": frames}),
        modifiers=modifiers or [],
    )
    return Routine(blocks=[block], active_index=0)


# --- routing -----------------------------------------------------------------

def test_route_parcel_respects_accepted_types():
    image_display = CollectingDisplay((ImageFrameParcel,))
    histogram_display = CollectingDisplay((HistogramCubeParcel,))
    frame = ImageFrameParcel(data=np.zeros((1, 4, 4), dtype=np.float32), channel_labels=["a"])
    route_parcel(frame, [image_display, histogram_display])
    assert image_display.received == [frame]
    assert histogram_display.received == []


# --- storage -----------------------------------------------------------------

def test_storage_writes_tiff_pages_and_npz(tmp_path):
    storage = FrameStorage()
    storage.begin(tmp_path / "run")
    for value in (0.0, 1.0):
        storage.save(ImageFrameParcel(data=np.full((2, 4, 4), value, dtype=np.float32), channel_labels=["a", "b"]))
    storage.save(HistogramCubeParcel(data=np.zeros((4, 4, 8), dtype=np.float32), bin_width_ps=100.0, laser_period_ps=12500.0))
    storage.finish()
    assert (tmp_path / "run_a.tiff").exists() and (tmp_path / "run_b.tiff").exists()
    assert (tmp_path / "run_histograms.npz").exists()
    assert tifffile.imread(str(tmp_path / "run_a.tiff")).shape == (2, 4, 4)  # two frames -> two pages


def test_storage_ignores_partial_frames(tmp_path):
    storage = FrameStorage()
    storage.begin(tmp_path / "run")
    storage.save(PartialImageParcel(data=np.zeros((1, 4, 4), dtype=np.float32), channel_labels=["a"]))
    storage.finish()
    assert list(tmp_path.glob("*.tiff")) == []


# --- compatibility -----------------------------------------------------------

def test_check_flags_missing_instrument():
    state = AppState(displays=[CollectingDisplay((ImageFrameParcel,))], routine=simulated_routine())
    report = check_routine(state)  # no SimulatedDAQ added
    assert report.blocked
    assert any("instrument" in issue.message for issue in report.issues)


def test_check_flags_no_display_for_data():
    state = AppState(instruments=[SimulatedDAQ()], routine=simulated_routine())
    report = check_routine(state)
    assert report.blocked


def test_check_flags_unrealizable_modifier():
    block = RoutineBlock(
        modality=ModalityKey.flim,
        modifiers=[ModifierSlot(key=ModifierKey.mask, available=True, enabled=True)],
    )
    state = AppState(displays=[CollectingDisplay((ImageFrameParcel, HistogramCubeParcel))],
                     routine=Routine(blocks=[block]))
    report = check_routine(state)
    assert any("modifier" in issue.message for issue in report.issues)


def test_check_passes_for_valid_simulated_setup():
    state = AppState(
        instruments=[SimulatedDAQ()],
        displays=[CollectingDisplay((ImageFrameParcel,))],
        routine=simulated_routine(),
    )
    assert not check_routine(state).blocked


# --- controller end-to-end (the data-flow smoke test) ------------------------

def test_controller_play_runs_simulated_and_delivers_frames():
    display = CollectingDisplay((ImageFrameParcel,))
    state = AppState(instruments=[SimulatedDAQ()], displays=[display], routine=simulated_routine(frames=3))
    controller = Controller(state)
    events = []
    controller.on_started = lambda: events.append("started")
    controller.on_stopped = lambda: events.append("stopped")

    report = controller.play()
    assert not report.blocked
    controller.runner.thread.join(timeout=5)

    assert len(display.received) == 3
    assert all(isinstance(p, ImageFrameParcel) for p in display.received)
    assert events == ["started", "stopped"]
    assert state.run_status.value == "idle"


def test_controller_play_blocks_and_does_not_start_when_incompatible():
    state = AppState(instruments=[SimulatedDAQ()], routine=simulated_routine())  # no display
    controller = Controller(state)
    report = controller.play()
    assert report.blocked
    assert not controller.runner.is_running()


def test_controller_prepare_modifier_hook_attaches_runtime_data():
    mask = np.ones((8, 8), dtype=np.uint8)
    slot = ModifierSlot(key=ModifierKey.mask, available=True, enabled=True,
                        values=values(**{"DAQ Port": 0, "DAQ Line": 0}))
    display = CollectingDisplay((ImageFrameParcel,))
    state = AppState(instruments=[SimulatedDAQ()], displays=[display],
                     routine=simulated_routine(frames=1, modifiers=[slot]))
    controller = Controller(state)
    controller.prepare_modifier = lambda modifier, slot: setattr(modifier, "mask", mask)

    report = controller.play()
    assert not report.blocked
    controller.runner.thread.join(timeout=5)
    assert len(display.received) == 1  # ran to completion with the mask applied
