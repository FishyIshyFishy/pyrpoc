import numpy as np

from pyrpoc_next.acquisition import (
    ConfocalModality,
    MaskModifier,
    Runner,
    SimulatedFlimModality,
    SimulatedModality,
    SplitConfocalModality,
    modality_registry,
    modifier_registry,
)
from pyrpoc_next.instruments import SimulatedDAQ, SimulatedTagger
from pyrpoc_next.structs.keys import InstrumentKey, ModalityKey, ModifierKey
from pyrpoc_next.structs.parameters import coerce_parameter_values
from pyrpoc_next.structs.parcels import HistogramCubeParcel, ImageFrameParcel


def coerced(modality_cls, **overrides):
    return coerce_parameter_values(modality_cls.manifest.parameter_groups, overrides)


class StubDaq:
    """Returns a fixed sample cube; stands in for the NIDAQ in readout tests."""

    device_name = "Dev1"

    def __init__(self, cube):
        self.cube = cube

    def run(self, scan, ttl_signals=None):
        return self.cube


def collect(modality, frame_limit):
    parcels = []
    Runner().run_sync(modality, parcels.append, frame_limit=frame_limit)
    return parcels


def test_registry_has_all_modalities():
    assert set(modality_registry.available()) == {
        ModalityKey.confocal, ModalityKey.split_confocal, ModalityKey.flim,
        ModalityKey.simulated, ModalityKey.simulated_flim,
    }


def test_simulated_modality_emits_image_frames():
    modality = SimulatedModality()
    modality.configure(coerced(SimulatedModality, **{"X Pixels": 16, "Y Pixels": 8, "Active AI Channels": [0, 1]}),
                       {InstrumentKey.simulated_daq: SimulatedDAQ()}, [])
    parcels = collect(modality, frame_limit=3)
    assert len(parcels) == 3
    assert all(isinstance(p, ImageFrameParcel) for p in parcels)
    assert parcels[0].data.shape == (2, 8, 16)


def test_frame_limit_respected():
    modality = SimulatedModality()
    modality.configure(coerced(SimulatedModality, **{"X Pixels": 8, "Y Pixels": 8}),
                       {InstrumentKey.simulated_daq: SimulatedDAQ()}, [])
    assert len(collect(modality, frame_limit=1)) == 1


def test_simulated_mask_modifier_brightens_pixels():
    shape = {"X Pixels": 8, "Y Pixels": 8, "Active AI Channels": [0]}
    baseline = SimulatedModality()
    baseline.configure(coerced(SimulatedModality, **shape), {InstrumentKey.simulated_daq: SimulatedDAQ()}, [])
    plain = collect(baseline, frame_limit=1)[0].data

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    masked = SimulatedModality()
    masked.configure(coerced(SimulatedModality, **shape), {InstrumentKey.simulated_daq: SimulatedDAQ()},
                     [MaskModifier(mask=mask)])
    boosted = collect(masked, frame_limit=1)[0].data
    assert boosted[0, 4, 4] > plain[0, 4, 4]  # frame index 0 is deterministic for both


def test_simulated_flim_emits_intensity_and_histogram():
    modality = SimulatedFlimModality()
    modality.configure(coerced(SimulatedFlimModality, **{"X Pixels": 16, "Y Pixels": 8}),
                       {InstrumentKey.simulated_tagger: SimulatedTagger()}, [])
    parcels = collect(modality, frame_limit=1)
    kinds = [type(p) for p in parcels]
    assert ImageFrameParcel in kinds and HistogramCubeParcel in kinds
    cube = next(p for p in parcels if isinstance(p, HistogramCubeParcel))
    assert cube.data.shape == (8, 16, 125)


def test_confocal_readout_means_over_samples():
    cube = np.zeros((2, 4, 4, 3), dtype=np.float32)
    cube[:, :, :, :] = np.arange(3, dtype=np.float32)  # mean over samples = 1.0
    modality = ConfocalModality()
    modality.configure(coerced(ConfocalModality, **{"X Pixels": 8, "Y Pixels": 8, "Active AI Channels": [0, 1]}),
                       {InstrumentKey.ni_daq: StubDaq(cube)}, [])
    frame = modality.acquire_frame(0)[0]
    assert frame.data.shape == (2, 4, 4)
    assert np.allclose(frame.data, 1.0)
    assert frame.channel_labels == ["ai0", "ai1"]


def test_split_readout_produces_two_windows_per_channel():
    cube = np.zeros((1, 4, 4, 5), dtype=np.float32)
    cube[:, :, :, :2] = 1.0   # t0 window -> mean 1.0
    cube[:, :, :, 3:] = 4.0   # t2 window (after t0=2 + t1=1) -> mean 4.0
    modality = SplitConfocalModality()
    values = coerced(SplitConfocalModality, **{"X Pixels": 8, "Y Pixels": 8, "Active AI Channels": [0],
                                               "t0 Samples": 2, "t1 Samples": 1})
    modality.configure(values, {InstrumentKey.ni_daq: StubDaq(cube)}, [])
    frame = modality.acquire_frame(0)[0]
    assert frame.data.shape == (2, 4, 4)  # one channel -> t0 + t2
    assert np.allclose(frame.data[0], 1.0) and np.allclose(frame.data[1], 4.0)
    assert frame.channel_labels == ["ai0_t0", "ai0_t2"]


def test_flim_declares_no_mask_support():
    from pyrpoc_next.acquisition import FlimModality

    assert ModifierKey.mask not in FlimModality.manifest.realizable_modifiers


def test_modifier_registry_builds_mask_from_values():
    modifier = modifier_registry.build(ModifierKey.mask, {"DAQ Port": 1, "DAQ Line": 3})
    assert isinstance(modifier, MaskModifier)
    assert (modifier.daq_port, modifier.daq_line) == (1, 3)


def test_runner_threaded_start_and_stop():
    modality = SimulatedModality()
    modality.configure(coerced(SimulatedModality, **{"X Pixels": 8, "Y Pixels": 8}),
                       {InstrumentKey.simulated_daq: SimulatedDAQ()}, [])
    seen = []
    finished = []
    runner = Runner()
    runner.start(modality, seen.append, frame_limit=2, on_finished=finished.append)
    runner.thread.join(timeout=5)
    assert len(seen) == 2
    assert finished == [None]
