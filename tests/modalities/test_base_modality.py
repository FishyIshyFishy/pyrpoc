from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pyrpoc.backend_utils.acquired_data import AcquiredData, DataKind
from pyrpoc.modalities.base_modality import AcquisitionParameters, BaseModality


@dataclass(frozen=True)
class FakeParams(AcquisitionParameters):
    """Only the shared acquisition fields are needed for these tests."""


class FakeModality(BaseModality):
    modality_key = "fake"

    def __init__(self, frames_per_call: int = 1):
        super().__init__()
        self.frames_per_call = frames_per_call
        self.stop_calls = 0
        self.raise_on_acquire = False

    def load_params(self, params: dict) -> None:
        self.parameters = FakeParams(
            save_enabled=bool(params.get("save_enabled", False)),
            save_path=str(params.get("save_path", "acq")),
            num_frames=int(params.get("num_frames", 1)),
        )

    def load_instruments(self, instruments: dict) -> None:
        pass

    def acquire_once(self, on_data) -> None:
        if self.raise_on_acquire:
            raise RuntimeError("boom")
        for _ in range(self.frames_per_call):
            on_data(AcquiredData(data=np.zeros((2, 3), dtype=np.float32), kind=DataKind.INTENSITY_FRAME))

    def stop(self) -> None:
        self.stop_calls += 1
        self._running = False


# --------------------------------------------------------------------------- #
# shared frame utilities
# --------------------------------------------------------------------------- #

def test_split_channels_2d_returns_single():
    mod = FakeModality()
    out = mod.split_channels(np.zeros((4, 5), dtype=np.float32))
    assert len(out) == 1
    assert out[0].shape == (4, 5)


def test_split_channels_3d_splits_per_channel():
    mod = FakeModality()
    out = mod.split_channels(np.zeros((3, 4, 5), dtype=np.float32))
    assert len(out) == 3
    assert all(c.shape == (4, 5) for c in out)


def test_split_channels_rejects_bad_ndim():
    mod = FakeModality()
    with pytest.raises(ValueError):
        mod.split_channels(np.zeros((5,), dtype=np.float32))


def test_resolve_channel_labels_default():
    mod = FakeModality()
    assert mod.resolve_channel_labels(2) == ["channel_0", "channel_1"]


def test_parameters_as_dict_empty_when_unconfigured():
    assert FakeModality().parameters_as_dict() == {}


def test_parameters_as_dict_serializes_dataclass():
    mod = FakeModality()
    mod.load_params({"num_frames": 4})
    assert mod.parameters_as_dict() == {"save_enabled": False, "save_path": "acq", "num_frames": 4}


# --------------------------------------------------------------------------- #
# get_frame_limit
# --------------------------------------------------------------------------- #

def test_get_frame_limit_returns_num_frames():
    mod = FakeModality()
    mod.load_params({"num_frames": 5})
    assert mod.get_frame_limit() == 5


def test_get_frame_limit_unconfigured_raises():
    with pytest.raises(ValueError):
        FakeModality().get_frame_limit()


def test_get_frame_limit_rejects_zero():
    mod = FakeModality()
    mod.load_params({"num_frames": 0})
    with pytest.raises(ValueError):
        mod.get_frame_limit()


# --------------------------------------------------------------------------- #
# configure template + savepath
# --------------------------------------------------------------------------- #

def test_configure_runs_substeps_and_marks_configured():
    mod = FakeModality()
    mod.configure({"num_frames": 2}, {}, [])
    assert mod._configured is True
    assert isinstance(mod.parameters, FakeParams)


def test_load_savepath_disabled():
    mod = FakeModality()
    mod.load_params({"save_enabled": False})
    mod.load_savepath()
    assert mod._save_root_path is None


def test_load_savepath_strips_tiff_suffix():
    mod = FakeModality()
    mod.load_params({"save_enabled": True, "save_path": "out/acq.tiff"})
    mod.load_savepath()
    assert mod._save_root_path is not None
    assert mod._save_root_path.suffix == ""
    assert mod._save_root_path.name == "acq"


# --------------------------------------------------------------------------- #
# acquire_continuous worker loop
# --------------------------------------------------------------------------- #

def run_continuous(mod: FakeModality, *, frame_limit, should_stop):
    frames: list[AcquiredData] = []
    finished: list[tuple[int, Exception | None]] = []
    errors: list[Exception] = []
    thread = mod.acquire_continuous(
        on_frame=frames.append,
        frame_limit=frame_limit,
        should_stop=should_stop,
        on_error=errors.append,
        on_finished=lambda count, err: finished.append((count, err)),
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    return frames, finished, errors


def test_acquire_continuous_respects_frame_limit():
    mod = FakeModality(frames_per_call=1)
    frames, finished, errors = run_continuous(mod, frame_limit=3, should_stop=lambda: False)
    assert len(frames) == 3
    assert finished == [(3, None)]
    assert errors == []
    assert mod.stop_calls == 1


def test_acquire_continuous_stops_immediately_when_requested():
    mod = FakeModality()
    frames, finished, _ = run_continuous(mod, frame_limit=None, should_stop=lambda: True)
    assert frames == []
    assert finished == [(0, None)]


def test_acquire_continuous_reports_errors():
    mod = FakeModality()
    mod.raise_on_acquire = True
    frames, finished, errors = run_continuous(mod, frame_limit=5, should_stop=lambda: False)
    assert frames == []
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    count, err = finished[0]
    assert isinstance(err, RuntimeError)
