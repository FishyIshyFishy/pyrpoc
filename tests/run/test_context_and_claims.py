"""RunContext and device claims."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pyrpoc.core.errors import Cancelled, MissingDevice
from pyrpoc.core.streams import Image2D
from pyrpoc.data.dataset import Dataset, Provenance
from pyrpoc.devices import DAQ, Galvo
from pyrpoc.devices.base import Device
from pyrpoc.run import claims
from pyrpoc.run.program import Program, RunContext


def make_ctx(**kwargs) -> RunContext:
    dataset = Dataset(stream="intensity", spec=Image2D, provenance=Provenance("fake"))
    return RunContext(
        params=kwargs.pop("params", None),
        devices=kwargs.pop("devices", {}),
        datasets=kwargs.pop("datasets", {"intensity": dataset}),
        cancel=kwargs.pop("cancel", threading.Event()),
        **kwargs,
    )


# --- publish ---------------------------------------------------------------


def test_publish_appends_to_the_named_dataset():
    ctx = make_ctx()
    assert ctx.publish("intensity", np.zeros((1, 2, 2), np.float32)) == 0
    assert len(ctx.datasets["intensity"]) == 1


def test_publish_names_the_declared_streams_when_the_name_is_wrong():
    ctx = make_ctx()
    with pytest.raises(KeyError, match="intensity"):
        ctx.publish("histogram", np.zeros((1, 2, 2), np.float32))


def test_publish_rejects_an_array_that_fails_its_contract():
    ctx = make_ctx()
    with pytest.raises(ValueError, match="3 dimensions"):
        ctx.publish("intensity", np.zeros((2, 2), np.float32))


def test_publish_sets_channel_labels_once():
    ctx = make_ctx()
    ctx.publish("intensity", np.zeros((2, 2, 2), np.float32), channels=["ai0", "ai1"])
    ctx.publish("intensity", np.zeros((2, 2, 2), np.float32), channels=["x", "y"])
    assert ctx.datasets["intensity"].channel_labels == ["ai0", "ai1"]


def test_publish_passes_coordinates_through():
    ctx = make_ctx()
    ctx.publish("intensity", np.zeros((1, 2, 2), np.float32), z=3.0)
    assert ctx.datasets["intensity"].coords(0) == {"z": 3.0}


def test_status_is_optional():
    make_ctx().status("nothing listening")


def test_status_reaches_its_callback():
    seen = []
    make_ctx(on_status=seen.append).status("frame 1")
    assert seen == ["frame 1"]


# --- frames ----------------------------------------------------------------


def test_frames_yields_a_bounded_range():
    assert list(make_ctx().frames(3)) == [0, 1, 2]


def test_frames_is_unbounded_in_continuous_mode():
    ctx = make_ctx(continuous=True)
    out = []
    for index in ctx.frames(2):
        out.append(index)
        if index == 5:
            break
    assert out == [0, 1, 2, 3, 4, 5]


def test_frames_checks_for_cancellation_before_each_iteration():
    cancel = threading.Event()
    ctx = make_ctx(cancel=cancel)
    seen = []
    with pytest.raises(Cancelled):
        for index in ctx.frames(10):
            seen.append(index)
            if index == 1:
                cancel.set()
    assert seen == [0, 1]


def test_frames_raises_immediately_if_already_cancelled():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        list(make_ctx(cancel=cancel).frames(3))


def test_frames_none_without_continuous_is_also_unbounded():
    """An outer program can drive an inner loop; section 13's nested-runs hook."""
    ctx = make_ctx()
    out = [index for index, _ in zip(ctx.frames(None), range(4))]
    assert out == [0, 1, 2, 3]


# --- cancel and sleep ------------------------------------------------------


def test_check_cancel_raises_only_once_set():
    cancel = threading.Event()
    ctx = make_ctx(cancel=cancel)
    ctx.check_cancel()
    assert ctx.cancelled() is False
    cancel.set()
    assert ctx.cancelled() is True
    with pytest.raises(Cancelled):
        ctx.check_cancel()


def test_sleep_returns_early_when_cancelled():
    cancel = threading.Event()
    ctx = make_ctx(cancel=cancel)
    threading.Timer(0.05, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(Cancelled):
        ctx.sleep(5.0)
    assert time.monotonic() - started < 2.0, "sleep blocked instead of waking on cancel"


def test_sleep_waits_when_not_cancelled():
    started = time.monotonic()
    make_ctx().sleep(0.05)
    assert time.monotonic() - started >= 0.04


def test_zero_sleep_still_checks_cancellation():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        make_ctx(cancel=cancel).sleep(0)


# --- claims ----------------------------------------------------------------


class Standalone(Device):
    display_name = "Standalone"


def test_expand_follows_backed_by():
    assert claims.expand([Galvo]) == [Galvo, DAQ]


def test_expand_keeps_declaration_order_without_duplicates():
    assert claims.expand([Galvo, DAQ]) == [Galvo, DAQ]
    assert claims.expand([DAQ, Galvo]) == [DAQ, Galvo]


def test_expand_leaves_an_unbacked_device_alone():
    assert claims.expand([Standalone]) == [Standalone]


def test_resolve_binds_every_class_to_an_instance():
    daq, galvo = DAQ(), Galvo()
    bound = claims.resolve([Galvo], [daq, galvo])
    assert bound == {Galvo: galvo, DAQ: daq}


def test_resolve_raises_naming_everything_absent():
    with pytest.raises(MissingDevice) as excinfo:
        claims.resolve([Galvo], [])
    assert set(excinfo.value.missing) == {"Galvo", "NI-DAQ"}


def test_missing_lists_classes_not_instances():
    assert claims.missing([Galvo], [DAQ()]) == [Galvo]
    assert claims.missing([Galvo], [DAQ(), Galvo()]) == []


def test_resolve_takes_the_first_matching_instance():
    first, second = DAQ(), DAQ()
    assert claims.resolve([DAQ], [first, second])[DAQ] is first


def test_a_program_needing_nothing_resolves_to_nothing():
    assert claims.resolve([], []) == {}


# --- program shape ---------------------------------------------------------


def test_program_base_declares_only_the_four_allowed_attributes():
    assert set(Program.uses) == set()
    assert Program.params is None
    assert Program.emits == {}
    assert callable(Program.run)
