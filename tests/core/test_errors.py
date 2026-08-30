"""One exception tree, so `except DeviceError` catches every device failure."""

from __future__ import annotations

from pyrpoc.core.errors import (
    Cancelled,
    DaqError,
    DeviceError,
    MissingDevice,
    ParameterError,
    PyrpocError,
    TaggerError,
)


def test_device_errors_share_a_base():
    """v3.0 had three unrelated DaqUnavailableError classes; catching one missed the others."""
    assert issubclass(DaqError, DeviceError)
    assert issubclass(TaggerError, DeviceError)
    assert issubclass(MissingDevice, DeviceError)
    assert issubclass(DeviceError, PyrpocError)


def test_cancelled_and_parameter_error_are_not_device_errors():
    assert not issubclass(Cancelled, DeviceError)
    assert not issubclass(ParameterError, DeviceError)


def test_missing_device_lists_what_is_missing():
    exc = MissingDevice(["Galvo", "DAQ"])
    assert exc.missing == ["Galvo", "DAQ"]
    assert "Galvo" in str(exc) and "DAQ" in str(exc)
