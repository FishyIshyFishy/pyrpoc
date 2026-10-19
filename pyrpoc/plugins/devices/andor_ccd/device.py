"""The Andor CCD on the spectrograph: a Newton DU970P read as one spectrum
per acquisition.

The session stays open between runs because Initialize takes seconds and
cooling takes minutes; the app opens it at launch and closes it on exit, and
the sensor stays cold after that (see ``COOLER_MAINTAIN_ON_SHUTDOWN``).

The camera has no shutter of its own, and pyrpoc does not shutter: the
camera's shutter output is held open so nothing moves during acquisition.
The Kymera's shutter is in BNC mode, meant to follow that output, but on this
rig open, closed and auto all passed the same light, so the output does not
reach the shutter in this light path. Dark spectra are taken by blocking the
light.

TODO(kymera): take the Kymera's shutter out of BNC mode and drive it over USB
with ATSpectrographSetShutter, so closing it is a deliberate, verified action.

TODO(kymera): control the Kymera 328i (KY-4105) through
``Andor SDK/ATSpectrograph/x64/atspectrograph.dll``. Found on this rig: it
enumerates only after this DLL's Initialize, must be closed with
ATSpectrographClose before ShutDown, has four gratings (600 l/mm blazed 500
and 650 nm, 1200 l/mm 600 nm, 1800 l/mm 500 nm), and was at zero order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pyrpoc.structs.plugins import params as P
from pyrpoc.structs.plugins.devices import Device, device_registry

from .sdk import ACQUIRING, IDLE, AndorError, AndorSession, Temperature, code_name

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QWidget

ReadMode = Literal["fvb", "single_track"]
Amplifier = Literal["em", "conventional"]

READ_MODES: dict[ReadMode, int] = {"fvb": 0, "single_track": 3}
AMPLIFIERS: dict[Amplifier, int] = {"em": 0, "conventional": 1}
# SetShutter(TTL high, permanently open, no closing or opening delay).
SHUTTER_HELD_OPEN = (1, 1, 0, 0)
ACQ_SINGLE = 1
ACQ_ACCUMULATE = 2
TRIGGER_INTERNAL = 0

# The ADC is 16-bit, so a pixel at this value per read clipped.
ADC_MAX = 65535

# How often a running acquisition is checked, so Stop is felt promptly.
POLL_S = 0.02


@dataclass
class AndorCcdConfig(P.Group):
    dll_path: str = P.path_field(
        "SDK DLL",
        r"C:\Program Files\Andor SDK\atmcd64d.dll",
        dialog_filter="Andor SDK (atmcd64d.dll)",
        tooltip="atmcd64d.dll from the standalone Andor SDK, not the one in Andor SOLIS",
    )
    target_temp_c: int = P.int_field(
        "Target Temperature (°C)",
        -70,
        minimum=-120,
        maximum=-10,
        tooltip="Set at connect; Set Temperature applies a change",
    )


@dataclass(frozen=True)
class SpectrumReadout:
    """How to read one spectrum, as plain values so programs can build it."""

    exposure_s: float
    accumulations: int
    read_mode: ReadMode
    track_centre: int
    track_height: int
    amplifier: Amplifier
    em_gain: int
    readout_mhz: float
    preamp_gain: float


@dataclass(frozen=True)
class AcceptedTimings:
    """What the camera made of the request; the exposure is rounded to its clock."""

    exposure_s: float
    accumulate_cycle_s: float
    kinetic_cycle_s: float


def match_index(table: Sequence[float], value: float, what: str) -> int:
    """Index of ``value`` in a table the camera reported as float32."""
    for index, entry in enumerate(table):
        if math.isclose(entry, value, rel_tol=1e-3):
            return index
    raise AndorError(f"this camera has no {what} {value:g}; it offers {[f'{v:g}' for v in table]}")


def check_track(centre: int, height: int, rows: int) -> None:
    """The track must fit on the sensor; the SDK would only say which
    argument it disliked."""
    top = centre + height // 2
    bottom = top - height + 1
    if bottom < 1 or top > rows:
        raise AndorError(
            f"a {height}-row track centred on row {centre} runs off the {rows}-row sensor"
        )


def clipped(counts: np.ndarray, accumulations: int) -> bool:
    """Whether any pixel hit the ADC ceiling in every accumulated read."""
    return int(counts.max()) >= ADC_MAX * accumulations


@device_registry.register("andor_ccd")
class AndorCCD(Device):
    display_name = "Andor CCD"
    owns_connection = True
    holds_session = True
    config_cls = AndorCcdConfig

    config: AndorCcdConfig

    def __init__(self, instance_id: str | None = None, user_label: str | None = None):
        super().__init__(instance_id=instance_id, user_label=user_label)
        self.session: AndorSession | None = None

    def summary(self) -> str:
        if self.session is None:
            return "not connected"
        info = self.session.info
        return f"{info.head_model} #{info.serial} - connected"

    @property
    def session_open(self) -> bool:
        return self.session is not None

    def open_session(self) -> None:
        if self.session is not None:
            raise AndorError(f"{self.name} is already connected")
        self.session = AndorSession.connect(self.config.dll_path, self.config.target_temp_c)

    def close_session(self) -> None:
        # Dropped first, so a camera that faults on shutdown is still closed.
        session = self.require_session()
        self.session = None
        session.close()

    def check_reachable(self) -> bool:
        self.require_session().temperature()
        return True

    def require_session(self) -> AndorSession:
        if self.session is None:
            raise AndorError(f"{self.name} is not connected; connect it in the Devices panel")
        return self.session

    def temperature(self) -> Temperature:
        return self.require_session().temperature()

    def apply_target_temperature(self) -> None:
        self.require_session().check("SetTemperature", self.config.target_temp_c)

    def set_cooler(self, on: bool) -> None:
        self.require_session().check("CoolerON" if on else "CoolerOFF")

    def configure(self, readout: SpectrumReadout) -> AcceptedTimings:
        """Program the camera for ``readout``; the returned timings are what
        it will actually do."""
        session = self.require_session()
        self.apply_acquisition(session, readout)
        self.apply_amplifier(session, readout)
        # Every run, so a shutter mode left behind by other software is undone.
        session.check("SetShutter", *SHUTTER_HELD_OPEN)
        session.check("SetExposureTime", readout.exposure_s)
        exposure, accumulate, kinetic = session.floats("GetAcquisitionTimings", count=3)
        return AcceptedTimings(exposure, accumulate, kinetic)

    def apply_acquisition(self, session: AndorSession, readout: SpectrumReadout) -> None:
        if readout.accumulations == 1:
            session.check("SetAcquisitionMode", ACQ_SINGLE)
        else:
            session.check("SetAcquisitionMode", ACQ_ACCUMULATE)
            session.check("SetNumberAccumulations", readout.accumulations)
            # Zero asks for the shortest cycle the exposure allows.
            session.check("SetAccumulationCycleTime", 0.0)
        session.check("SetTriggerMode", TRIGGER_INTERNAL)
        session.check("SetReadMode", READ_MODES[readout.read_mode])
        if readout.read_mode == "single_track":
            check_track(readout.track_centre, readout.track_height, session.info.height)
            session.check("SetSingleTrack", readout.track_centre, readout.track_height)

    def apply_amplifier(self, session: AndorSession, readout: SpectrumReadout) -> None:
        info = session.info
        amplifier = AMPLIFIERS[readout.amplifier]
        hs_index = match_index(info.hs_mhz[amplifier], readout.readout_mhz, "readout rate (MHz)")
        preamp_index = match_index(info.preamp_gains, readout.preamp_gain, "preamp gain")
        if not session.preamp_available(amplifier, hs_index, preamp_index):
            raise AndorError(
                f"preamp gain {readout.preamp_gain:g}x is not available at "
                f"{readout.readout_mhz:g} MHz on the {readout.amplifier} amplifier"
            )
        session.check("SetADChannel", 0)
        session.check("SetOutputAmplifier", amplifier)
        session.check("SetHSSpeed", amplifier, hs_index)
        session.check("SetVSSpeed", info.vs_index)
        session.check("SetPreAmpGain", preamp_index)
        if readout.amplifier == "em":
            low, high = info.em_gain_range
            if not low <= readout.em_gain <= high:
                raise AndorError(f"EM gain must be within {low}..{high}, not {readout.em_gain}")
            session.check("SetEMCCDGain", readout.em_gain)

    def acquire(self, sleep: Callable[[float], None]) -> np.ndarray:
        """One spectrum of ``width`` int32 counts, as configured. A program
        passes ``ctx.sleep`` so Stop interrupts the wait; it should then call
        ``abort()``."""
        session = self.require_session()
        session.check("StartAcquisition")
        state = session.status()
        while state == ACQUIRING:
            sleep(POLL_S)
            state = session.status()
        if state != IDLE:
            raise AndorError(f"acquisition failed: {code_name(state)}")
        return session.acquired_data(session.info.width)

    def abort(self) -> None:
        session = self.require_session()
        if session.status() == ACQUIRING:
            session.check("AbortAcquisition")

    def metadata(self) -> dict[str, Any]:
        info = asdict(self.require_session().info)
        info["temperature"] = asdict(self.temperature())
        return info

    def panel(self, parent: QWidget, on_change: Callable[[], None]) -> QWidget:
        from .panel import AndorCcdPanel

        return AndorCcdPanel(self, parent, on_change)
