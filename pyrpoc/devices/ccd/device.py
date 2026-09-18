"""The Andor Newton EMCCD: its SDK handle, its readout wiring, and one spectrum.

The split against the run parameters is the one the DAQ and the TimeTagger
draw. What lives here is how the head is fitted and characterised -- which
camera, which output amplifier, what readout rate it was characterised at, how
the shutter is wired, where the cooler is held. Exposure, EM gain and how many
spectra to take change from sample to sample and live in the program parameter
block.

Unlike the TimeTagger, whose handle exists only inside a FLIM run, this holds
its handle across runs. That is not a preference: ``Initialize`` takes seconds
on a Newton and the sensor takes minutes to reach its setpoint, while Pinpoint
Raman starts a fresh run on every picked point. Opening per run would put a
multi-second pause on every click and give consecutive spectra from one sample
different dark current. So the camera is opened once, released through
``Device.close()``, and ``SetCoolerMode`` keeps it cold even through that.

The spectral axis is the pixel index. Turning pixels into wavelengths is a
calibration done outside this application, which is also what ``Spectrum1D``
says about itself.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from pyrpoc.core import params as P
from pyrpoc.core.errors import CcdError

from ..base import Device
from ..registry import device_registry
from . import sdk2

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QWidget


#: Config strings to the numbers the SDK wants. Kept beside the fields that
#: produce them rather than inline at the call, so a new choice is one edit.
AMPLIFIERS = {"em": sdk2.AMP_EMCCD, "conventional": sdk2.AMP_CONVENTIONAL}
FAN_MODES = {"full": sdk2.FAN_FULL, "low": sdk2.FAN_LOW, "off": sdk2.FAN_OFF}
SHUTTER_MODES = {
    "auto": sdk2.SHUTTER_AUTO,
    "open": sdk2.SHUTTER_OPEN,
    "closed": sdk2.SHUTTER_CLOSED,
}
SHUTTER_TTL = {"high": sdk2.SHUTTER_TTL_HIGH, "low": sdk2.SHUTTER_TTL_LOW}

#: Full well of the 16-bit readout. A full-vertical-binning spectrum sums 400
#: rows into one output node before any EM gain is applied, so clipping is easy
#: to reach and invisible in a plot that has autoscaled to it.
FULL_SCALE_COUNTS = 65535


@dataclass
class CcdConfig(P.Group):
    camera_index: int = P.int_field(
        "Camera Index", 0, minimum=0, tooltip="Which head to open, 0 for the first"
    )
    output_amplifier: str = P.choice_field(
        "Output Amplifier",
        "conventional",
        choices=("conventional", "em"),
        tooltip="Conventional for low light with long exposures, EM for single "
        "photon sensitivity. Changing this changes which readout rates exist",
    )
    hs_speed_mhz: float = P.float_field(
        "Readout Rate (MHz)",
        1.0,
        minimum=0.0,
        step=0.1,
        tooltip="Horizontal shift rate. The nearest rate the head offers on the "
        "selected amplifier is used, and the chosen one is recorded per run",
    )
    preamp_gain_index: int = P.int_field(
        "Pre-Amp Gain Index",
        0,
        minimum=0,
        tooltip="Index into the head pre-amp gain table. Test Connection lists "
        "what the table holds",
    )
    target_temperature_c: int = P.int_field(
        "Cooler Target (C)",
        -60,
        minimum=-100,
        maximum=25,
        tooltip="Sensor setpoint. Reaching it takes minutes, so the camera is "
        "held open between runs rather than re-cooled for each one",
    )
    cool_on_connect: bool = P.bool_field(
        "Cool On Connect",
        True,
        tooltip="Start cooling as soon as the camera opens",
    )
    fan_mode: str = P.choice_field(
        "Fan",
        "full",
        choices=("full", "low", "off"),
        tooltip="Off is quieter and warmer; it needs water cooling on some heads",
    )
    shutter_ttl: str = P.choice_field(
        "Shutter TTL",
        "high",
        choices=("high", "low"),
        tooltip="TTL level that opens the shutter, as it is wired",
    )
    shutter_open_ms: int = P.int_field(
        "Shutter Open (ms)",
        30,
        minimum=0,
        tooltip="Time the shutter needs to open. Too short clips the exposure",
    )
    shutter_close_ms: int = P.int_field(
        "Shutter Close (ms)",
        30,
        minimum=0,
        tooltip="Time the shutter needs to close",
    )
    baseline_clamp: bool = P.bool_field(
        "Baseline Clamp",
        True,
        tooltip="Hold the bias level steady between readouts. Off makes the "
        "first spectrum after an idle period read high",
    )
    readout_margin_s: float = P.float_field(
        "Readout Margin (s)",
        10.0,
        minimum=1.0,
        tooltip="How long past the exposure to wait for data before giving up",
    )


@device_registry.register("ccd")
class CCD(Device):
    display_name = "Andor Newton CCD"
    owns_connection = True
    config_cls = CcdConfig

    config: CcdConfig

    def __init__(self, instance_id: str | None = None, user_label: str | None = None):
        super().__init__(instance_id=instance_id, user_label=user_label)
        self._lib = None
        self._opened = False
        #: Every SDK call is taken under this. SDK2 keeps one current camera
        #: per process, the DLL is not thread safe, and the calls arrive from
        #: at least two threads: a run executes on a fresh worker thread each
        #: time, while the panel polls temperature from the GUI thread.
        self._lock = threading.RLock()

        self.width = 0
        self.height = 0
        self.head_model = ""
        self.serial_number = 0

        #: Read back after configuring, not copied from what was requested.
        self.last_exposure_s = 0.0
        self.last_em_gain = 0
        self.last_hs_speed_mhz = 0.0
        self.last_vs_speed_us = 0.0
        self.last_preamp_gain = 0.0
        self.last_shutter_mode = "auto"
        self.last_amplifier = "conventional"
        self.last_max_counts = 0
        #: Most recent temperature reading, so the panel can show something
        #: without calling into the SDK while an acquisition holds the lock.
        self.last_temperature: tuple[float, int] | None = None

    # -- identity ---------------------------------------------------------- #

    @property
    def is_open(self) -> bool:
        return self._opened

    def summary(self) -> str:
        """One line that does not claim a cold camera when nothing is open.

        ``last_test_ok`` is restored from the session file, so after a relaunch
        it says OK for a camera that is powered off. Connection state is read
        from this process, which is why it is reported first.
        """
        if self._opened:
            reading = self.last_temperature
            where = f"{self.head_model or 'connected'}"
            if reading is not None:
                where += f", {reading[0]:.1f}C {self.temperature_status(reading[1])}"
            return where
        if self.last_test_ok is None:
            return "Not connected"
        return "Not connected (last test OK)" if self.last_test_ok else "Not connected (last test FAILED)"

    @staticmethod
    def temperature_status(code: int) -> str:
        return {
            sdk2.TEMPERATURE_OFF: "cooler off",
            sdk2.TEMPERATURE_NOT_REACHED: "cooling",
            sdk2.TEMPERATURE_NOT_STABILIZED: "not stabilised",
            sdk2.TEMPERATURE_STABILIZED: "stabilised",
            sdk2.TEMPERATURE_DRIFT: "drifting",
            sdk2.TEMPERATURE_OUT_RANGE: "setpoint out of range",
        }.get(int(code), sdk2.code_name(code))

    # -- connection -------------------------------------------------------- #

    def check_reachable(self) -> bool:
        """Open the camera and read its identity, leaving it open.

        Deliberately not open-then-close. On this head connecting *is* the slow
        part, and closing again would throw away the cooling it just started.
        So Test Connection is Connect; letting go is ``close()``.
        """
        self.open()
        return self._opened

    def open(self) -> None:
        """Bring up the camera, once. Safe to call when it is already open."""
        with self._lock:
            if self._opened:
                return

            lib = sdk2.load()

            # Camera selection happens before Initialize, which binds whichever
            # camera is current. Skipped for the first head, where there is
            # nothing to select and some single-camera installs have no handle
            # to hand out yet.
            if int(self.config.camera_index) > 0:
                sdk2.select_camera(lib, int(self.config.camera_index))

            sdk2.initialize(lib, sdk2.default_data_dir())

            self._lib = lib
            self._opened = True

            try:
                self.width, self.height = sdk2.detector_size(lib)
                self.head_model = sdk2.head_model(lib)
                self.serial_number = sdk2.serial_number(lib)
                self.apply_thermal_settings()
            except Exception:
                # A camera that came up but could not be interrogated is not a
                # camera we can use, and leaving the SDK initialised would stop
                # the next attempt from getting as far as this one.
                self.close()
                raise

    def close(self) -> None:
        """Release the camera. Safe twice, and safe if it was never opened.

        ``AbortAcquisition`` comes first because ``ShutDown`` can hang on a
        camera that is still exposing. The cooler is left running: the mode set
        at open tells the head to hold its temperature through shutdown, so
        quitting does not cost a cooldown next launch.
        """
        with self._lock:
            lib, self._lib, self._opened = self._lib, None, False
            if lib is None:
                return
            for name, args in (("AbortAcquisition", ()), ("ShutDown", ())):
                try:
                    sdk2.call(lib, name, *args)
                except Exception:
                    pass
            self.last_temperature = None

    def library(self):
        """The loaded DLL, or a readable error naming what has to happen first."""
        if self._lib is None or not self._opened:
            raise CcdError(f"{self.name} is not connected; open it first")
        return self._lib

    # -- cooling ------------------------------------------------------------ #

    def apply_thermal_settings(self) -> None:
        """Fan, setpoint and cooler, as configured.

        ``SetCoolerMode`` is what keeps holding the handle open from being the
        only way to stay cold: the head maintains its temperature when the SDK
        shuts down. Fan control is absent on some heads, so it is allowed to
        decline.
        """
        lib = self.library()
        with self._lock:
            sdk2.check(
                lib,
                "SetFanMode",
                FAN_MODES[self.config.fan_mode],
                ok=sdk2.UNSUPPORTED_CODES,
            )
            sdk2.check(lib, "SetCoolerMode", sdk2.COOLER_MAINTAIN_ON_SHUTDOWN)
            sdk2.check(lib, "SetTemperature", int(self.config.target_temperature_c))
            if self.config.cool_on_connect:
                sdk2.check(lib, "CoolerON")

    def set_cooler(self, on: bool) -> None:
        """Turn cooling on or off without waiting for it to take effect.

        Reaching a setpoint takes minutes, so this returns as soon as the head
        has been told. Whoever wants to know where it got to polls
        ``temperature()``.
        """
        lib = self.library()
        with self._lock:
            if on:
                sdk2.check(lib, "SetTemperature", int(self.config.target_temperature_c))
                sdk2.check(lib, "CoolerON")
            else:
                sdk2.check(lib, "CoolerOFF")

    def temperature(self, *, cached_if_busy: bool = True) -> tuple[float, int] | None:
        """``(celsius, status_code)``, or None when the camera is not open.

        ``cached_if_busy`` returns the last reading rather than waiting when an
        acquisition holds the lock. That is what the panel wants: a temperature
        readout should never stall the interface behind a ten second exposure,
        and a value a few seconds old is the right answer about a quantity that
        moves over minutes.
        """
        if not self._opened or self._lib is None:
            return None

        blocking = not cached_if_busy
        if not self._lock.acquire(blocking=blocking):
            return self.last_temperature
        try:
            if self._lib is None:
                return None
            self.last_temperature = sdk2.temperature(self._lib)
            return self.last_temperature
        finally:
            self._lock.release()

    # -- what the head can do ----------------------------------------------- #

    def speed_tables(self) -> dict[str, list[float]]:
        """What this head offers, for a panel that has to explain an index.

        The horizontal table is read for the amplifier currently configured,
        because that is the one whose indices mean anything right now.
        """
        lib = self.library()
        amplifier = AMPLIFIERS[self.config.output_amplifier]
        with self._lock:
            return {
                "hs_speeds_mhz": sdk2.hs_speeds(lib, 0, amplifier),
                "vs_speeds_us": sdk2.vs_speeds(lib),
                "preamp_gains": sdk2.preamp_gains(lib),
            }

    # -- configuring one readout -------------------------------------------- #

    def configure_for_spectrum(
        self,
        *,
        exposure_s: float,
        em_gain: int = 0,
        shutter_mode: str = "auto",
    ) -> None:
        """Set the camera up for single full-vertical-binning spectra.

        Plain arguments rather than a parameter block, because ``devices/`` sits
        below ``programs/`` and must not import one -- the same reason
        ``TimeTagger.start_flim_measurement`` takes three integers.

        The order below is load bearing:

        * ``SetOutputAmplifier`` before ``SetHSSpeed``, because the shift-speed
          table is per amplifier. Reversed, a valid-looking index selects from
          the wrong table and the data comes back with the wrong noise.
        * ``SetEMGainMode`` before ``GetEMGainRange``, because the range
          reported is the range for the current mode.
        * ``GetAcquisitionTimings`` last, once everything that affects timing is
          set, so the exposure read back is the one the camera will really use.

        No ``SetImage``: full vertical binning ignores it. The whole sensor
        height collapses to one row, so a readout is ``width`` numbers.
        """
        lib = self.library()
        amplifier = AMPLIFIERS[self.config.output_amplifier]
        with self._lock:
            sdk2.check(lib, "SetAcquisitionMode", sdk2.ACQ_MODE_SINGLE)
            sdk2.check(lib, "SetReadMode", sdk2.READ_MODE_FVB)
            sdk2.check(lib, "SetTriggerMode", sdk2.TRIGGER_INTERNAL)

            self.apply_shutter(shutter_mode)
            self.last_shutter_mode = shutter_mode

            sdk2.check(lib, "SetOutputAmplifier", amplifier)
            sdk2.check(lib, "SetADChannel", 0)
            hs_index = self.apply_hs_speed(amplifier)
            self.apply_vs_speed()
            self.apply_preamp_gain(amplifier, hs_index)
            self.apply_em_gain(amplifier, em_gain)
            self.last_amplifier = self.config.output_amplifier

            if self.config.baseline_clamp:
                sdk2.check(
                    lib, "SetBaselineClamp", 1, ok=sdk2.UNSUPPORTED_CODES
                )

            sdk2.check(lib, "SetExposureTime", float(exposure_s))
            self.last_exposure_s = sdk2.acquisition_timings(lib)[0]

            # Front-loads the setup the first StartAcquisition would otherwise
            # pay for. Any setting change above invalidates it, which is why it
            # is here and not in read_spectrum.
            sdk2.check(lib, "PrepareAcquisition")

    def apply_shutter(self, mode: str) -> None:
        """Open, close or automate the internal shutter.

        The camera powers up with the shutter closed, so a configuration that
        skips this returns spectra of nothing -- which reads as an exposure
        problem and is not one. Heads without a shutter decline, and that is
        not a failure.
        """
        lib = self.library()
        sdk2.check(
            lib,
            "SetShutter",
            SHUTTER_TTL[self.config.shutter_ttl],
            SHUTTER_MODES.get(mode, sdk2.SHUTTER_AUTO),
            int(self.config.shutter_close_ms),
            int(self.config.shutter_open_ms),
            ok=sdk2.UNSUPPORTED_CODES,
        )

    def apply_hs_speed(self, amplifier: int) -> int:
        """Pick the offered rate nearest the configured one; return its index.

        A rate in MHz rather than a stored index, because an index means
        nothing without the head to look it up in: the table differs per
        amplifier and per camera, so a saved index survives a camera swap while
        meaning something different afterwards. The index is returned because
        pre-amp availability is asked against the speed actually selected.
        """
        lib = self.library()
        speeds = sdk2.hs_speeds(lib, 0, amplifier)
        if not speeds:
            raise CcdError("the camera reports no horizontal shift speeds")
        target = float(self.config.hs_speed_mhz)
        index = min(range(len(speeds)), key=lambda i: abs(speeds[i] - target))
        sdk2.check(lib, "SetHSSpeed", amplifier, index)
        self.last_hs_speed_mhz = speeds[index]
        return index

    def apply_vs_speed(self) -> None:
        """Use the vertical shift speed the head recommends.

        Not a configuration field: the recommendation comes from the camera, so
        it stays correct across a swap, and there is nothing a person would
        reliably choose better.
        """
        lib = self.library()
        index, microseconds = sdk2.fastest_vs_speed(lib)
        sdk2.check(lib, "SetVSSpeed", index)
        self.last_vs_speed_us = microseconds

    def apply_preamp_gain(self, amplifier: int, hs_index: int) -> None:
        """Set the pre-amp gain, refusing one this readout cannot use.

        A gain that exists in the table is not necessarily available at every
        amplifier and shift speed, and the camera answers that separately from
        rejecting it -- so it is worth asking rather than finding out from a
        DRV_P1INVALID with no explanation attached.
        """
        lib = self.library()
        gains = sdk2.preamp_gains(lib)
        if not gains:
            return

        index = int(self.config.preamp_gain_index)
        offered = ", ".join(f"{gain:g}x" for gain in gains)
        if index >= len(gains):
            raise CcdError(
                f"pre-amp gain index {index} is out of range; this head offers "
                f"{len(gains)}: {offered}"
            )
        if not sdk2.preamp_available(lib, 0, amplifier, hs_index, index):
            raise CcdError(
                f"pre-amp gain {gains[index]:g}x (index {index}) is not available "
                f"at {self.last_hs_speed_mhz:g} MHz on the "
                f"{self.config.output_amplifier} output. This head offers: {offered}"
            )
        sdk2.check(lib, "SetPreAmpGain", index)
        self.last_preamp_gain = gains[index]

    def apply_em_gain(self, amplifier: int, em_gain: int) -> None:
        """Set EM gain, clamped to what the current gain mode allows.

        Only meaningful on the electron-multiplying output; on the
        conventional one there is no multiplication register to drive, so this
        does nothing and records a gain of one.
        """
        lib = self.library()
        if amplifier != sdk2.AMP_EMCCD:
            self.last_em_gain = 1
            return
        sdk2.check(lib, "SetEMGainMode", 0)
        low, high = sdk2.em_gain_range(lib)
        gain = max(low, min(int(em_gain), high))
        sdk2.check(lib, "SetEMCCDGain", gain)
        self.last_em_gain = gain

    # -- acquiring ----------------------------------------------------------- #

    def read_spectrum(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
        poll_s: float = 0.05,
    ) -> np.ndarray | None:
        """One ``(1, width)`` float32 spectrum, or None if asked to stop.

        Polls ``GetStatus`` rather than blocking in ``WaitForAcquisition``,
        because a blocking wait cannot be interrupted from the thread that is
        inside it -- and a run is stopped from a different thread. With a ten
        second exposure a blocking wait would make the Stop button do nothing
        for ten seconds; polling lets it take effect within ``poll_s``.

        ``should_stop`` is a plain callable rather than a run context: a device
        has no business knowing what a run is. Programs pass ``ctx.cancelled``.

        The lock is held for the whole exposure, so anything else that wants
        the camera waits for it. That is the intent -- a temperature poll must
        not interleave with a readout -- but it also means pressing Disconnect
        mid-exposure does not take effect until the exposure ends.
        """
        lib = self.library()
        timeout_s = self.last_exposure_s + float(self.config.readout_margin_s)
        deadline = time.monotonic() + timeout_s

        with self._lock:
            sdk2.check(lib, "StartAcquisition")
            try:
                while True:
                    if should_stop is not None and should_stop():
                        sdk2.call(lib, "AbortAcquisition")
                        return None
                    state = sdk2.status(lib)
                    if state == sdk2.IDLE:
                        break
                    if state != sdk2.ACQUIRING:
                        raise CcdError(
                            f"acquisition failed: {sdk2.code_name(state)} ({state})"
                        )
                    if time.monotonic() > deadline:
                        raise CcdError(
                            f"the camera produced no data within {timeout_s:g}s "
                            f"({self.last_exposure_s:.3f}s exposure plus a "
                            f"{self.config.readout_margin_s:g}s margin)"
                        )
                    time.sleep(poll_s)
                counts = sdk2.acquired_data(lib, self.width)
            except Exception:
                # Covers the timeout above as well: a camera left mid-exposure
                # refuses the next StartAcquisition.
                sdk2.call(lib, "AbortAcquisition")
                raise

        self.last_max_counts = int(counts.max()) if counts.size else 0
        return counts.astype(np.float32, copy=False)[np.newaxis]

    def abort(self) -> None:
        """Stop any acquisition and close the shutter, keeping the handle.

        What a run does on its way out. Distinct from ``close()``: the camera
        stays initialised and stays cold, so the next run starts immediately
        and its dark current matches this one. Tolerant of being called when
        nothing is running, which is the usual case -- a run that finished
        normally is already idle.
        """
        with self._lock:
            if self._lib is None or not self._opened:
                return
            try:
                sdk2.call(self._lib, "AbortAcquisition")
                self.apply_shutter("closed")
            except Exception:
                pass

    @property
    def saturated(self) -> bool:
        """Whether the last readout reached the top of the 16-bit range.

        Worth asking after every spectrum: binning four hundred rows into one
        output node and then amplifying reaches full scale easily, and a plot
        that has autoscaled to a clipped peak looks entirely reasonable.
        """
        return self.last_max_counts >= int(0.98 * FULL_SCALE_COUNTS)

    def instrument_metadata(self) -> dict[str, object]:
        """What the camera actually did, for the dataset that came out of it.

        Read-backs only. The configuration is already written into every run
        metadata file by the runner, so repeating it here would be two copies
        of one thing with no way to tell which was true.
        """
        reading = self.last_temperature
        return {
            "head_model": self.head_model,
            "serial_number": self.serial_number,
            "detector_width": self.width,
            "detector_height": self.height,
            "read_mode": "full_vertical_binning",
            "exposure_s": self.last_exposure_s,
            "output_amplifier": self.last_amplifier,
            "em_gain": self.last_em_gain,
            "hs_speed_mhz": self.last_hs_speed_mhz,
            "vs_speed_us": self.last_vs_speed_us,
            "preamp_gain": self.last_preamp_gain,
            "shutter_mode": self.last_shutter_mode,
            "sensor_temperature_c": None if reading is None else reading[0],
            "temperature_status": None if reading is None else self.temperature_status(reading[1]),
        }

    # -- panel --------------------------------------------------------------- #

    def panel(self, parent: "QWidget | None" = None, on_change=None) -> "QWidget | None":
        from .panel import CcdPanel

        return CcdPanel(self, parent=parent, on_change=on_change)
