"""ctypes binding for the Andor SDK2 calls this application makes.

Andor ships no package on PyPI, so this is a hand-written binding over
``atmcd64d.dll``. It follows the same lazy-import shape as ``import nidaqmx``
in the scan code and ``from Swabian import TimeTagger`` in the tagger driver:
the DLL is located and loaded on first use, never at import time, so
``pyrpoc.devices`` stays importable on a machine with no SDK installed.

Only the calls the driver beside this one makes are declared, and every one of
them declares ``argtypes`` -- which ``declare_all`` applies rather than leaving
to ctypes. An undeclared ``float`` argument is promoted to a C double, and the
camera then receives a bit pattern that is not the number it was passed.
``SetExposureTime``, ``GetAcquisitionTimings``, ``GetTemperatureF``,
``GetHSSpeed``, ``GetVSSpeed`` and ``GetPreAmpGain`` are all 32-bit ``float``.

SDK2 keeps one current camera per process: ``SetCurrentCamera`` is global state
by the DLL own design. Serialising access is the caller job, and ``CCD`` holds
the lock that does it.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_int32, c_long, c_ulong
from pathlib import Path

import numpy as np

from pyrpoc.core.errors import CcdError

# --------------------------------------------------------------------------- #
# Return codes                                                                 #
# --------------------------------------------------------------------------- #

SUCCESS = 20002
ERROR_FILELOAD = 20006
NO_NEW_DATA = 20024

TEMPERATURE_OFF = 20034
TEMPERATURE_NOT_STABILIZED = 20035
TEMPERATURE_STABILIZED = 20036
TEMPERATURE_NOT_REACHED = 20037
TEMPERATURE_OUT_RANGE = 20038
TEMPERATURE_NOT_SUPPORTED = 20039
TEMPERATURE_DRIFT = 20040

#: Every code ``GetTemperatureF`` returns in normal operation. It reports the
#: temperature *status* as its return value and the reading in the out-param,
#: so passing it through a plain success check raises on a healthy camera.
TEMPERATURE_CODES = frozenset(
    {
        TEMPERATURE_OFF,
        TEMPERATURE_NOT_STABILIZED,
        TEMPERATURE_STABILIZED,
        TEMPERATURE_NOT_REACHED,
        TEMPERATURE_OUT_RANGE,
        TEMPERATURE_DRIFT,
    }
)

ACQUIRING = 20072
IDLE = 20073
TEMPCYCLE = 20074
NOT_INITIALIZED = 20075
NOT_SUPPORTED = 20991
NOT_AVAILABLE = 20992

#: What a head answers when it simply does not have the thing being set --
#: no internal shutter, no fan control, no baseline clamp. A capability the
#: camera lacks is a fact about the camera, not a failed run.
UNSUPPORTED_CODES = frozenset({NOT_SUPPORTED, NOT_AVAILABLE})

ERROR_NAMES: dict[int, str] = {
    20001: "DRV_ERROR_CODES",
    20002: "DRV_SUCCESS",
    20003: "DRV_VXDNOTINSTALLED",
    20004: "DRV_ERROR_SCAN",
    20005: "DRV_ERROR_CHECK_SUM",
    20006: "DRV_ERROR_FILELOAD",
    20007: "DRV_UNKNOWN_FUNCTION",
    20008: "DRV_ERROR_VXD_INIT",
    20009: "DRV_ERROR_ADDRESS",
    20010: "DRV_ERROR_PAGELOCK",
    20011: "DRV_ERROR_PAGEUNLOCK",
    20012: "DRV_ERROR_BOARDTEST",
    20013: "DRV_ERROR_ACK",
    20014: "DRV_ERROR_UP_FIFO",
    20015: "DRV_ERROR_PATTERN",
    20017: "DRV_ACQUISITION_ERRORS",
    20018: "DRV_ACQ_BUFFER",
    20019: "DRV_ACQ_DOWNFIFO_FULL",
    20020: "DRV_PROC_UNKNOWN_INSTRUCTION",
    20021: "DRV_ILLEGAL_OP_CODE",
    20022: "DRV_KINETIC_TIME_NOT_MET",
    20023: "DRV_ACCUM_TIME_NOT_MET",
    20024: "DRV_NO_NEW_DATA",
    20026: "DRV_SPOOLERROR",
    20034: "DRV_TEMPERATURE_OFF",
    20035: "DRV_TEMPERATURE_NOT_STABILIZED",
    20036: "DRV_TEMPERATURE_STABILIZED",
    20037: "DRV_TEMPERATURE_NOT_REACHED",
    20038: "DRV_TEMPERATURE_OUT_RANGE",
    20039: "DRV_TEMPERATURE_NOT_SUPPORTED",
    20040: "DRV_TEMPERATURE_DRIFT",
    20049: "DRV_GENERAL_ERRORS",
    20050: "DRV_INVALID_AUX",
    20051: "DRV_COF_NOTLOADED",
    20052: "DRV_FPGAPROG",
    20053: "DRV_FLEXERROR",
    20054: "DRV_GPIBERROR",
    20064: "DRV_DATATYPE",
    20065: "DRV_DRIVER_ERRORS",
    20066: "DRV_P1INVALID",
    20067: "DRV_P2INVALID",
    20068: "DRV_P3INVALID",
    20069: "DRV_P4INVALID",
    20070: "DRV_INIERROR",
    20071: "DRV_COFERROR",
    20072: "DRV_ACQUIRING",
    20073: "DRV_IDLE",
    20074: "DRV_TEMPCYCLE",
    20075: "DRV_NOT_INITIALIZED",
    20076: "DRV_P5INVALID",
    20077: "DRV_P6INVALID",
    20078: "DRV_INVALID_MODE",
    20079: "DRV_INVALID_FILTER",
    20080: "DRV_I2CERRORS",
    20081: "DRV_I2CDEVNOTFOUND",
    20082: "DRV_I2CTIMEOUT",
    20083: "DRV_P7INVALID",
    20089: "DRV_USBERROR",
    20090: "DRV_IOCERROR",
    20091: "DRV_VRMVERSIONERROR",
    20093: "DRV_USB_INTERRUPT_ENDPOINT_ERROR",
    20094: "DRV_RANDOM_TRACK_ERROR",
    20095: "DRV_INVALID_TRIGGER_MODE",
    20096: "DRV_LOAD_FIRMWARE_ERROR",
    20097: "DRV_DIVIDE_BY_ZERO_ERROR",
    20098: "DRV_INVALID_RINGEXPOSURES",
    20099: "DRV_BINNING_ERROR",
    20100: "DRV_INVALID_AMPLIFIER",
    20101: "DRV_INVALID_COUNTCONVERT_MODE",
    20115: "DRV_ERROR_MAP",
    20116: "DRV_ERROR_UNMAP",
    20117: "DRV_ERROR_MDL",
    20118: "DRV_ERROR_UNMDL",
    20119: "DRV_ERROR_BUFFSIZE",
    20121: "DRV_ERROR_NOHANDLE",
    20130: "DRV_GATING_NOT_AVAILABLE",
    20131: "DRV_FPGA_VOLTAGE_ERROR",
    20990: "DRV_ERROR_NOCAMERA",
    20991: "DRV_NOT_SUPPORTED",
    20992: "DRV_NOT_AVAILABLE",
}


def code_name(code: int) -> str:
    """The DRV_ symbol for a return code, or the number when it is unknown."""
    return ERROR_NAMES.get(int(code), f"DRV_{int(code)}")


# --------------------------------------------------------------------------- #
# Modes                                                                        #
# --------------------------------------------------------------------------- #

READ_MODE_FVB = 0
READ_MODE_MULTI_TRACK = 1
READ_MODE_RANDOM_TRACK = 2
READ_MODE_SINGLE_TRACK = 3
READ_MODE_IMAGE = 4

ACQ_MODE_SINGLE = 1
ACQ_MODE_ACCUMULATE = 2
ACQ_MODE_KINETIC = 3
ACQ_MODE_FAST_KINETIC = 4
ACQ_MODE_RUN_TILL_ABORT = 5

TRIGGER_INTERNAL = 0
TRIGGER_EXTERNAL = 1
TRIGGER_EXTERNAL_START = 6
TRIGGER_EXTERNAL_EXPOSURE = 7
TRIGGER_EXTERNAL_FVB_EM = 9
TRIGGER_SOFTWARE = 10

SHUTTER_AUTO = 0
SHUTTER_OPEN = 1
SHUTTER_CLOSED = 2

#: TTL level the shutter output drives to open. A property of how the shutter
#: is wired, which is why the driver takes it from configuration.
SHUTTER_TTL_LOW = 0
SHUTTER_TTL_HIGH = 1

#: Which output the charge is read through. Inverting these two produces data
#: that looks fine and has the wrong noise characteristics, so they are named.
AMP_EMCCD = 0
AMP_CONVENTIONAL = 1

FAN_FULL = 0
FAN_LOW = 1
FAN_OFF = 2

#: ``SetCoolerMode(1)`` keeps the sensor cold through ``ShutDown``. Without it,
#: quitting the application costs a full cooldown on the next launch.
COOLER_MAINTAIN_ON_SHUTDOWN = 1


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

DLL_NAME = "atmcd64d.dll"

#: The DLL and its ``Drivers`` folder, checked into the repo beside this
#: module rather than found on the machine -- see ``vendor/README.md``. This is
#: the primary location; it comes before the fallbacks below so a checked-in
#: SDK always wins over whatever else happens to be installed.
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

#: Where an Andor SOLIS or standalone SDK install normally lands, checked if
#: nothing was found in ``VENDOR_DIR``. Kept as a fallback for a machine that
#: has the vendor software installed but nothing checked into the repo yet.
DEFAULT_DIRS = (
    r"C:\Program Files\Andor SOLIS",
    r"C:\Program Files\Andor SDK",
    r"C:\Program Files\Andor Driver Pack 2",
    r"C:\Program Files (x86)\Andor SOLIS",
    r"C:\Program Files (x86)\Andor SDK",
)

#: Detector ``.ini`` and firmware live in a subfolder of the install, not
#: beside the DLL. ``Initialize`` wants that folder, and gets
#: DRV_ERROR_FILELOAD when given the wrong one.
DATA_SUBDIRS = ("Drivers", "")

_lib: ctypes.CDLL | None = None
_lib_dir: Path | None = None


def candidate_dirs() -> list[Path]:
    """Every directory that might hold the DLL, in priority order.

    ``PYRPOC_ANDOR_DLL`` stays as an escape hatch for a dev machine that wants
    to point at a different SDK without touching the repo, but there is no
    per-device config field any more -- the DLL location is a property of the
    install, not of a run.
    """
    out: list[Path] = []
    env = os.environ.get("PYRPOC_ANDOR_DLL", "").strip()
    if env:
        path = Path(env).expanduser()
        out.append(path.parent if path.suffix.lower() == ".dll" else path)
    out.append(VENDOR_DIR)
    out.extend(Path(name) for name in DEFAULT_DIRS)
    return out


def load() -> ctypes.CDLL:
    """Find and load ``atmcd64d.dll``, once per process.

    ``os.add_dll_directory`` is what makes this work rather than raising
    WinError 126 on a path that is correct: since Python 3.8 the DLL search no
    longer consults ``PATH``, and ``atmcd64d.dll`` loads dependents out of its
    own folder.
    """
    global _lib, _lib_dir
    if _lib is not None:
        return _lib

    tried: list[str] = []
    for folder in candidate_dirs():
        target = folder / DLL_NAME
        tried.append(str(target))
        if not target.is_file():
            continue
        try:
            with os.add_dll_directory(str(folder)):
                lib = ctypes.WinDLL(str(target))
        except OSError as exc:
            raise CcdError(f"found {target} but could not load it: {exc}") from exc
        declare_all(lib)
        _lib, _lib_dir = lib, folder
        return lib

    raise CcdError(
        f"could not find {DLL_NAME}. Expected it under {VENDOR_DIR} -- see "
        f"vendor/README.md -- or install Andor SOLIS or the Andor SDK, or set "
        f"the PYRPOC_ANDOR_DLL environment variable. Tried: " + ", ".join(tried)
    )


def library_dir() -> Path | None:
    """The folder the loaded DLL came from, or None before the first load."""
    return _lib_dir


def default_data_dir() -> str:
    """A guess at the folder ``Initialize`` wants, from where the DLL was found.

    Only a guess: installs differ, which is why the device carries an
    overridable field rather than relying on this.
    """
    folder = _lib_dir
    if folder is None:
        return ""
    for name in DATA_SUBDIRS:
        candidate = folder / name if name else folder
        if candidate.is_dir():
            return str(candidate)
    return str(folder)


# --------------------------------------------------------------------------- #
# Symbol table                                                                 #
# --------------------------------------------------------------------------- #

#: name -> argtypes. Every SDK2 call returns ``unsigned int``, so only the
#: arguments vary. A call taking no arguments still appears here with an empty
#: tuple, which is what makes "declared" and "callable" the same thing.
SIGNATURES: dict[str, tuple] = {
    # identity and lifecycle
    "Initialize": (c_char_p,),
    "ShutDown": (),
    "GetAvailableCameras": (POINTER(c_long),),
    "GetCameraHandle": (c_long, POINTER(c_long)),
    "SetCurrentCamera": (c_long,),
    "GetDetector": (POINTER(c_int), POINTER(c_int)),
    "GetHeadModel": (c_char_p,),
    "GetCameraSerialNumber": (POINTER(c_int),),
    # readout geometry and timing
    "SetAcquisitionMode": (c_int,),
    "SetReadMode": (c_int,),
    "SetTriggerMode": (c_int,),
    "SetExposureTime": (c_float,),
    "GetAcquisitionTimings": (POINTER(c_float), POINTER(c_float), POINTER(c_float)),
    "SetBaselineClamp": (c_int,),
    # amplifier, speeds and gain
    "SetOutputAmplifier": (c_int,),
    "GetNumberADChannels": (POINTER(c_int),),
    "SetADChannel": (c_int,),
    "GetNumberHSSpeeds": (c_int, c_int, POINTER(c_int)),
    "GetHSSpeed": (c_int, c_int, c_int, POINTER(c_float)),
    "SetHSSpeed": (c_int, c_int),
    "GetNumberVSSpeeds": (POINTER(c_int),),
    "GetVSSpeed": (c_int, POINTER(c_float)),
    "SetVSSpeed": (c_int,),
    "GetFastestRecommendedVSSpeed": (POINTER(c_int), POINTER(c_float)),
    "GetNumberPreAmpGains": (POINTER(c_int),),
    "GetPreAmpGain": (c_int, POINTER(c_float)),
    "SetPreAmpGain": (c_int,),
    "IsPreAmpGainAvailable": (c_int, c_int, c_int, c_int, POINTER(c_int)),
    "SetEMGainMode": (c_int,),
    "GetEMGainRange": (POINTER(c_int), POINTER(c_int)),
    "SetEMCCDGain": (c_int,),
    # shutter, cooling, fan
    "SetShutter": (c_int, c_int, c_int, c_int),
    "SetTemperature": (c_int,),
    "GetTemperatureF": (POINTER(c_float),),
    "CoolerON": (),
    "CoolerOFF": (),
    "SetCoolerMode": (c_int,),
    "SetFanMode": (c_int,),
    # acquisition
    "PrepareAcquisition": (),
    "StartAcquisition": (),
    "WaitForAcquisitionTimeOut": (c_int,),
    "CancelWait": (),
    "GetStatus": (POINTER(c_int),),
    "GetAcquiredData": (POINTER(c_int32), c_ulong),
    "AbortAcquisition": (),
}


def declare_all(lib: ctypes.CDLL) -> None:
    """Apply every signature in the table to the loaded library.

    A symbol the DLL does not export is skipped rather than fatal: SDK2
    versions differ in what they provide, and a call one head lacks should fail
    where it is used with its name attached, not at load time for every camera.
    """
    for name, argtypes in SIGNATURES.items():
        func = getattr(lib, name, None)
        if func is None:
            continue
        func.restype = c_ulong
        func.argtypes = list(argtypes)


def call(lib: ctypes.CDLL, name: str, *args) -> int:
    """Invoke a declared symbol and return its raw code.

    Refuses an undeclared symbol rather than letting ctypes guess the
    marshalling, which is the whole reason the table exists.
    """
    if name not in SIGNATURES:
        raise CcdError(f"{name} is not declared in sdk2.SIGNATURES")
    func = getattr(lib, name, None)
    if func is None:
        raise CcdError(f"{name} is not exported by this version of {DLL_NAME}")
    return int(func(*args))


def check(lib: ctypes.CDLL, name: str, *args, ok: frozenset[int] | None = None) -> int:
    """Invoke a symbol and raise unless the code is success or explicitly allowed.

    ``ok`` exists because part of this API reports state through the return
    value. ``GetTemperatureF`` returns the temperature status and
    ``WaitForAcquisitionTimeOut`` returns DRV_NO_NEW_DATA on a timeout; neither
    is a failure. ``GetStatus`` is the opposite case and needs no allowance --
    it returns success and puts the state in its out-param.
    """
    code = call(lib, name, *args)
    if code == SUCCESS or (ok is not None and code in ok):
        return code
    raise CcdError(f"{name} failed: {code_name(code)} ({code})")


# --------------------------------------------------------------------------- #
# Typed wrappers for the calls with out-params                                 #
# --------------------------------------------------------------------------- #


def available_cameras(lib: ctypes.CDLL) -> int:
    total = c_long(0)
    check(lib, "GetAvailableCameras", byref(total))
    return int(total.value)


def select_camera(lib: ctypes.CDLL, index: int) -> None:
    """Make camera ``index`` current. Must happen before ``Initialize``."""
    handle = c_long(0)
    check(lib, "GetCameraHandle", c_long(int(index)), byref(handle))
    check(lib, "SetCurrentCamera", handle)


def initialize(lib: ctypes.CDLL, data_dir: str) -> None:
    """Load the detector files and bring up the current camera.

    ``data_dir`` is the folder holding the detector ``.ini`` and firmware, not
    the folder holding the DLL. Takes several seconds on a Newton.
    """
    check(lib, "Initialize", c_char_p(str(data_dir).encode("mbcs", "replace")))


def detector_size(lib: ctypes.CDLL) -> tuple[int, int]:
    """``(width, height)`` in pixels. 1600 x 400 on a Newton 971."""
    width, height = c_int(0), c_int(0)
    check(lib, "GetDetector", byref(width), byref(height))
    return int(width.value), int(height.value)


def head_model(lib: ctypes.CDLL) -> str:
    buffer = ctypes.create_string_buffer(64)
    check(lib, "GetHeadModel", buffer)
    return buffer.value.decode("ascii", "replace").strip()


def serial_number(lib: ctypes.CDLL) -> int:
    """Which physical camera this is, which the head model cannot answer."""
    number = c_int(0)
    check(lib, "GetCameraSerialNumber", byref(number))
    return int(number.value)


def acquisition_timings(lib: ctypes.CDLL) -> tuple[float, float, float]:
    """``(exposure, accumulate_cycle, kinetic_cycle)`` in seconds, as accepted.

    The camera rounds the exposure it was asked for to what its clock can
    express. This is the number worth recording; the requested one is not.
    """
    exposure, accumulate, kinetic = c_float(0.0), c_float(0.0), c_float(0.0)
    check(
        lib, "GetAcquisitionTimings", byref(exposure), byref(accumulate), byref(kinetic)
    )
    return float(exposure.value), float(accumulate.value), float(kinetic.value)


def em_gain_range(lib: ctypes.CDLL) -> tuple[int, int]:
    """The valid EM gain range *for the current gain mode*.

    Reading it before ``SetEMGainMode`` validates against the wrong range,
    which is how a gain that passed a bounds check is rejected by the camera.
    """
    low, high = c_int(0), c_int(0)
    check(lib, "GetEMGainRange", byref(low), byref(high))
    return int(low.value), int(high.value)


def ad_channel_count(lib: ctypes.CDLL) -> int:
    count = c_int(0)
    check(lib, "GetNumberADChannels", byref(count))
    return int(count.value)


def hs_speeds(lib: ctypes.CDLL, ad_channel: int, amplifier: int) -> list[float]:
    """Horizontal shift speeds in MHz for one AD channel and output amplifier.

    The table is per amplifier, so an index valid on the EM output can be out
    of range on the conventional one.
    """
    count = c_int(0)
    check(
        lib,
        "GetNumberHSSpeeds",
        c_int(int(ad_channel)),
        c_int(int(amplifier)),
        byref(count),
    )
    speeds: list[float] = []
    for index in range(int(count.value)):
        speed = c_float(0.0)
        check(
            lib,
            "GetHSSpeed",
            c_int(int(ad_channel)),
            c_int(int(amplifier)),
            c_int(index),
            byref(speed),
        )
        speeds.append(float(speed.value))
    return speeds


def vs_speeds(lib: ctypes.CDLL) -> list[float]:
    """Vertical shift speeds in microseconds per row."""
    count = c_int(0)
    check(lib, "GetNumberVSSpeeds", byref(count))
    speeds: list[float] = []
    for index in range(int(count.value)):
        speed = c_float(0.0)
        check(lib, "GetVSSpeed", c_int(index), byref(speed))
        speeds.append(float(speed.value))
    return speeds


def fastest_vs_speed(lib: ctypes.CDLL) -> tuple[int, float]:
    """``(index, microseconds)`` the camera recommends for vertical shift.

    Preferred over a hand-chosen index: the recommendation comes from the head,
    so it stays right across a camera swap.
    """
    index, speed = c_int(0), c_float(0.0)
    check(lib, "GetFastestRecommendedVSSpeed", byref(index), byref(speed))
    return int(index.value), float(speed.value)


def preamp_gains(lib: ctypes.CDLL) -> list[float]:
    count = c_int(0)
    check(lib, "GetNumberPreAmpGains", byref(count))
    gains: list[float] = []
    for index in range(int(count.value)):
        gain = c_float(0.0)
        check(lib, "GetPreAmpGain", c_int(index), byref(gain))
        gains.append(float(gain.value))
    return gains


def preamp_available(
    lib: ctypes.CDLL,
    ad_channel: int,
    amplifier: int,
    hs_index: int,
    preamp_index: int,
) -> bool:
    """Whether a preamp gain can be used with this channel, amplifier and speed.

    A preamp index that exists is not necessarily usable at every readout
    speed, and this is the only way to know before the camera rejects it.
    """
    status_out = c_int(0)
    check(
        lib,
        "IsPreAmpGainAvailable",
        c_int(int(ad_channel)),
        c_int(int(amplifier)),
        c_int(int(hs_index)),
        c_int(int(preamp_index)),
        byref(status_out),
    )
    return bool(status_out.value)


def temperature(lib: ctypes.CDLL) -> tuple[float, int]:
    """``(celsius, status_code)``.

    The status is the *return* code, which is never DRV_SUCCESS while cooling
    is configured. Compare it against the ``TEMPERATURE_*`` constants; it is
    not an error.
    """
    reading = c_float(0.0)
    code = call(lib, "GetTemperatureF", byref(reading))
    if code != SUCCESS and code not in TEMPERATURE_CODES:
        raise CcdError(f"GetTemperatureF failed: {code_name(code)} ({code})")
    return float(reading.value), code


def status(lib: ctypes.CDLL) -> int:
    """``DRV_IDLE``, ``DRV_ACQUIRING`` or an acquisition error code.

    The state arrives in the out-param and the return is a plain success code,
    which is the reverse of most of this API and why it has its own wrapper.
    """
    state = c_int(0)
    check(lib, "GetStatus", byref(state))
    return int(state.value)


def acquired_data(lib: ctypes.CDLL, size: int) -> np.ndarray:
    """``size`` counts out of the camera, as int32.

    Andor ``at_32`` is a signed 32-bit integer, not a 16-bit word. Under full
    vertical binning ``size`` is the detector *width*: the rows have already
    been summed into one row on the chip, so asking for width times height
    overruns what the camera has to give.
    """
    count = int(size)
    if count <= 0:
        raise CcdError(f"cannot read {count} points off the detector")
    buffer = (c_int32 * count)()
    check(lib, "GetAcquiredData", buffer, c_ulong(count))
    return np.ctypeslib.as_array(buffer).astype(np.int32, copy=True)
