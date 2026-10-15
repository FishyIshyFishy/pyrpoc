"""The Andor SDK2 over ctypes: one DLL per process, one camera per session.

The DLL comes from the standalone Andor SDK install, never from Andor SOLIS:
SOLIS ships a file with the same name that only dispatches to SOLIS's own
libraries, and loaded alone it finds no cameras. The SDK is not
redistributable, so it is not bundled with pyrpoc.
"""

from __future__ import annotations

import ctypes
import functools
import os
import threading
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_int32, c_long, c_ulong
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyrpoc.structs.plugins.devices import DeviceError

SUCCESS = 20002
ACQUIRING = 20072
IDLE = 20073

# GetTemperatureF reports the cooler's state as its return code, not success.
TEMPERATURE_STATUS = {
    20034: "cooler off",
    20035: "not stabilized",
    20036: "stabilized",
    20037: "not reached",
    20038: "out of range",
    20040: "drifting",
    ACQUIRING: "acquiring",
}

# Keeps the sensor cold through ShutDown, so quitting does not cost a cooldown.
COOLER_MAINTAIN_ON_SHUTDOWN = 1

CODE_NAMES = {
    20003: "DRV_VXDNOTINSTALLED",
    20004: "DRV_ERROR_SCAN",
    20006: "DRV_ERROR_FILELOAD",
    20007: "DRV_UNKNOWN_FUNCTION",
    20013: "DRV_ERROR_ACK",
    20017: "DRV_ACQUISITION_ERRORS",
    20018: "DRV_ACQ_BUFFER",
    20019: "DRV_ACQ_DOWNFIFO_FULL",
    20022: "DRV_KINETIC_TIME_NOT_MET",
    20023: "DRV_ACCUM_TIME_NOT_MET",
    20024: "DRV_NO_NEW_DATA",
    20066: "DRV_P1INVALID",
    20067: "DRV_P2INVALID",
    20068: "DRV_P3INVALID",
    20069: "DRV_P4INVALID",
    20070: "DRV_INIERROR",
    20072: "DRV_ACQUIRING",
    20073: "DRV_IDLE",
    20074: "DRV_TEMPCYCLE",
    20075: "DRV_NOT_INITIALIZED",
    20076: "DRV_P5INVALID",
    20078: "DRV_INVALID_MODE",
    20089: "DRV_USBERROR",
    20099: "DRV_BINNING_ERROR",
    20100: "DRV_INVALID_AMPLIFIER",
    20990: "DRV_ERROR_NOCAMERA",
    20991: "DRV_NOT_SUPPORTED",
    20992: "DRV_NOT_AVAILABLE",
}

P_INT = POINTER(c_int)
P_FLOAT = POINTER(c_float)

# Every export returns an unsigned status code. Declaring the arguments is
# what keeps a Python float from reaching a C ``float`` as a double.
SIGNATURES: dict[str, list[type]] = {
    "Initialize": [c_char_p],
    "ShutDown": [],
    "GetAvailableCameras": [POINTER(c_long)],
    "GetCameraHandle": [c_long, POINTER(c_long)],
    "SetCurrentCamera": [c_long],
    "GetHeadModel": [c_char_p],
    "GetCameraSerialNumber": [P_INT],
    "GetDetector": [P_INT, P_INT],
    "GetPixelSize": [P_FLOAT, P_FLOAT],
    "GetNumberAmp": [P_INT],
    "GetNumberHSSpeeds": [c_int, c_int, P_INT],
    "GetHSSpeed": [c_int, c_int, c_int, P_FLOAT],
    "GetFastestRecommendedVSSpeed": [P_INT, P_FLOAT],
    "GetNumberPreAmpGains": [P_INT],
    "GetPreAmpGain": [c_int, P_FLOAT],
    "IsPreAmpGainAvailable": [c_int, c_int, c_int, c_int, P_INT],
    "GetEMGainRange": [P_INT, P_INT],
    "GetTemperatureRange": [P_INT, P_INT],
    "SetAcquisitionMode": [c_int],
    "SetReadMode": [c_int],
    "SetTriggerMode": [c_int],
    "SetSingleTrack": [c_int, c_int],
    "SetNumberAccumulations": [c_int],
    "SetAccumulationCycleTime": [c_float],
    "SetExposureTime": [c_float],
    "GetAcquisitionTimings": [P_FLOAT, P_FLOAT, P_FLOAT],
    "SetADChannel": [c_int],
    "SetOutputAmplifier": [c_int],
    "SetHSSpeed": [c_int, c_int],
    "SetVSSpeed": [c_int],
    "SetPreAmpGain": [c_int],
    "SetEMCCDGain": [c_int],
    "SetShutter": [c_int, c_int, c_int, c_int],
    "SetTemperature": [c_int],
    "GetTemperatureF": [P_FLOAT],
    "CoolerON": [],
    "CoolerOFF": [],
    "SetCoolerMode": [c_int],
    "StartAcquisition": [],
    "AbortAcquisition": [],
    "GetStatus": [P_INT],
    "GetAcquiredData": [POINTER(c_int32), c_ulong],
}


class AndorError(DeviceError):
    """An Andor SDK call failed or the camera could not be found."""


def code_name(code: int) -> str:
    return CODE_NAMES.get(code, str(code))


class AndorSDK:
    """The DLL's exports with their C signatures declared."""

    def __init__(self, dll: ctypes.WinDLL):
        self.dll = dll
        for name, argtypes in SIGNATURES.items():
            func = getattr(dll, name)
            func.argtypes = argtypes
            func.restype = c_ulong

    def call(self, name: str, *args: object) -> int:
        return getattr(self.dll, name)(*args)

    def check(self, name: str, *args: object) -> None:
        code = self.call(name, *args)
        if code != SUCCESS:
            raise AndorError(f"{name} failed: {code_name(code)}")


@functools.cache
def load_sdk(dll_path: str) -> AndorSDK:
    """The one ``AndorSDK`` for this process. Its folder is added to the DLL
    search so the library finds anything it loads beside itself."""
    path = Path(dll_path)
    if not path.is_file():
        raise AndorError(f"{path} does not exist; install the Andor SDK or set the DLL path")
    os.add_dll_directory(str(path.parent))
    return AndorSDK(ctypes.WinDLL(str(path)))


@dataclass(frozen=True)
class CameraInfo:
    """What the head reports once at connect; none of it changes per run."""

    head_model: str
    serial: int
    width: int
    height: int
    pixel_um: float
    # Horizontal shift speeds in MHz, indexed by output amplifier.
    hs_mhz: tuple[tuple[float, ...], ...]
    vs_index: int
    preamp_gains: tuple[float, ...]
    em_gain_range: tuple[int, int]


@dataclass(frozen=True)
class Temperature:
    """``celsius`` is None while acquiring: the camera does not read it then."""

    celsius: float | None
    status: str


def select_first_camera(sdk: AndorSDK) -> None:
    count = c_long()
    sdk.check("GetAvailableCameras", byref(count))
    if count.value == 0:
        raise AndorError(
            "no Andor camera found; check it is powered and that the DLL path points at "
            "the standalone Andor SDK, not Andor SOLIS"
        )
    handle = c_long()
    sdk.check("GetCameraHandle", c_long(0), byref(handle))
    sdk.check("SetCurrentCamera", handle)


def initialize(sdk: AndorSDK) -> None:
    code = sdk.call("Initialize", c_char_p(b""))
    if code != SUCCESS:
        raise AndorError(f"Initialize failed: {code_name(code)}; is Andor SOLIS still open?")


def read_ints(sdk: AndorSDK, name: str, *args: object, count: int) -> list[int]:
    outs = [c_int() for _ in range(count)]
    sdk.check(name, *args, *(byref(out) for out in outs))
    return [out.value for out in outs]


def read_floats(sdk: AndorSDK, name: str, *args: object, count: int) -> list[float]:
    outs = [c_float() for _ in range(count)]
    sdk.check(name, *args, *(byref(out) for out in outs))
    return [out.value for out in outs]


def read_hs_speeds(sdk: AndorSDK, amplifier: int) -> tuple[float, ...]:
    (count,) = read_ints(sdk, "GetNumberHSSpeeds", 0, amplifier, count=1)
    return tuple(
        read_floats(sdk, "GetHSSpeed", 0, amplifier, index, count=1)[0] for index in range(count)
    )


def read_info(sdk: AndorSDK) -> CameraInfo:
    model = ctypes.create_string_buffer(64)
    sdk.check("GetHeadModel", model)
    width, height = read_ints(sdk, "GetDetector", count=2)
    (amplifiers,) = read_ints(sdk, "GetNumberAmp", count=1)
    (preamps,) = read_ints(sdk, "GetNumberPreAmpGains", count=1)
    vs_index = c_int()
    sdk.check("GetFastestRecommendedVSSpeed", byref(vs_index), byref(c_float()))
    low, high = read_ints(sdk, "GetEMGainRange", count=2)
    return CameraInfo(
        head_model=model.value.decode("ascii", "replace"),
        serial=read_ints(sdk, "GetCameraSerialNumber", count=1)[0],
        width=width,
        height=height,
        pixel_um=read_floats(sdk, "GetPixelSize", count=2)[0],
        hs_mhz=tuple(read_hs_speeds(sdk, amp) for amp in range(amplifiers)),
        vs_index=vs_index.value,
        preamp_gains=tuple(
            read_floats(sdk, "GetPreAmpGain", index, count=1)[0] for index in range(preamps)
        ),
        em_gain_range=(low, high),
    )


class AndorSession:
    """One initialized camera. Calls take a lock because a program's thread
    and the GUI's temperature readout both talk to it."""

    def __init__(self, sdk: AndorSDK, info: CameraInfo):
        self.sdk = sdk
        self.info = info
        self.lock = threading.Lock()

    @classmethod
    def connect(cls, dll_path: str, target_c: int) -> AndorSession:
        sdk = load_sdk(dll_path)
        select_first_camera(sdk)
        initialize(sdk)
        sdk.check("SetCoolerMode", COOLER_MAINTAIN_ON_SHUTDOWN)
        sdk.check("SetTemperature", target_c)
        sdk.check("CoolerON")
        return cls(sdk, read_info(sdk))

    def check(self, name: str, *args: object) -> None:
        with self.lock:
            self.sdk.check(name, *args)

    def floats(self, name: str, *args: object, count: int) -> list[float]:
        with self.lock:
            return read_floats(self.sdk, name, *args, count=count)

    def preamp_available(self, amplifier: int, hs_index: int, preamp_index: int) -> bool:
        with self.lock:
            (available,) = read_ints(
                self.sdk, "IsPreAmpGainAvailable", 0, amplifier, hs_index, preamp_index, count=1
            )
        return available == 1

    def status(self) -> int:
        with self.lock:
            (state,) = read_ints(self.sdk, "GetStatus", count=1)
        return state

    def temperature(self) -> Temperature:
        reading = c_float()
        with self.lock:
            code = self.sdk.call("GetTemperatureF", byref(reading))
        if code == ACQUIRING:
            return Temperature(None, TEMPERATURE_STATUS[code])
        if code not in TEMPERATURE_STATUS:
            raise AndorError(f"GetTemperatureF failed: {code_name(code)}")
        return Temperature(reading.value, TEMPERATURE_STATUS[code])

    def acquired_data(self, count: int) -> np.ndarray:
        buffer = (c_int32 * count)()
        self.check("GetAcquiredData", buffer, c_ulong(count))
        return np.ctypeslib.as_array(buffer).copy()

    def close(self) -> None:
        self.check("ShutDown")
