"""The ToupTek camera SDK (``toupcam.dll``) over ctypes: one DLL per process,
one handle per camera.

Only the calls a program needs are declared. Frames are pulled as the SDK's
default 24-bit colour, which needs no option constants set, and are left to
the caller to reduce.
"""

from __future__ import annotations

import ctypes
import functools
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyrpoc.structs.plugins.devices import DeviceError

# ``pyrpoc/assets/sdks``, three levels up from ``plugins/devices/touptek_camera``.
DLL_PATH = Path(__file__).resolve().parents[3] / "assets" / "sdks" / "toupcam.dll"

MAX_CAMERAS = 128
MAX_RESOLUTIONS = 16
BITS = 24

EVENT_IMAGE = 0x0004
EVENT_ERROR = 0x0080
EVENT_DISCONNECTED = 0x0081
EVENT_NOFRAMETIMEOUT = 0x0082
FAULTS = {
    EVENT_ERROR: "the camera reported an error",
    EVENT_DISCONNECTED: "the camera was disconnected",
    EVENT_NOFRAMETIMEOUT: "the camera stopped sending frames",
}

E_PENDING = 0x8000000A

# Another program streaming from the camera does not stop Toupcam_Open; it
# shows up instead as these two codes on the first setting or start.
IN_USE_HINT = "; is ToupView or another program using the camera?"

# The SDK's failures are Windows HRESULTs, named so a failed call reads as a cause.
MEANINGS = {
    0x80070005: "access denied",
    0x80070057: "invalid argument",
    0x80004001: "not supported by this camera",
    0x80004003: "null pointer",
    0x8000FFFF: "unexpected call in this state" + IN_USE_HINT,
    0x8001010E: "called from the wrong thread",
    0x8007001F: "the camera stopped working",
    0x800700AA: "the camera is busy" + IN_USE_HINT,
    E_PENDING: "no frame ready yet",
    0x8001011F: "timed out",
}

EventCallback = ctypes.WINFUNCTYPE(None, ctypes.c_uint, ctypes.c_void_p)


class ToupcamError(DeviceError):
    """A ToupTek SDK call failed."""


def describe_hresult(code: int) -> str:
    code &= 0xFFFFFFFF
    return MEANINGS.get(code, f"error 0x{code:08X}")


class Resolution(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint), ("height", ctypes.c_uint)]


class ModelV2(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_wchar_p),
        ("flag", ctypes.c_ulonglong),
        ("maxspeed", ctypes.c_uint),
        ("preview", ctypes.c_uint),
        ("still", ctypes.c_uint),
        ("maxfanspeed", ctypes.c_uint),
        ("ioctrol", ctypes.c_uint),
        ("xpixsz", ctypes.c_float),
        ("ypixsz", ctypes.c_float),
        ("res", Resolution * MAX_RESOLUTIONS),
    ]


class DeviceV2(ctypes.Structure):
    _fields_ = [
        ("displayname", ctypes.c_wchar * 64),
        ("id", ctypes.c_wchar * 64),
        ("model", ctypes.POINTER(ModelV2)),
    ]


@dataclass(frozen=True)
class CameraInfo:
    """One camera the SDK can see: its name, the id to open it by, and the
    frame sizes it offers, largest first."""

    name: str
    id: str
    resolutions: tuple[tuple[int, int], ...]


def row_pitch(width: int) -> int:
    """Bytes per row of a 24-bit frame: the SDK pads rows to four bytes."""
    return ((width * BITS + 31) // 32) * 4


def unpack_rgb24(buffer: np.ndarray, width: int, height: int) -> np.ndarray:
    """A pulled frame as ``(H, W, 3)`` uint8, with the row padding dropped."""
    rows = buffer[: row_pitch(width) * height].reshape(height, row_pitch(width))
    return rows[:, : width * 3].reshape(height, width, 3).copy()


class ToupcamSDK:
    """The DLL's exports with their C signatures declared, so ctypes neither
    guesses argument types nor truncates the handle."""

    def __init__(self, dll: ctypes.WinDLL):
        self.dll = dll
        handle = ctypes.c_void_p
        uint_p = ctypes.POINTER(ctypes.c_uint)
        ushort_p = ctypes.POINTER(ctypes.c_ushort)
        signatures = {
            "Toupcam_Version": ([], ctypes.c_wchar_p),
            "Toupcam_EnumV2": ([ctypes.POINTER(DeviceV2)], ctypes.c_uint),
            "Toupcam_Open": ([ctypes.c_wchar_p], handle),
            "Toupcam_Close": ([handle], None),
            "Toupcam_put_eSize": ([handle, ctypes.c_uint], ctypes.c_long),
            "Toupcam_get_Size": (
                [handle, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
                ctypes.c_long,
            ),
            "Toupcam_put_AutoExpoEnable": ([handle, ctypes.c_int], ctypes.c_long),
            "Toupcam_put_ExpoTime": ([handle, ctypes.c_uint], ctypes.c_long),
            "Toupcam_get_ExpTimeRange": ([handle, uint_p, uint_p, uint_p], ctypes.c_long),
            "Toupcam_put_ExpoAGain": ([handle, ctypes.c_ushort], ctypes.c_long),
            "Toupcam_get_ExpoAGainRange": ([handle, ushort_p, ushort_p, ushort_p], ctypes.c_long),
            "Toupcam_StartPullModeWithCallback": (
                [handle, EventCallback, ctypes.c_void_p],
                ctypes.c_long,
            ),
            "Toupcam_PullImage": (
                [handle, ctypes.c_void_p, ctypes.c_int, uint_p, uint_p],
                ctypes.c_long,
            ),
            "Toupcam_Stop": ([handle], ctypes.c_long),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(dll, name)
            function.argtypes = argtypes
            function.restype = restype

    def check(self, what: str, code: int) -> None:
        if code < 0:
            raise ToupcamError(f"could not {what}: {describe_hresult(code)}")

    def version(self) -> str:
        return self.dll.Toupcam_Version()

    def cameras(self) -> list[CameraInfo]:
        found = (DeviceV2 * MAX_CAMERAS)()
        count = self.dll.Toupcam_EnumV2(found)
        cameras = []
        for device in found[:count]:
            model = device.model.contents
            sizes = tuple((r.width, r.height) for r in model.res[: model.preview])
            cameras.append(CameraInfo(device.displayname, device.id, sizes))
        return cameras


@functools.cache
def load_sdk() -> ToupcamSDK:
    """The one ``ToupcamSDK`` for this process."""
    return ToupcamSDK(ctypes.WinDLL(str(DLL_PATH)))


def find_camera(cameras: list[CameraInfo], wanted: str) -> CameraInfo:
    """The camera named or identified by ``wanted``, or the first one when it
    is blank."""
    if not cameras:
        raise ToupcamError("no ToupTek camera found; check its USB cable")
    if not wanted:
        return cameras[0]
    for camera in cameras:
        if wanted in (camera.name, camera.id):
            return camera
    names = ", ".join(camera.name for camera in cameras)
    raise ToupcamError(f"no ToupTek camera called {wanted!r}; found {names}")


class ToupcamSession:
    """One open camera. Calls take a lock because a program's thread and the
    GUI can both be talking to it; frames arrive on the SDK's own thread and
    only set an event."""

    def __init__(self, sdk: ToupcamSDK, handle: int, camera: CameraInfo):
        self.sdk = sdk
        self.handle = handle
        self.camera = camera
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.fault: str | None = None
        self.streaming = False
        # Kept on the session: ctypes would free a callback the DLL still calls.
        self.callback = EventCallback(self.on_event)

    @classmethod
    def connect(cls, wanted: str) -> ToupcamSession:
        sdk = load_sdk()
        camera = find_camera(sdk.cameras(), wanted)
        handle = sdk.dll.Toupcam_Open(camera.id)
        if not handle:
            raise ToupcamError(
                f"could not open {camera.name}; close ToupView or any other program using it"
            )
        return cls(sdk, handle, camera)

    def on_event(self, event: int, _context: int) -> None:
        if event == EVENT_IMAGE:
            self.frame_ready.set()
        elif event in FAULTS:
            self.fault = FAULTS[event]
            # Wakes a waiting grab so it reports the fault instead of timing out.
            self.frame_ready.set()

    def exposure_range_us(self) -> tuple[int, int]:
        low, high, default = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        with self.lock:
            self.sdk.check(
                "read the exposure range",
                self.sdk.dll.Toupcam_get_ExpTimeRange(
                    self.handle, ctypes.byref(low), ctypes.byref(high), ctypes.byref(default)
                ),
            )
        return low.value, high.value

    def gain_range_pct(self) -> tuple[int, int]:
        low, high, default = ctypes.c_ushort(), ctypes.c_ushort(), ctypes.c_ushort()
        with self.lock:
            self.sdk.check(
                "read the gain range",
                self.sdk.dll.Toupcam_get_ExpoAGainRange(
                    self.handle, ctypes.byref(low), ctypes.byref(high), ctypes.byref(default)
                ),
            )
        return low.value, high.value

    def start(self, size_index: int, exposure_us: int, gain_pct: int) -> None:
        """Apply the settings and start streaming. The frame size can only
        change while stopped, so it is set first."""
        dll = self.sdk.dll
        with self.lock:
            self.sdk.check("set the frame size", dll.Toupcam_put_eSize(self.handle, size_index))
            self.sdk.check("turn off auto exposure", dll.Toupcam_put_AutoExpoEnable(self.handle, 0))
            self.sdk.check("set the exposure", dll.Toupcam_put_ExpoTime(self.handle, exposure_us))
            self.sdk.check("set the gain", dll.Toupcam_put_ExpoAGain(self.handle, gain_pct))
            self.fault = None
            self.frame_ready.clear()
            self.sdk.check(
                "start streaming",
                dll.Toupcam_StartPullModeWithCallback(self.handle, self.callback, None),
            )
            self.streaming = True

    def frame_size(self) -> tuple[int, int]:
        width, height = ctypes.c_int(), ctypes.c_int()
        with self.lock:
            self.sdk.check(
                "read the frame size",
                self.sdk.dll.Toupcam_get_Size(
                    self.handle, ctypes.byref(width), ctypes.byref(height)
                ),
            )
        return width.value, height.value

    def pull(self, buffer: np.ndarray) -> tuple[int, int] | None:
        """Copy the newest frame into ``buffer``: its size, or None if none is
        ready yet."""
        width, height = ctypes.c_uint(), ctypes.c_uint()
        with self.lock:
            code = self.sdk.dll.Toupcam_PullImage(
                self.handle,
                buffer.ctypes.data,
                BITS,
                ctypes.byref(width),
                ctypes.byref(height),
            )
        if code & 0xFFFFFFFF == E_PENDING:
            return None
        self.sdk.check("read a frame", code)
        return width.value, height.value

    def stop(self) -> None:
        with self.lock:
            if not self.streaming:
                return
            self.streaming = False
            self.sdk.check("stop streaming", self.sdk.dll.Toupcam_Stop(self.handle))

    def close(self) -> None:
        try:
            self.stop()
        finally:
            with self.lock:
                self.sdk.dll.Toupcam_Close(self.handle)
