"""The Prior Scientific SDK over ctypes: one DLL per process, one session per
controller.

The DLL exports five functions; everything else is a text command such as
``controller.stage.position.get`` sent through ``PriorScientificSDK_cmd``.
"""

from __future__ import annotations

import ctypes
import functools
import threading
from pathlib import Path

from pyrpoc.structs.plugins.devices import DeviceError

# ``pyrpoc/assets/sdks``, two levels up from ``pyrpoc/devices/prior_stage``.
DLL_PATH = Path(__file__).resolve().parents[2] / "assets" / "sdks" / "PriorScientificSDK.dll"

# The size Prior's own examples use; every reply is a short line of text.
RX_SIZE = 1000

# Codes observed from the DLL, named so a failed connect reads as a cause.
MEANINGS = {
    -10002: "the COM port could not be opened",
    -10004: "the controller is not connected",
}


class PriorError(DeviceError):
    """A Prior SDK call or controller command failed."""


class PriorSDK:
    """The DLL's exports with their C signatures declared, so ctypes neither
    guesses argument types nor truncates return values."""

    def __init__(self, dll: ctypes.WinDLL):
        self.dll = dll
        dll.PriorScientificSDK_Initialise.argtypes = []
        dll.PriorScientificSDK_Initialise.restype = ctypes.c_int
        dll.PriorScientificSDK_Version.argtypes = [ctypes.c_char_p]
        dll.PriorScientificSDK_Version.restype = ctypes.c_int
        dll.PriorScientificSDK_OpenNewSession.argtypes = []
        dll.PriorScientificSDK_OpenNewSession.restype = ctypes.c_int
        dll.PriorScientificSDK_CloseSession.argtypes = [ctypes.c_int]
        dll.PriorScientificSDK_CloseSession.restype = ctypes.c_int
        dll.PriorScientificSDK_cmd.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
        dll.PriorScientificSDK_cmd.restype = ctypes.c_int

        ret = dll.PriorScientificSDK_Initialise()
        if ret != 0:
            raise PriorError(f"could not initialise the Prior SDK (code {ret})")

    def version(self) -> str:
        rx = ctypes.create_string_buffer(RX_SIZE)
        ret = self.dll.PriorScientificSDK_Version(rx)
        if ret != 0:
            raise PriorError(f"could not read the Prior SDK version (code {ret})")
        return rx.value.decode()

    def open_session(self) -> int:
        session = self.dll.PriorScientificSDK_OpenNewSession()
        if session < 0:
            raise PriorError(f"could not open a Prior SDK session (code {session})")
        return session

    def close_session(self, session: int) -> None:
        ret = self.dll.PriorScientificSDK_CloseSession(session)
        if ret != 0:
            raise PriorError(f"could not close Prior SDK session {session} (code {ret})")

    def cmd(self, session: int, text: str) -> str:
        rx = ctypes.create_string_buffer(RX_SIZE)
        ret = self.dll.PriorScientificSDK_cmd(session, text.encode(), rx)
        reply = rx.value.decode().strip()
        if ret != 0:
            meaning = MEANINGS.get(ret, f"reply {reply!r}")
            raise PriorError(f"{text!r} failed (code {ret}): {meaning}")
        return reply


@functools.cache
def load_sdk() -> PriorSDK:
    """The one ``PriorSDK`` for this process: ``Initialise`` runs once."""
    return PriorSDK(ctypes.WinDLL(str(DLL_PATH)))


class PriorSession:
    """One controller on one COM port. Commands take a lock because a
    program's thread and the GUI can both be talking to the stage."""

    def __init__(self, sdk: PriorSDK, session: int):
        self.sdk = sdk
        self.session = session
        self.lock = threading.Lock()

    @classmethod
    def connect(cls, com_port: int) -> PriorSession:
        sdk = load_sdk()
        session = sdk.open_session()
        try:
            sdk.cmd(session, f"controller.connect {com_port}")
        except PriorError:
            sdk.close_session(session)
            raise
        return cls(sdk, session)

    def cmd(self, text: str) -> str:
        with self.lock:
            return self.sdk.cmd(self.session, text)

    def close(self) -> None:
        with self.lock:
            try:
                self.sdk.cmd(self.session, "controller.disconnect")
            finally:
                self.sdk.close_session(self.session)
