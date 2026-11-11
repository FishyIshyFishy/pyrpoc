"""The Zaber linear stage: one axis on one COM port, moved in microns. On the rig it is the delay
line that sets the pump-Stokes delay, which hyperspectral imaging steps through."""

# The session stays open between runs; the app opens it at launch and closes it on exit. Opening
# it never moves the stage: homing is asked for, from its panel or by a run that needs it.

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from zaber_motion import MotionLibException, Units
from zaber_motion.ascii import Axis, Connection

from pyrpoc.core.device import Connection as DeviceConnection
from pyrpoc.core.device import Device, DeviceConfig, DeviceError, device_registry
from pyrpoc.utils.value_rules import int_field

# How often a wait asks the stage whether a move has finished.
POLL_S = 0.05
UM = Units.LENGTH_MICROMETRES


class ZaberError(DeviceError):
    """A Zaber stage could not be reached, or refused a command."""


@contextmanager
def reported() -> Iterator[None]:
    """Raise a Zaber library fault as a ``ZaberError``, so callers catch one device error."""
    try:
        yield
    except MotionLibException as exc:
        raise ZaberError(str(exc)) from exc


@dataclass
class ZaberConfig(DeviceConfig):
    com_port: int = int_field(
        "COM Port", 3, minimum=1, maximum=256, tooltip="COM port of the Zaber stage (3 = COM3)"
    )
    axis: int = int_field(
        "Axis", 1, minimum=1, maximum=9, tooltip="Axis on the controller; 1 for a one-axis stage"
    )


@dataclass(frozen=True)
class ZaberSession:
    """An open serial connection and the axis it drives."""

    connection: Connection
    axis: Axis

    @classmethod
    def open(cls, com_port: int, axis_number: int) -> ZaberSession:
        """Open the port and find the first device on it."""
        with reported():
            connection = Connection.open_serial_port(f"COM{com_port}")
        # Closed again on any fault, so a failed connect never holds the port.
        try:
            with reported():
                devices = connection.detect_devices()
                if not devices:
                    raise ZaberError(f"no Zaber device answered on COM{com_port}")
                return cls(connection, devices[0].get_axis(axis_number))
        except ZaberError:
            connection.close()
            raise


@device_registry.register("zaber_stage")
class ZaberStage(Device):
    display_name = "Zaber Stage"
    connection = DeviceConnection.SESSION
    config_cls = ZaberConfig

    config: ZaberConfig

    def __init__(self, instance_id: str | None = None):
        super().__init__(instance_id=instance_id)
        self.session: ZaberSession | None = None

    def connection_info(self) -> str:
        return f"COM{self.config.com_port}"

    @property
    def session_open(self) -> bool:
        return self.session is not None

    def open_session(self) -> None:
        if self.session is not None:
            raise ZaberError(f"{self.display_name} is already connected")
        self.session = ZaberSession.open(self.config.com_port, self.config.axis)

    def close_session(self) -> None:
        # Dropped first, so a port that faults on close still leaves the device closed.
        session = self.require_session()
        self.session = None
        with reported():
            session.connection.close()

    def require_session(self) -> ZaberSession:
        if self.session is None:
            raise ZaberError(
                f"{self.display_name} is not connected; connect it in the Devices panel"
            )
        return self.session

    def position_um(self) -> float:
        with reported():
            return self.require_session().axis.get_position(UM)

    def homed(self) -> bool:
        with reported():
            return self.require_session().axis.is_homed()

    def busy(self) -> bool:
        with reported():
            return self.require_session().axis.is_busy()

    def move_to(self, position_um: float) -> None:
        """Start a move to ``position_um``; ``wait_until_stopped`` waits for it."""
        with reported():
            self.require_session().axis.move_absolute(position_um, UM, wait_until_idle=False)

    def move_by(self, step_um: float) -> None:
        with reported():
            self.require_session().axis.move_relative(step_um, UM, wait_until_idle=False)

    def home(self) -> None:
        """Start homing: the stage drives to its home sensor, which absolute moves count from."""
        with reported():
            self.require_session().axis.home(wait_until_idle=False)

    def stop(self) -> None:
        with reported():
            self.require_session().axis.stop(wait_until_idle=False)

    def wait_until_stopped(self, sleep: Callable[[float], None]) -> None:
        """Block until the stage stops, waiting with ``sleep``."""
        # A program passes ``ctx.sleep`` so a stopped run interrupts the wait; it should then
        # call ``stop()``.
        while self.busy():
            sleep(POLL_S)
