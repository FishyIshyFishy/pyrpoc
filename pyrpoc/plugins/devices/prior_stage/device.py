"""The Prior Scientific controller: an XY stage and a Z focus drive on one COM
port, in microns.

The session stays open between runs; the app opens it at launch and closes it
on exit. Speed, acceleration and jerk are never written implicitly, because
someone may have tuned them outside pyrpoc: they are read, and set only when
asked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pyrpoc.structs.plugins import params as P
from pyrpoc.structs.plugins.devices import Device, device_registry

from .sdk import PriorError, PriorSession

if TYPE_CHECKING:  # pragma: no cover
    from PyQt6.QtWidgets import QWidget

# "xy" is what the SDK calls the stage; "z" is the focus drive.
Axis = Literal["xy", "z"]
PREFIX: dict[Axis, str] = {"xy": "controller.stage", "z": "controller.z"}

# How often a wait asks the controller whether a move has finished.
POLL_S = 0.05


@dataclass(frozen=True)
class MotionProfile:
    """How an axis moves: top speed (µm/s), acceleration (µm/s²), jerk (µm/s³)."""

    speed: float
    acc: float
    jerk: float


@dataclass
class PriorConfig(P.Group):
    com_port: int = P.int_field(
        "COM Port", 4, minimum=1, maximum=256, tooltip="COM port of the Prior controller (4 = COM4)"
    )


def parse_numbers(reply: str, count: int) -> list[float]:
    """A reply is comma-separated numbers; anything else is a fault on the
    line, not a value to guess at."""
    parts = reply.split(",")
    if len(parts) != count:
        raise PriorError(f"expected {count} number(s) from the controller, got {reply!r}")
    try:
        return [float(part) for part in parts]
    except ValueError:
        raise PriorError(f"expected numbers from the controller, got {reply!r}") from None


def format_number(value: float) -> str:
    """Up to three decimals, without trailing zeros: 1000.0 -> '1000'."""
    return f"{value:.3f}".rstrip("0").rstrip(".")


@device_registry.register("prior_stage")
class PriorStage(Device):
    display_name = "Prior Stage"
    owns_connection = True
    holds_session = True
    config_cls = PriorConfig

    config: PriorConfig

    def __init__(self, instance_id: str | None = None, user_label: str | None = None):
        super().__init__(instance_id=instance_id, user_label=user_label)
        self.session: PriorSession | None = None

    def summary(self) -> str:
        state = "connected" if self.session_open else "not connected"
        return f"COM{self.config.com_port} - {state}"

    @property
    def session_open(self) -> bool:
        return self.session is not None

    def open_session(self) -> None:
        if self.session is not None:
            raise PriorError(f"{self.name} is already connected")
        self.session = PriorSession.connect(self.config.com_port)

    def close_session(self) -> None:
        # Dropped first, so a controller that faults on disconnect still
        # leaves the device closed rather than half-open.
        session = self.require_session()
        self.session = None
        session.close()

    def check_reachable(self) -> bool:
        self.command("controller.model.get")
        return True

    def require_session(self) -> PriorSession:
        if self.session is None:
            raise PriorError(f"{self.name} is not connected; connect it in the Devices panel")
        return self.session

    def command(self, text: str) -> str:
        return self.require_session().cmd(text)

    def read_number(self, text: str) -> float:
        (value,) = parse_numbers(self.command(text), 1)
        return value

    def xy_position(self) -> tuple[float, float]:
        x, y = parse_numbers(self.command("controller.stage.position.get"), 2)
        return x, y

    def z_position(self) -> float:
        return self.read_number("controller.z.position.get")

    def move_xy_to(self, x: float, y: float) -> None:
        self.command(f"controller.stage.goto-position {format_number(x)} {format_number(y)}")

    def move_xy_by(self, dx: float, dy: float) -> None:
        self.command(f"controller.stage.move-relative {format_number(dx)} {format_number(dy)}")

    def move_z_to(self, z: float) -> None:
        self.command(f"controller.z.goto-position {format_number(z)}")

    def move_z_by(self, dz: float) -> None:
        self.command(f"controller.z.move-relative {format_number(dz)}")

    def busy(self, axis: Axis) -> bool:
        return self.read_number(f"{PREFIX[axis]}.busy.get") != 0

    def wait_until_stopped(self, axis: Axis, sleep: Callable[[float], None]) -> None:
        """Block until ``axis`` stops. A program passes ``ctx.sleep`` so a
        stopped run interrupts the wait; its ``finally`` should then call
        ``stop_smoothly()``."""
        while self.busy(axis):
            sleep(POLL_S)

    def stop_smoothly(self) -> None:
        self.command("controller.stop.smoothly")

    def stop_abruptly(self) -> None:
        self.command("controller.stop.abruptly")

    def motion(self, axis: Axis) -> MotionProfile:
        prefix = PREFIX[axis]
        return MotionProfile(
            speed=self.read_number(f"{prefix}.speed.get"),
            acc=self.read_number(f"{prefix}.acc.get"),
            jerk=self.read_number(f"{prefix}.jerk.get"),
        )

    def set_motion(self, axis: Axis, profile: MotionProfile) -> None:
        prefix = PREFIX[axis]
        self.command(f"{prefix}.speed.set {format_number(profile.speed)}")
        self.command(f"{prefix}.acc.set {format_number(profile.acc)}")
        self.command(f"{prefix}.jerk.set {format_number(profile.jerk)}")

    def panel(self, parent: QWidget, on_change: Callable[[], None]) -> QWidget:
        from .panel import PriorStagePanel

        return PriorStagePanel(self, parent, on_change)
