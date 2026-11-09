"""The Simulated Dish: a simulated sample on a simulated stage, which simulated programs
acquire from."""

# The dish lives as long as the device does, so its cells, dose and drift carry over from run
# to run as a real sample's would. Changing the settings makes a new dish.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyrpoc.core.device import Device, DeviceConfig
from pyrpoc.utils.simulation import Axis, Dish, DishSettings, FocalSurface
from pyrpoc.utils.value_rules import float_field, int_field

POLL_S = 0.05


@dataclass
class DishConfig(DeviceConfig):
    seed: int = int_field("Seed", 7, minimum=0, tooltip="Which dish: the same seed, the same cells")
    cell_speed_um_min: float = float_field(
        "Cell Speed (µm/min)", 1.0, minimum=0.0, decimals=3, tooltip="Mean speed cells move at"
    )
    clock_speed: float = float_field(
        "Clock Speed",
        20.0,
        minimum=1.0,
        decimals=1,
        step=1.0,
        tooltip="How many times faster than wall time the dish's time runs",
    )
    focus_tilt_x: float = float_field(
        "Focus Tilt X (µm/µm)", 0.004, decimals=5, step=0.001, tooltip="Focus change per µm in x"
    )
    focus_tilt_y: float = float_field(
        "Focus Tilt Y (µm/µm)", -0.003, decimals=5, step=0.001, tooltip="Focus change per µm in y"
    )
    focus_drift_um_min: float = float_field(
        "Focus Drift (µm/min)", 0.03, decimals=4, step=0.01, tooltip="How fast focus drifts"
    )
    stage_drift_x_um_min: float = float_field(
        "Stage Drift X (µm/min)", 0.05, decimals=4, step=0.01, tooltip="How fast the stage creeps"
    )
    stage_drift_y_um_min: float = float_field(
        "Stage Drift Y (µm/min)", -0.03, decimals=4, step=0.01, tooltip="How fast the stage creeps"
    )
    um_per_volt_x: float = float_field(
        "True µm per Volt X", 102.0, minimum=1.0, decimals=3, tooltip="What calibration should find"
    )
    um_per_volt_y: float = float_field(
        "True µm per Volt Y", 97.5, minimum=1.0, decimals=3, tooltip="What calibration should find"
    )

    def settings(self) -> DishSettings:
        return DishSettings(
            seed=self.seed,
            cell_speed_um_min=self.cell_speed_um_min,
            clock_speed=self.clock_speed,
            focus=FocalSurface(0.0, self.focus_tilt_x, self.focus_tilt_y, self.focus_drift_um_min),
            stage_drift_um_min=(self.stage_drift_x_um_min, self.stage_drift_y_um_min),
            um_per_volt=(self.um_per_volt_x, self.um_per_volt_y),
        )


class SimulatedDish(Device):
    display_name = "Simulated Dish"
    config_cls = DishConfig

    config: DishConfig

    def __init__(self, instance_id: str | None = None):
        super().__init__(instance_id=instance_id)
        self.current: Dish | None = None

    def summary(self) -> str:
        return f"seed {self.config.seed}, {self.config.clock_speed:g}× time"

    @property
    def dish(self) -> Dish:
        """The dish its settings make, made again when they change."""
        settings = self.config.settings()
        if self.current is None or self.current.settings != settings:
            self.current = Dish(settings)
        return self.current

    def now(self) -> float:
        """Seconds of dish time."""
        return self.dish.clock.now()

    def wait(self, seconds: float, sleep: Callable[[float], None]) -> None:
        """Wait ``seconds`` of dish time with ``sleep``, which takes wall seconds."""
        sleep(self.dish.clock.wall_seconds(seconds))

    def xy_position(self) -> tuple[float, float]:
        x, y, _ = self.dish.stage.position()
        return x, y

    def z_position(self) -> float:
        return self.dish.stage.position()[2]

    def move_xy_to(self, x: float, y: float) -> None:
        self.dish.stage.move_xy_to(x, y)

    def move_z_to(self, z: float) -> None:
        self.dish.stage.move_z_to(z)

    def busy(self, axis: Axis) -> bool:
        return self.dish.stage.busy(axis)

    def wait_until_stopped(self, axis: Axis, sleep: Callable[[float], None]) -> None:
        """Block until ``axis`` stops, waiting with ``sleep``."""
        while self.busy(axis):
            sleep(min(POLL_S, self.dish.clock.wall_seconds(self.dish.stage.remaining_s(axis))))

    def stop_smoothly(self) -> None:
        self.dish.stage.stop()
