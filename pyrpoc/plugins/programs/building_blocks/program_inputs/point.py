"""The point a point acquisition parks the galvos at, and the pixel it was
picked from, with its own rules and editor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from pyrpoc.core.input_section import InputSection
from pyrpoc.core.pixel_pick import PixelPick
from pyrpoc.core.program_inputs import (
    Editor,
    Field,
    ParameterError,
    declare,
)

from .scan import ScanGeometry


@dataclass(frozen=True)
class Point:
    """One parked galvo position in volts, and the image and pixel it was picked
    from, which a typed point has none of."""

    # Volts are what a run reads, so it reproduces from its saved parameters
    # alone: a pixel only means something against the scan it was picked from.
    fast_v: float = 0.0
    slow_v: float = 0.0
    source_id: str = ""
    source_label: str = ""
    pixel_x: int = -1
    pixel_y: int = -1

    @classmethod
    def from_pick(cls, pick: PixelPick) -> Point:
        """The position the picked pixel was measured at."""
        # From the clicked image's provenance, not the current inputs, which may
        # have been edited since the image was taken.
        image = pick.output
        scan = ScanGeometry.from_parameters(image.provenance.parameters, image.label)
        fast_v, slow_v = scan.voltage_at(pick.x, pick.y)
        return cls(fast_v, slow_v, image.id, image.label, pick.x, pick.y)

    @property
    def picked(self) -> bool:
        """Whether this point was picked from data rather than typed."""
        return bool(self.source_id) and self.pixel_x >= 0 and self.pixel_y >= 0

    def describe(self) -> str:
        """Where this point came from, for the label under the spin boxes."""
        if not self.picked:
            return "typed"
        return f"{self.source_label or self.source_id} px ({self.pixel_x}, {self.pixel_y})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast_v": self.fast_v,
            "slow_v": self.slow_v,
            "source_id": self.source_id,
            "source_label": self.source_label,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
        }

    @classmethod
    def from_value(cls, raw: Any) -> Point:
        """A point from the form (already a ``Point``) or from JSON (a dict)."""
        if isinstance(raw, Point):
            return raw
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ParameterError("a point must be an object with fast_v/slow_v")
        try:
            return cls(
                fast_v=float(raw.get("fast_v", 0.0)),
                slow_v=float(raw.get("slow_v", 0.0)),
                source_id=str(raw.get("source_id", "")),
                source_label=str(raw.get("source_label", "")),
                pixel_x=int(raw.get("pixel_x", -1)),
                pixel_y=int(raw.get("pixel_y", -1)),
            )
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"a point's volts and pixel must be numbers: {exc}") from exc


# Shown under the spin boxes when no point has been set yet.
NO_ORIGIN = "—"


class PointPicker(QWidget):
    """Galvo volts, typed or picked off an image, and where they came from."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._point = Point()
        # True while ``set_value`` drives the spin boxes, so a programmatic
        # write keeps the provenance a hand edit clears.
        self._programmatic = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        volts = QHBoxLayout()
        volts.setContentsMargins(0, 0, 0, 0)
        self.fast_spin = self.build_spin("Fast axis (X) volts")
        self.slow_spin = self.build_spin("Slow axis (Y) volts")
        volts.addWidget(QLabel("X", self))
        volts.addWidget(self.fast_spin, 1)
        volts.addWidget(QLabel("Y", self))
        volts.addWidget(self.slow_spin, 1)
        root.addLayout(volts)
        self.origin_label = QLabel(f"from: {NO_ORIGIN}", self)
        self.origin_label.setEnabled(False)
        root.addWidget(self.origin_label)

        self.fast_spin.valueChanged.connect(self.on_spin_changed)
        self.slow_spin.valueChanged.connect(self.on_spin_changed)

    def build_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.01)
        spin.setSuffix(" V")
        spin.setToolTip(tooltip)
        return spin

    def value(self) -> Point:
        return Point(
            self.fast_spin.value(),
            self.slow_spin.value(),
            self._point.source_id,
            self._point.source_label,
            self._point.pixel_x,
            self._point.pixel_y,
        )

    def set_value(self, point: Point) -> None:
        """Blocks -> widget, keeping the provenance the point carries."""
        self._point = point
        self._programmatic = True
        try:
            self.fast_spin.setValue(point.fast_v)
            self.slow_spin.setValue(point.slow_v)
        finally:
            self._programmatic = False
        self.origin_label.setText(f"from: {point.describe() if point.picked else NO_ORIGIN}")

    def on_spin_changed(self) -> None:
        """Mark a hand edit as typed, since it no longer names a picked pixel."""
        if not self._programmatic:
            self._point = Point(self.fast_spin.value(), self.slow_spin.value())
            self.origin_label.setText("from: typed")
        self.changed.emit()

    def summary(self) -> str:
        return f"{self.fast_spin.value():.3f} / {self.slow_spin.value():.3f} V"


@dataclass(frozen=True)
class PointField(Field):
    """The rules for a galvo position: volts, plus the pixel they were picked from, if any."""

    # ``coerce`` takes a live ``Point`` as well as JSON, since every value is re-checked before a
    # run.

    accepts = PixelPick

    def coerce(self, value: Any) -> Point:
        return Point.from_value(value)

    def encode(self, value: Point) -> Any:
        return value.to_dict()

    def place(self, value: Point, slot: str, item: PixelPick) -> Point:
        """The position the picked pixel was measured at."""
        del value, slot
        return Point.from_pick(item)

    def editor(self, parent: Any) -> Editor:
        picker = PointPicker(parent)
        return Editor(
            picker,
            get=picker.value,
            set=picker.set_value,
            connect=lambda cb: picker.changed.connect(cb),
            summary=picker.summary,
            spec=self,
        )


# Its own section rather than part of the detector's, because the galvos and
# the detector are configured independently.
TARGET = declare(
    "point.target",
    PointField(
        "Target",
        "Galvo position to park at. Type volts, or click a pixel in an image panel whose "
        "'Click sets' is Target",
    ),
    Point(),
)

POINT_SECTION = InputSection("Point", (TARGET,))
