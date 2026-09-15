"""Galvo volts, typed or picked off an image.

Pulled out of the generic form-builder engine because it is not a generic
widget: it knows about ``Point``/``PointField`` (an acquisition parameter
type) and about arming, the UI half of the picking feature in
``pyrpoc.app.picking``. Registers ``build_point`` into that engine's
``BUILDERS`` on import, which is why ``panels/acquisition/__init__.py``
imports this module before building any form that might contain a
``PointField``.

The button is a widget affordance, not a parameter: which is the whole reason
arming is not a field. ``Browse...`` on a path field is the same idea -- press
it, something outside the form temporarily takes over to fill one value, it
ends. Nothing about the button is validated, encoded or persisted, so the
application can never come back from a relaunch armed at hardware.
"""

from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyrpoc.programs.components import Point, PointField
from pyrpoc.qt_widgets.param_form import BUILDERS, FieldWidget

#: Shown under the spin boxes when no point has been set yet.
NO_ORIGIN = "—"


class PointPicker(QWidget):
    """Galvo volts, typed or picked off an image.

    It is checkable because arming outlives the press: the click that fills it
    happens somewhere else entirely. ``set_armed`` exists for that reason -- the
    application decides when arming ends, and says so.
    """

    changed = pyqtSignal()
    arm_requested = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._point = Point()
        #: True while ``set_value`` is driving the spin boxes, so a programmatic
        #: write keeps the provenance a hand edit is supposed to clear.
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

        self.pick_btn = QPushButton("⊕ Acquire at point…", self)
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip(
            "Arm, then click a point on an image to park the galvos there and "
            "acquire. Clicking moves hardware."
        )
        root.addWidget(self.pick_btn)

        self.origin_label = QLabel(f"from: {NO_ORIGIN}", self)
        self.origin_label.setEnabled(False)
        root.addWidget(self.origin_label)

        self.fast_spin.valueChanged.connect(self.on_spin_changed)
        self.slow_spin.valueChanged.connect(self.on_spin_changed)
        self.pick_btn.toggled.connect(self.arm_requested.emit)

    def build_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.01)
        spin.setSuffix(" V")
        spin.setToolTip(tooltip)
        return spin

    # -- value -------------------------------------------------------------- #

    def value(self) -> Point:
        return Point(
            self.fast_spin.value(),
            self.slow_spin.value(),
            self._point.source_id,
            self._point.source_label,
            self._point.pixel_x,
            self._point.pixel_y,
        )

    def set_value(self, value: Any) -> None:
        """Blocks -> widget. Keeps the provenance the incoming point carries."""
        point = Point.from_dict(value)
        self._point = point
        self._programmatic = True
        try:
            self.fast_spin.setValue(point.fast_v)
            self.slow_spin.setValue(point.slow_v)
        finally:
            self._programmatic = False
        origin = point.describe() if point.picked else NO_ORIGIN
        self.origin_label.setText(f"from: {origin}")

    def on_spin_changed(self) -> None:
        """A hand edit is no longer the pixel it was picked from, so say so.

        ``changed`` is emitted either way. A programmatic set only ever happens
        inside ``ParamForm.reload``, whose ``_loading`` guard swallows it, so
        emitting unconditionally costs nothing and means a typed value can never
        be the one case that fails to reach the block.
        """
        if not self._programmatic:
            self._point = Point(self.fast_spin.value(), self.slow_spin.value())
            self.origin_label.setText("from: typed")
        self.changed.emit()

    def summary(self) -> str:
        return f"{self.fast_spin.value():.3f} / {self.slow_spin.value():.3f} V"

    # -- arming ------------------------------------------------------------- #

    def set_armed(self, active: bool) -> None:
        """Reflect the application's arming state without asking for a change."""
        if self.pick_btn.isChecked() == bool(active):
            return
        self.pick_btn.blockSignals(True)
        self.pick_btn.setChecked(bool(active))
        self.pick_btn.blockSignals(False)


def build_point(spec: PointField, parent) -> FieldWidget:
    del spec
    picker = PointPicker(parent)

    # Statement bodies rather than lambdas: ``connect`` returns a Connection,
    # and the hooks are declared as returning None.
    def connect(cb: Callable[[], None]) -> None:
        picker.changed.connect(lambda *_: cb())

    def arm(cb: Callable[[bool], None]) -> None:
        picker.arm_requested.connect(cb)

    return FieldWidget(
        picker,
        get=picker.value,
        set=picker.set_value,
        connect=connect,
        summary=picker.summary,
        arm=arm,
        set_armed=picker.set_armed,
    )


BUILDERS[PointField] = build_point
