"""How the line-profile and histogram plots are drawn, shared by every tile, and the small
menu that edits them."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

MAX_BINS = 512
LIMIT = 1e12


@dataclass
class AxisRange:
    """An axis that fits the data while ``auto``, and otherwise spans ``low`` to ``high``."""

    auto: bool
    low: float
    high: float


class MeasureSettings(QObject):
    """The plot settings every tile's readout follows."""

    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.profile_y = AxisRange(auto=True, low=0.0, high=1.0)
        self.hist_x = AxisRange(auto=True, low=0.0, high=1.0)
        self.hist_y = AxisRange(auto=True, low=0.0, high=1.0)
        self.auto_bins = True
        self.bins = 64
        self.log_counts = False


class RangeRow(QWidget):
    """An autoscale checkbox with a low and a high box that apply while it is off."""

    edited = pyqtSignal()

    def __init__(self, label: str, parent: QWidget):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.auto = QCheckBox("Auto", self)
        self.low = self.spin()
        self.high = self.spin()
        row.addWidget(self.auto)
        row.addWidget(self.low, 1)
        row.addWidget(QLabel("to", self))
        row.addWidget(self.high, 1)
        self.auto.setToolTip(f"Fit the {label} to the data")
        self.auto.toggled.connect(self.on_edited)
        self.low.valueChanged.connect(self.on_edited)
        self.high.valueChanged.connect(self.on_edited)

    def spin(self) -> QDoubleSpinBox:
        box = QDoubleSpinBox(self)
        box.setRange(-LIMIT, LIMIT)
        box.setDecimals(4)
        box.setKeyboardTracking(False)
        box.setMinimumWidth(70)
        return box

    def on_edited(self) -> None:
        self.low.setEnabled(not self.auto.isChecked())
        self.high.setEnabled(not self.auto.isChecked())
        self.edited.emit()

    def load(self, axis: AxisRange) -> None:
        for widget in (self.auto, self.low, self.high):
            widget.blockSignals(True)
        self.auto.setChecked(axis.auto)
        self.low.setValue(axis.low)
        self.high.setValue(axis.high)
        for widget in (self.auto, self.low, self.high):
            widget.blockSignals(False)
        self.low.setEnabled(not axis.auto)
        self.high.setEnabled(not axis.auto)

    def store(self, axis: AxisRange) -> None:
        axis.auto = self.auto.isChecked()
        axis.low = self.low.value()
        axis.high = self.high.value()


class SettingsForm(QWidget):
    """The fields for the plot that is showing: profile or histogram."""

    def __init__(self, settings: MeasureSettings, parent: QWidget):
        super().__init__(parent)
        self.settings = settings
        form = QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)
        self.profile_y = RangeRow("value axis", self)
        self.hist_x = RangeRow("value axis", self)
        self.hist_y = RangeRow("count axis", self)
        self.auto_bins = QCheckBox("Auto", self)
        self.bins = QSpinBox(self)
        self.bins.setRange(2, MAX_BINS)
        self.bins.setKeyboardTracking(False)
        bins_row = QWidget(self)
        bins_layout = QHBoxLayout(bins_row)
        bins_layout.setContentsMargins(0, 0, 0, 0)
        bins_layout.addWidget(self.auto_bins)
        bins_layout.addWidget(self.bins, 1)
        self.log_counts = QCheckBox("Logarithmic", self)
        self.log_counts.setToolTip("Plot log10(1 + count), so small counts stay visible")

        self.profile_rows = [(QLabel("Value", self), self.profile_y)]
        self.hist_rows = [
            (QLabel("Value", self), self.hist_x),
            (QLabel("Count", self), self.hist_y),
            (QLabel("Bins", self), bins_row),
            (QLabel("Count scale", self), self.log_counts),
        ]
        for label, field in self.profile_rows + self.hist_rows:
            form.addRow(label, field)

        for row in (self.profile_y, self.hist_x, self.hist_y):
            row.edited.connect(self.store)
        self.auto_bins.toggled.connect(self.store)
        self.bins.valueChanged.connect(self.store)
        self.log_counts.toggled.connect(self.store)
        settings.changed.connect(self.load)
        self.load()

    def show_profile(self, profile: bool) -> None:
        for rows, shown in ((self.profile_rows, profile), (self.hist_rows, not profile)):
            for label, field in rows:
                label.setVisible(shown)
                field.setVisible(shown)

    def load(self) -> None:
        s = self.settings
        self.profile_y.load(s.profile_y)
        self.hist_x.load(s.hist_x)
        self.hist_y.load(s.hist_y)
        for widget in (self.auto_bins, self.bins, self.log_counts):
            widget.blockSignals(True)
        self.auto_bins.setChecked(s.auto_bins)
        self.bins.setValue(s.bins)
        self.bins.setEnabled(not s.auto_bins)
        self.log_counts.setChecked(s.log_counts)
        for widget in (self.auto_bins, self.bins, self.log_counts):
            widget.blockSignals(False)

    def store(self) -> None:
        s = self.settings
        self.profile_y.store(s.profile_y)
        self.hist_x.store(s.hist_x)
        self.hist_y.store(s.hist_y)
        s.auto_bins = self.auto_bins.isChecked()
        s.bins = self.bins.value()
        s.log_counts = self.log_counts.isChecked()
        s.changed.emit()
