"""A slider with two handles selecting a span within a range.

Two ``QSlider``s stacked with "Low" and "High" labels say what each one is
named; one slider with two handles says what they *do* -- everything outside
the span is excluded, and that is legible without reading a label, because the
groove is only accented between the handles.

It lives in views/ rather than shell/ because views/ may not import shell/.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import QSizePolicy, QWidget

_GROOVE_HEIGHT = 5
_HANDLE_RADIUS = 7
_LOW = 0
_HIGH = 1


class RangeSlider(QWidget):
    """Horizontal slider with a low and a high handle over one groove."""

    #: Low and high, after clamping. Emitted whenever either moves.
    values_changed = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 100
        self._low = 0
        self._high = 100
        self._pressed: int | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        return QSize(160, 2 * _HANDLE_RADIUS + 6)

    def minimumSizeHint(self) -> QSize:
        return QSize(4 * _HANDLE_RADIUS, 2 * _HANDLE_RADIUS + 6)

    def minimum(self) -> int:
        return self._minimum

    def maximum(self) -> int:
        return self._maximum

    def low(self) -> int:
        return self._low

    def high(self) -> int:
        return self._high

    def setRange(self, minimum: int, maximum: int) -> None:
        self._minimum = int(minimum)
        self._maximum = max(int(maximum), int(minimum))
        self.setValues(self._low, self._high)

    def setLow(self, value: int) -> None:
        self.setValues(value, self._high)

    def setHigh(self, value: int) -> None:
        self.setValues(self._low, value)

    def setValues(self, low: int, high: int) -> None:
        low = self._clamp(low)
        high = self._clamp(high)
        if high < low:
            low, high = high, low
        if (low, high) == (self._low, self._high):
            self.update()
            return
        self._low = low
        self._high = high
        self.update()
        self.values_changed.emit(self._low, self._high)

    def _clamp(self, value: int) -> int:
        return max(self._minimum, min(self._maximum, int(value)))

    def _span(self) -> int:
        return max(self._maximum - self._minimum, 1)

    def _track(self) -> QRect:
        return QRect(
            _HANDLE_RADIUS,
            0,
            max(self.width() - 2 * _HANDLE_RADIUS, 1),
            self.height(),
        )

    def _x_for(self, value: int) -> int:
        track = self._track()
        fraction = (value - self._minimum) / self._span()
        return track.left() + round(fraction * (track.width() - 1))

    def _value_for(self, x: int) -> int:
        track = self._track()
        fraction = (x - track.left()) / max(track.width() - 1, 1)
        return self._clamp(round(self._minimum + fraction * self._span()))

    def _handle_at(self, pos: QPoint) -> int:
        """Which handle a click grabs.

        Distance decides, so a click between the handles drags the nearer one;
        a tie goes to whichever handle the click can still move, which matters
        when both sit on the same value at an end of the range.
        """
        low_dx = abs(pos.x() - self._x_for(self._low))
        high_dx = abs(pos.x() - self._x_for(self._high))
        if low_dx < high_dx:
            return _LOW
        if high_dx < low_dx:
            return _HIGH
        return _HIGH if pos.x() >= self._x_for(self._high) else _LOW

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._pressed = self._handle_at(event.pos())
        self._drag_to(event.pos())
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None or self._pressed is None:
            super().mouseMoveEvent(event)
            return
        self._drag_to(event.pos())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None or self._pressed is None:
            super().mouseReleaseEvent(event)
            return
        self._pressed = None
        event.accept()

    def _drag_to(self, pos: QPoint) -> None:
        """Move the held handle, letting it push past the other one.

        Swapping (rather than blocking at the other handle) means a drag never
        stalls: carry the low handle past the high one and it becomes the high
        one, which is what the span under the cursor says should happen.
        """
        value = self._value_for(pos.x())
        if self._pressed == _LOW:
            if value > self._high:
                self._pressed = _HIGH
                self.setValues(self._high, value)
                return
            self.setValues(value, self._high)
            return
        if value < self._low:
            self._pressed = _LOW
            self.setValues(value, self._low)
            return
        self.setValues(self._low, value)

    def keyPressEvent(self, event) -> None:
        if event is None:
            return
        step = 10 if event.modifiers() & Qt.KeyboardModifier.ControlModifier else 1
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self.setValues(self._low - step, self._high)
            return
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self.setValues(self._low, self._high + step)
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        accent = palette.color(palette.ColorRole.Highlight)
        if not self.isEnabled():
            accent = QColor(accent.red(), accent.green(), accent.blue(), 110)

        track = self._track()
        groove_y = (self.height() - _GROOVE_HEIGHT) // 2
        groove = QRect(track.left(), groove_y, track.width(), _GROOVE_HEIGHT)
        radius = _GROOVE_HEIGHT / 2

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(palette.color(palette.ColorRole.Mid))
        painter.drawRoundedRect(groove, radius, radius)

        low_x = self._x_for(self._low)
        high_x = self._x_for(self._high)
        painter.setBrush(accent)
        painter.drawRoundedRect(
            QRect(low_x, groove_y, max(high_x - low_x, 1), _GROOVE_HEIGHT), radius, radius
        )

        centre_y = self.height() // 2
        painter.setPen(palette.color(palette.ColorRole.Mid))
        painter.setBrush(palette.color(palette.ColorRole.Button).darker(140))
        for x in (low_x, high_x):
            painter.drawEllipse(QPoint(x, centre_y), _HANDLE_RADIUS, _HANDLE_RADIUS)
        painter.end()
