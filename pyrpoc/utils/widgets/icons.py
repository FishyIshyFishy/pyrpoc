"""Icons from ``utils/assets/``, looked up by name and drawn in the theme's
button text colour."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QByteArray, QPoint, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QGuiApplication, QIcon, QIconEngine, QPainter, QPalette, QPixmap
from PyQt6.QtSvg import QSvgRenderer

# ``pyrpoc/utils/assets``, beside this package.
ASSETS = Path(__file__).resolve().parents[1] / "assets"


class PaletteSvgEngine(QIconEngine):
    """Renders an SVG with ``currentColor`` as the current palette's button
    text, the disabled shade when the icon is drawn disabled."""

    def __init__(self, svg: bytes):
        super().__init__()
        self.svg = svg

    def paint(self, painter: QPainter | None, rect: QRect, mode: QIcon.Mode, state: QIcon.State):
        del state
        group = (
            QPalette.ColorGroup.Disabled
            if mode == QIcon.Mode.Disabled
            else QPalette.ColorGroup.Active
        )
        color = QGuiApplication.palette().color(group, QPalette.ColorRole.ButtonText)
        svg = self.svg.replace(b"currentColor", color.name().encode())
        QSvgRenderer(QByteArray(svg)).render(painter, QRectF(rect))

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(
        self, size: QSize, mode: QIcon.Mode, state: QIcon.State, scale: float
    ) -> QPixmap:
        # Rendered at device pixels; QIcon sets the pixmap's ratio from its size.
        pixmap = QPixmap(size * scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(QPoint(0, 0), pixmap.size()), mode, state)
        painter.end()
        return pixmap

    def clone(self) -> QIconEngine:
        return PaletteSvgEngine(self.svg)


def asset_icon(name: str) -> QIcon:
    """The icon ``<name>.svg``, or ``<name>.png``, from ``utils/assets/``."""
    # A missing file is a packaging bug, so it raises rather than falling back.
    svg = ASSETS / f"{name}.svg"
    if svg.is_file():
        return QIcon(PaletteSvgEngine(svg.read_bytes()))
    png = ASSETS / f"{name}.png"
    if png.is_file():
        return QIcon(str(png))
    raise FileNotFoundError(f"no icon named {name!r} in {ASSETS}")
