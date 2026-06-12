"""Per-channel tile widget for the tiled display.

A tile is the image + histogram + autoscale unit shown for one channel. The
builder here only constructs widgets; the display owns signal wiring so the
analysis controller can attach ROIs to ``tile.plot``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget


@dataclass
class ChannelTile:
    root: QWidget
    name_edit: QLineEdit
    autoscale_box: QCheckBox
    plot: pg.PlotWidget
    image_item: pg.ImageItem
    hist_widget: pg.HistogramLUTWidget
    min_val: float = 0.0
    max_val: float = 1.0


channel_palette = [
    (255, 80, 80),
    (80, 220, 120),
    (70, 150, 255),
    (255, 200, 70),
    (190, 110, 255),
    (70, 230, 230),
    (255, 120, 210),
    (180, 180, 180),
]


def channel_color(index: int) -> tuple[int, int, int]:
    return channel_palette[index % len(channel_palette)]


def make_default_lut() -> pg.ColorMap:
    return pg.ColorMap(
        pos=np.array([0.0, 0.999, 1.0], dtype=float),
        color=np.array(
            [
                [0, 0, 0, 255],
                [255, 255, 255, 255],
                [255, 0, 0, 255],
            ],
            dtype=np.ubyte,
        ),
    )


def build_channel_tile(parent: QWidget, index: int, lut: pg.ColorMap) -> ChannelTile:
    root = QWidget(parent)
    root_layout = QVBoxLayout(root)
    root_layout.setContentsMargins(6, 6, 6, 6)

    name_edit = QLineEdit(root)
    name_edit.setText(f"Input {index + 1}")
    root_layout.addWidget(name_edit)

    body = QHBoxLayout()
    root_layout.addLayout(body, 1)

    plot = pg.PlotWidget(root)
    plot.setMenuEnabled(False)
    plot.hideButtons()
    plot.setAspectLocked(True)
    plot.invertY(True)
    image_item = pg.ImageItem()
    image_item.setColorMap(lut)
    plot.addItem(image_item)
    body.addWidget(plot, 1)

    hist_widget = pg.HistogramLUTWidget(root)
    hist_widget.setImageItem(image_item)
    hist_widget.item.gradient.setColorMap(lut)
    right_col = QVBoxLayout()
    right_col.setContentsMargins(0, 0, 0, 0)
    right_col.setSpacing(4)
    right_col.addWidget(hist_widget, 1)

    autoscale_box = QCheckBox("Autoscale", root)
    autoscale_box.setChecked(True)
    right_col.addWidget(autoscale_box)
    body.addLayout(right_col)

    return ChannelTile(
        root=root,
        name_edit=name_edit,
        autoscale_box=autoscale_box,
        plot=plot,
        image_item=image_item,
        hist_widget=hist_widget,
    )
