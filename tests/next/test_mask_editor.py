"""Mask editor: ROI -> binary mask, and the display-source dialog."""

from __future__ import annotations

import numpy as np


def test_editor_generates_binary_mask_from_roi(qapp):
    from pyrpoc_next.gui.panels.mask_editor import MaskEditorWidget

    # a single 128x128 channel, uniform so the threshold passes everywhere
    data = np.full((1, 128, 128), 100.0, dtype=np.float32)
    editor = MaskEditorWidget(image_data=data)
    editor.low_spin.setValue(0)
    editor.high_spin.setValue(200)  # threshold window covers the value 100

    # a triangle ROI in the top-left quadrant
    editor.add_roi([(10.0, 10.0), (60.0, 10.0), (10.0, 60.0)])
    mask = editor.generate_mask()

    assert mask is not None
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})  # strictly binary
    assert mask[15, 15] == 255  # inside the triangle + threshold
    assert mask[100, 100] == 0  # outside the ROI


def test_editor_returns_none_without_roi(qapp):
    from pyrpoc_next.gui.panels.mask_editor import MaskEditorWidget

    editor = MaskEditorWidget(image_data=np.zeros((1, 16, 16), dtype=np.float32))
    assert editor.generate_mask() is None


def test_dialog_loads_display_data(qapp):
    from pyrpoc_next.gui.displays import StreamedDisplay
    from pyrpoc_next.gui.panels.mask_editor import MaskEditorDialog
    from pyrpoc_next.structs.parcels import ImageFrameParcel

    display = StreamedDisplay()
    display.handle(ImageFrameParcel(data=np.random.rand(1, 32, 48).astype(np.float32),
                                    channel_labels=["A"]))
    dialog = MaskEditorDialog([display])
    assert dialog.source_combo.count() == 1  # the display is offered as a source
    # editor picked up the display's 32x48 geometry
    assert editor_shape(dialog.editor) == (32, 48)


def editor_shape(editor) -> tuple[int, int]:
    return (editor._h, editor._w)
