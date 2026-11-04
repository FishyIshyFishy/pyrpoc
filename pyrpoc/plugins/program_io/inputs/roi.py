"""The centred rectangle a region scan covers when no mask is drawn."""

from __future__ import annotations

from pyrpoc.utils.value_rules import float_input

CENTER_REGION_FRACTION = float_input(
    "center_region_fraction",
    "Center Fraction",
    0.25,
    minimum=0.01,
    maximum=1.0,
    step=0.05,
    decimals=3,
    tooltip="Width and height of the centred rectangle, as a fraction of the scan, "
    "when no mask is drawn",
)
