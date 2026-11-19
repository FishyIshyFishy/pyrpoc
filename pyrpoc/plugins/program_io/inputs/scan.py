"""The program inputs of a scan: the beam's path and pixels, the DAQ's rate and lines, the
status and static digital lines, the region scanned when no mask is drawn, the mosaic grid,
frames per run, and the focus search."""

from __future__ import annotations

from pyrpoc.core.program_input import declare
from pyrpoc.utils.value_rules import float_input, int_input

from .values.digital_line import DigitalLine, DigitalLineField

X_PIXELS = int_input("x_pixels", "X Pixels", 512, minimum=8, tooltip="Number of pixels in X")
Y_PIXELS = int_input("y_pixels", "Y Pixels", 512, minimum=8, tooltip="Number of pixels in Y")
EXTRA_LEFT = int_input(
    "extra_left",
    "Extra Steps Left",
    300,
    minimum=0,
    tooltip="Extra scan steps at the left edge",
)
EXTRA_RIGHT = int_input(
    "extra_right",
    "Extra Steps Right",
    20,
    minimum=0,
    tooltip="Extra scan steps at the right edge",
)
FAST_AXIS_OFFSET = float_input(
    "fast_axis_offset", "Fast Axis Offset (V)", 0.0, tooltip="Fast-axis offset, in volts"
)
FAST_AXIS_AMPLITUDE = float_input(
    "fast_axis_amplitude",
    "Fast Axis Amplitude (V)",
    1.0,
    minimum=1e-6,
    tooltip="Fast-axis amplitude, in volts",
)
SLOW_AXIS_OFFSET = float_input(
    "slow_axis_offset", "Slow Axis Offset (V)", 0.0, tooltip="Slow-axis offset, in volts"
)
SLOW_AXIS_AMPLITUDE = float_input(
    "slow_axis_amplitude",
    "Slow Axis Amplitude (V)",
    1.0,
    minimum=1e-6,
    tooltip="Slow-axis amplitude, in volts",
)
DWELL_TIME_US = float_input(
    "dwell_time_us", "Dwell Time (us)", 2.0, minimum=0.1, step=1.0, tooltip="Pixel dwell time"
)

SAMPLE_RATE_MHZ = float_input(
    "sample_rate_mhz",
    "Sample Rate (MHz)",
    1.0,
    minimum=0.000001,
    maximum=5.0,
    step=0.1,
    tooltip="DAQ AO sample rate in MHz; the FLIM pixel clock divides down from it",
)
FRAME_TRIGGER_PFI = int_input(
    "frame_trigger_pfi",
    "Frame Trigger PFI Line",
    0,
    minimum=0,
    tooltip="PFI line that exports the AO start trigger (frame marker)",
)
PIXEL_CLOCK_CTR = int_input(
    "pixel_clock_ctr",
    "Pixel Clock Counter",
    0,
    minimum=0,
    tooltip="Counter used to generate the pixel clock",
)
PIXEL_CLOCK_PFI = int_input(
    "pixel_clock_pfi",
    "Pixel Clock PFI Line",
    1,
    minimum=0,
    tooltip="PFI line that outputs the pixel clock",
)

# Status lines follow the scan on port 0; the defaults are the LabVIEW imager's lines.
ACTIVE_LINE = declare(
    "active_line",
    DigitalLineField("Active", "Drive this line high for the whole frame"),
    DigitalLine(0, 0, True),
)
IN_IMAGE_LINE = declare(
    "in_image_line",
    DigitalLineField(
        "In Image", "Drive this line high while the beam is on the image, not the extra steps"
    ),
    DigitalLine(0, 1, True),
)
NEW_LINE_LINE = declare(
    "new_line_line",
    DigitalLineField("New Line", "Drive this line high for the first samples of each line"),
    DigitalLine(0, 2, True),
)
NEW_PIXEL_LINE = declare(
    "new_pixel_line",
    DigitalLineField("New Pixel", "Drive this line high for the first half of each pixel"),
    DigitalLine(0, 3, True),
)
STATUS_LINES = (ACTIVE_LINE, IN_IMAGE_LINE, NEW_LINE_LINE, NEW_PIXEL_LINE)

# Static lines are set once before the scan and keep their level after it.
STATIC_LINES = tuple(
    declare(
        f"static_line_{index}",
        DigitalLineField(
            f"Static Line {index + 1}", "Checked, the line is held high; unchecked, low"
        ),
        DigitalLine(1, index, False),
    )
    for index in range(3)
)

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

X_TILES = int_input("x_tiles", "X Tiles", 3, minimum=1, tooltip="Tiles along the stage's x axis")
Y_TILES = int_input("y_tiles", "Y Tiles", 3, minimum=1, tooltip="Tiles along the stage's y axis")
# One spacing serves both axes; the display finds the overlap from the images.
TILE_SPACING_UM = float_input(
    "tile_spacing_um",
    "Spacing (um)",
    100.0,
    minimum=0.001,
    step=10.0,
    decimals=3,
    tooltip="Stage step between neighbouring tiles, in both x and y",
)

NUM_FRAMES = int_input("num_frames", "Frames", 1, minimum=1, tooltip="Number of frames to capture")

FOCUS_SEARCH_RANGE_UM = float_input(
    "focus_search_range_um",
    "Search Range (um)",
    50.0,
    minimum=0.001,
    step=10.0,
    decimals=3,
    tooltip="The search never moves z further than this from where it started",
)
FOCUS_INITIAL_STEP_UM = float_input(
    "focus_initial_step_um",
    "Initial Step (um)",
    5.0,
    minimum=0.001,
    step=1.0,
    decimals=3,
    tooltip="The first z step; it halves each time neither direction improves focus",
)
FOCUS_FINAL_STEP_UM = float_input(
    "focus_final_step_um",
    "Final Step (um)",
    0.5,
    minimum=0.001,
    step=0.1,
    decimals=3,
    tooltip="The search stops once the step would halve below this",
)
