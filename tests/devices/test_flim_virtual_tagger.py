"""TimeTagger FLIM behaviour tests.

These run the *real* Swabian Flim measurement over a synthetic, fully
deterministic photon stream built on a virtual TimeTagger — no hardware needed.
They document and verify the assumptions our FLIM implementation depends on:

  * lifetime resolution comes from the laser(start)->photon(click) time
    difference, measured by the TDC, NOT from the pixel clock;
  * the pixel clock only assigns photons to pixels, so it needs no fine timing;
  * the frame marker resets the pixel index each frame;
  * the Flim measurement histograms on the FPGA and returns an
    (n_pixels, n_bins) array — the 80 MHz laser is never streamed.

If the virtual-tagger stack is unavailable they skip, so the rest of the suite
still runs on machines without the Swabian SDK. Run this suite in the lab (or
here, since the SDK is installed) before trusting an implementation change.
"""

from __future__ import annotations

import numpy as np
import pytest

TT = pytest.importorskip("TimeTagger")

_flim_stack_ok = (
    hasattr(TT, "Flim")
    and hasattr(TT, "EventGenerator")
    and hasattr(TT, "Experimental")
    and hasattr(TT.Experimental, "PatternSignalGenerator")
    and TT.hasTimeTaggerVirtualLicense()
)

pytestmark = pytest.mark.skipif(
    not _flim_stack_ok, reason="Swabian virtual TimeTagger FLIM stack unavailable"
)

from pyrpoc.devices.time_tagger.device import TimeTagger as TimeTaggerDevice
from pyrpoc.programs.hardware.tagger import flim_intensity, reshape_flim_frame


laser_period_ps = 12_500  # 80 MHz


class SyntheticFlim:
    """Owns a virtual tagger plus the generators that synthesize a FLIM stream.

    The generator objects must stay referenced or their virtual channels are
    freed, so they are held on the instance.
    """

    def __init__(
        self,
        x_pixels,
        y_pixels,
        extra_left=0,
        extra_right=0,
        delay_ps=2_000,
        photon_divider=4,
        laser_per_pixel=40,
        n_bins=30,
        binwidth_ps=100,
    ):
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.extra_left = extra_left
        self.extra_right = extra_right
        self.total_x = x_pixels + extra_left + extra_right
        self.n_pixels = self.total_x * y_pixels
        self.delay_ps = delay_ps
        self.photon_divider = photon_divider
        self.laser_per_pixel = laser_per_pixel
        self.n_bins = n_bins
        self.binwidth_ps = binwidth_ps
        self.pixel_dwell_ps = laser_per_pixel * laser_period_ps
        self.frame_period_ps = self.n_pixels * self.pixel_dwell_ps
        self._gens = []
        self.tagger = TT.createTimeTaggerVirtual()
        self._build_channels()

    def _pattern(self, period_ps, start_delay=0):
        gen = TT.Experimental.PatternSignalGenerator(
            self.tagger, sequence=[int(period_ps)], repeat=True, start_delay=int(start_delay)
        )
        self._gens.append(gen)
        return gen.getChannel()

    def _build_channels(self):
        self.laser_ch = self._pattern(laser_period_ps)
        det = TT.EventGenerator(
            self.tagger, trigger_channel=self.laser_ch,
            pattern=[self.delay_ps], trigger_divider=self.photon_divider,
        )
        self._gens.append(det)
        self.det_ch = det.getChannel()
        # frame marker sits a hair before the aligned pixel marker
        self.frame_ch = self._pattern(self.frame_period_ps, start_delay=50)
        self.pixel_ch = self._pattern(self.pixel_dwell_ps, start_delay=100)

    def run_and_read(self, n_frames=5):
        device = TimeTaggerDevice()
        device.tagger = self.tagger
        device.config.laser_channel = self.laser_ch
        device.config.detector_channel = self.det_ch
        device.config.pixel_channel = self.pixel_ch
        device.config.frame_channel = self.frame_ch
        flim = device.start_flim_measurement(
            n_pixels=self.n_pixels, n_bins=self.n_bins, binwidth_ps=self.binwidth_ps,
        )
        self.tagger.run()
        flim.startFor(int((n_frames + 2) * self.frame_period_ps))
        flim.waitUntilFinished()

        frames_acquired = flim.getFramesAcquired()
        ready_frame = flim.getReadyFrameEx()
        histograms = ready_frame.getHistograms()
        ready = np.asarray(histograms, dtype=np.float64)
        cube = reshape_flim_frame(
            histograms, self.n_bins,
            self.y_pixels, self.total_x, self.extra_left, self.x_pixels,
        )
        flim.stop()
        return frames_acquired, ready, cube

    def free(self):
        self._gens.clear()
        TT.freeTimeTagger(self.tagger)


@pytest.fixture(scope="module")
def basic_run():
    setup = SyntheticFlim(x_pixels=4, y_pixels=3)
    frames, ready, cube = setup.run_and_read()
    expected_peak_bin = setup.delay_ps // setup.binwidth_ps
    expected_per_pixel = setup.laser_per_pixel / setup.photon_divider
    setup.free()
    return {
        "frames": frames,
        "ready_flat": ready,
        "cube": cube,
        "x_pixels": 4,
        "y_pixels": 3,
        "n_bins": setup.n_bins,
        "expected_peak_bin": expected_peak_bin,
        "expected_per_pixel": expected_per_pixel,
    }


def test_flim_measurement_returns_per_pixel_histograms(basic_run):
    # FPGA output is (n_pixels, n_bins): a decay histogram per pixel.
    assert basic_run["ready_flat"].shape == (
        basic_run["x_pixels"] * basic_run["y_pixels"],
        basic_run["n_bins"],
    )
    assert basic_run["cube"].shape == (
        basic_run["y_pixels"], basic_run["x_pixels"], basic_run["n_bins"]
    )


def test_lifetime_lands_in_correct_bin_independent_of_pixel(basic_run):
    # Every illuminated pixel peaks at the bin matching the laser->photon delay,
    # proving lifetime timing is independent of the (coarse) pixel clock.
    cube = basic_run["cube"]
    flat = cube.reshape(-1, basic_run["n_bins"])
    lit = flat[flat.sum(axis=1) > 0]
    assert lit.shape[0] == flat.shape[0]  # all pixels received photons
    peak_bins = np.argmax(lit, axis=1)
    assert np.all(peak_bins == basic_run["expected_peak_bin"])


def test_intensity_is_photon_count_and_uniform(basic_run):
    # Uniform illumination -> roughly equal counts per pixel; the pixel clock
    # partitions photons correctly with no smearing.
    intensity = flim_intensity(basic_run["cube"])
    assert intensity.shape == (basic_run["y_pixels"], basic_run["x_pixels"])
    expected = basic_run["expected_per_pixel"]
    assert intensity.min() > 0
    assert intensity.max() / intensity.min() < 1.5
    assert abs(intensity.mean() - expected) < 0.4 * expected


def test_ready_frame_is_one_frame_not_accumulated(basic_run):
    # getReadyFrame returns a single completed frame; the frame marker resets
    # the accumulators each frame instead of summing the whole run.
    assert basic_run["frames"] > 1
    total = basic_run["ready_flat"].sum()
    expected_total = basic_run["expected_per_pixel"] * basic_run["x_pixels"] * basic_run["y_pixels"]
    assert abs(total - expected_total) < 0.4 * expected_total


def test_overscan_columns_are_clipped():
    setup = SyntheticFlim(x_pixels=3, y_pixels=2, extra_left=2, extra_right=1)
    _frames, ready, cube = setup.run_and_read()
    setup.free()
    assert ready.shape[0] == setup.total_x * setup.y_pixels  # full scan incl. overscan
    assert cube.shape == (2, 3, setup.n_bins)                 # clipped to the ROI


def test_raw_laser_rate_would_overflow_usb():
    """Why we must NOT stream raw tags: the 80 MHz laser sync alone exceeds the
    TimeTagger USB rate. The Flim measurement avoids this by histogramming on
    the FPGA."""
    t = TT.createTimeTaggerVirtual()
    gen = TT.Experimental.PatternSignalGenerator(t, sequence=[laser_period_ps], repeat=True)
    laser_ch = gen.getChannel()
    stream = TT.TimeTagStream(t, 2_000_000, [laser_ch])
    t.run()
    stream.startFor(int(1000 * laser_period_ps))  # 1000 laser periods of virtual time
    stream.waitUntilFinished()
    data = stream.getData()
    ts = np.asarray(data.getTimestamps())
    TT.freeTimeTagger(t)

    assert ts.size > 100
    periods = np.diff(ts)
    assert np.all(periods == laser_period_ps)            # clean 80 MHz
    rate_hz = 1e12 / laser_period_ps
    tt20_usb_limit = 8e6
    assert rate_hz > tt20_usb_limit                      # 80 Mtags/s >> 8 Mtags/s
