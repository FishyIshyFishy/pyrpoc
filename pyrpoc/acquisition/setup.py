from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Setup:
    """Once-per-run hardware arming, distinct from the command stream.

    ``run`` arms resources shared across commands (e.g. a TimeTagger
    measurement); ``teardown`` frees them. Default is a no-op.
    """

    def run(self) -> None:
        return None

    def teardown(self) -> None:
        return None


@dataclass
class FlimRunState:
    """Shared handle populated by FlimSetup and read by the FLIM command factory."""

    flim: object | None = None
    simulated: bool = False


class FlimSetup(Setup):
    """Arms a TimeTagger Flim measurement once per run.

    Falls back to a simulated run (``run_state.simulated=True``) if the tagger
    cannot be created, so the FLIM preset is still exercisable without hardware.
    """

    def __init__(self, *, run_state: FlimRunState, tagger_instrument: Any, config: dict[str, Any]):
        self.run_state = run_state
        self.tagger_instrument = tagger_instrument
        self.config = config

    def run(self) -> None:
        rs = self.run_state
        c = self.config
        tagger = self.tagger_instrument
        try:
            tagger.create_tagger()
            tagger.configure_for_flim(
                c["laser_channel"],
                c["detector_channel"],
                c["pixel_channel"],
                c["frame_channel"],
                c["laser_trigger_v"],
                c["detector_trigger_v"],
                c["pixel_trigger_v"],
                c["frame_trigger_v"],
                laser_input_delay_ps=c["laser_input_delay_ps"],
            )
            rs.flim = tagger.create_flim_measurement(
                c["laser_channel"],
                c["detector_channel"],
                c["pixel_channel"],
                c["frame_channel"],
                n_pixels=c["n_pixels"],
                n_bins=c["n_bins"],
                binwidth_ps=c["binwidth_ps"],
            )
            rs.simulated = False
        except Exception:
            rs.simulated = True
            rs.flim = None

    def teardown(self) -> None:
        rs = self.run_state
        if rs.flim is not None:
            try:
                rs.flim.stop()  # pyright:ignore
            except Exception:
                pass
            rs.flim = None
        try:
            self.tagger_instrument.free_tagger()
        except Exception:
            pass
