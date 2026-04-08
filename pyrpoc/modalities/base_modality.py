from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
import threading
from typing import Any

import numpy as np

from pyrpoc.backend_utils.array_contracts import CONTRACT_CHW_FLOAT32
from pyrpoc.backend_utils.contracts import ParameterGroups
from pyrpoc.backend_utils.parameter_utils import validate_parameter_groups
from pyrpoc.instruments.base_instrument import BaseInstrument
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl


class AcquisitionParameters:
    """Base class for all modality parameter dataclasses.

    Subclasses should be frozen dataclasses so that parameter state is
    immutable after configure() assigns them.

    Example:
        @dataclass(frozen=True)
        class ConfocalParameters(AcquisitionParameters):
            x_pixels: int
            ...
    """


class BaseModality(ABC):
    MODALITY_KEY: str = "base_modality"
    DISPLAY_NAME: str = "Base Modality"
    PARAMETERS: ParameterGroups = {}
    REQUIRED_INSTRUMENTS: list[type[BaseInstrument]] = []
    OPTIONAL_INSTRUMENTS: list[type[BaseInstrument]] = []
    ALLOWED_OPTOCONTROLS: list[type[BaseOptoControl]] = []
    OUTPUT_DATA_CONTRACT: str = CONTRACT_CHW_FLOAT32
    ALLOWED_DISPLAYS: list[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        required = getattr(cls, "REQUIRED_INSTRUMENTS", [])
        optional = getattr(cls, "OPTIONAL_INSTRUMENTS", [])
        allowed_opto = getattr(cls, "ALLOWED_OPTOCONTROLS", [])

        if not isinstance(required, list):
            raise TypeError("REQUIRED_INSTRUMENTS must be a list")
        if not isinstance(optional, list):
            raise TypeError("OPTIONAL_INSTRUMENTS must be a list")
        if not isinstance(allowed_opto, list):
            raise TypeError("ALLOWED_OPTOCONTROLS must be a list")

        for instrument_cls in [*required, *optional]:
            if not isinstance(instrument_cls, type) or not issubclass(instrument_cls, BaseInstrument):
                raise TypeError(
                    f"{cls.__name__} instrument requirements must contain BaseInstrument subclasses"
                )
        for optocontrol_cls in allowed_opto:
            if not isinstance(optocontrol_cls, type) or not issubclass(optocontrol_cls, BaseOptoControl):
                raise TypeError(
                    f"{cls.__name__} ALLOWED_OPTOCONTROLS must contain BaseOptoControl subclasses"
                )

        validate_parameter_groups(getattr(cls, "PARAMETERS", {}))

        output_contract = getattr(cls, "OUTPUT_DATA_CONTRACT", CONTRACT_CHW_FLOAT32)
        if not isinstance(output_contract, str) or not output_contract.strip():
            raise TypeError("OUTPUT_DATA_CONTRACT must be a non-empty string")

    def __init__(self):
        self._running = False
        self._configured = False
        self._warn_callback: Callable[[str], None] | None = None
        self.parameters: AcquisitionParameters | None = None
        # Storage state — shared across all DAQ modalities
        self._save_enabled = False
        self._save_root_path: Path | None = None
        self._save_json_path: Path | None = None
        self._save_channel_paths: dict[str, Path] = {}
        self._save_channel_labels: list[str] = []
        self._saved_frame_count = 0
        self._run_id = 0
        self._run_started_at = ""
        self._run_frame_limit: int | None = 1

    def _emit_warning(self, message: str) -> None:
        """Emit a non-fatal warning to the user via the service layer."""
        if self._warn_callback is not None:
            self._warn_callback(message)

    @classmethod
    def get_contract(cls) -> dict[str, Any]:
        return {
            "modality_key": cls.MODALITY_KEY,
            "display_name": cls.DISPLAY_NAME,
            "parameters": cls.PARAMETERS,
            "required_instruments": cls.REQUIRED_INSTRUMENTS,
            "optional_instruments": cls.OPTIONAL_INSTRUMENTS,
            "allowed_optocontrols": cls.ALLOWED_OPTOCONTROLS,
            "output_data_contract": cls.OUTPUT_DATA_CONTRACT,
            "allowed_displays": cls.ALLOWED_DISPLAYS,
        }

    # ------------------------------------------------------------------ #
    # Configure — template method; subclasses implement the 3 sub-steps  #
    # ------------------------------------------------------------------ #

    def configure(
        self,
        params: dict[str, Any],
        instruments: dict[type[BaseInstrument], BaseInstrument],
        opto_controls: list[BaseOptoControl],
    ) -> None:
        self.load_params(params)
        self.load_instruments(instruments)
        self.load_optocontrols(opto_controls)
        self.load_savepath()
    
        self._save_json_path = None
        self._save_channel_paths = {}
        self._save_channel_labels = []
        self._saved_frame_count = 0
        self._run_frame_limit = 1
        self._configured = True
    
    @abstractmethod
    def load_params(self, params: dict[str, Any]) -> None:
        """Build and assign self.parameters from the raw params dict."""
        raise NotImplementedError

    @abstractmethod
    def load_instruments(self, instruments: dict[type[BaseInstrument], BaseInstrument]) -> None:
        """Retrieve and validate required instruments from the instruments dict."""
        raise NotImplementedError

    def load_optocontrols(self, opto_controls: list[BaseOptoControl]) -> None:
        """Process optocontrols. Default is a no-op; override as needed."""

    def load_savepath(self):
        save_enabled = bool(getattr(self.parameters, "save_enabled", False))
        self._save_enabled = save_enabled
        if save_enabled:
            raw_path = getattr(self.parameters, "save_path", None)
            if raw_path is None:
                raise ValueError("save_path is required when save_enabled is true")
            path = Path(raw_path).expanduser() if not isinstance(raw_path, Path) else raw_path
            if path.suffix.lower() in {".tif", ".tiff"}:
                path = path.with_suffix("")
            self._save_root_path = path
        else:
            self._save_root_path = None


    # ------------------------------------------------------------------ #
    # Acquisition lifecycle                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def acquire_once(self) -> np.ndarray:
        """Acquire a single frame. Must be fully self-contained: set up
        hardware, acquire, and tear down before returning."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Signal continuous acquisition to stop and perform any cleanup
        needed for hardware that may have been interrupted mid-frame."""
        raise NotImplementedError

    def prepare_acquisition_storage(self, *, frame_limit: int | None) -> None:
        pass

    def save_acquired_frame(self, frame: np.ndarray, *, frame_index: int) -> None:
        pass

    def finalize_acquisition_storage(
        self,
        *,
        frame_count: int,
        frame_limit: int | None,
        error: Exception | None,
    ) -> None:
        pass

    def get_frame_limit(self) -> int | None:
        limit = int(self.parameters.num_frames)
        if limit < 1:
            raise ValueError("num_frames must be >= 1")
        return limit

    def get_active_channel_labels(self) -> list[str]:
        return []

    # ------------------------------------------------------------------ #
    # Shared frame utilities                                              #
    # ------------------------------------------------------------------ #

    def _split_channels(self, data: np.ndarray) -> list[np.ndarray]:
        if data.ndim == 2:
            return [data]
        if data.ndim == 3:
            return [data[index] for index in range(data.shape[0])]
        raise ValueError(f"unsupported frame dimensions {data.ndim}")

    def _resolve_channel_labels(self, channel_count: int) -> list[str]:
        active_labels = self.get_active_channel_labels()
        if active_labels and len(active_labels) == channel_count:
            return list(active_labels)
        return [f"channel_{index}" for index in range(channel_count)]

    def _parameters_as_dict(self) -> dict[str, Any]:
        """Serialize self.parameters to a plain dict for metadata/storage."""
        return asdict(self.parameters) if self.parameters is not None else {}

    # ------------------------------------------------------------------ #
    # Acquisition thread machinery                                        #
    # ------------------------------------------------------------------ #

    def acquire_continuous(
        self,
        on_frame: Callable[[np.ndarray], None],
        *,
        frame_limit: int | None = None,
        should_stop: Callable[[], bool],
        on_error: Callable[[Exception], None],
        on_finished: Callable[[int, Exception | None], None],
    ) -> threading.Thread:
        """Start continuous acquisition in a background thread.

        Loops acquire_once() until should_stop() returns True, frame_limit
        is reached, or an exception occurs. Subclasses may override to
        optimise setup/teardown across frames, but the default implementation
        works for any modality that implements acquire_once().
        """
        thread = threading.Thread(
            target=self._build_continuous_worker(on_frame, frame_limit, should_stop, on_error, on_finished),
            daemon=True,
        )
        thread.start()
        return thread

    def _build_continuous_worker(
        self,
        on_frame: Callable[[np.ndarray], None],
        frame_limit: int | None,
        should_stop: Callable[[], bool],
        on_error: Callable[[Exception], None],
        on_finished: Callable[[int, Exception | None], None],
    ) -> Callable[[], None]:
        """Returns the worker callable for the continuous acquisition thread.
        Override in a subclass to change how acquisition is driven.
        The returned callable must eventually call on_finished(count, error_or_none).
        """
        def worker() -> None:
            acquired = 0
            error_seen = None
            self._running = True
            try:
                while not should_stop():
                    frame = self.acquire_once()
                    acquired += 1
                    on_frame(frame)
                    if frame_limit is not None and acquired >= frame_limit:
                        break
            except Exception as exc:
                error_seen = exc
                on_error(exc)
            finally:
                self._running = False
                try:
                    self.stop()
                except Exception as stop_exc:
                    if error_seen is None:
                        error_seen = stop_exc
                        on_error(stop_exc)
                if on_finished is not None:
                    on_finished(acquired, error_seen)
        return worker
