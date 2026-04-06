from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import threading
from typing import Any

import numpy as np

from pyrpoc.backend_utils.array_contracts import CONTRACT_CHW_FLOAT32
from pyrpoc.backend_utils.contracts import ParameterGroups
from pyrpoc.backend_utils.parameter_utils import validate_parameter_groups
from pyrpoc.instruments.base_instrument import BaseInstrument
from pyrpoc.optocontrols.base_optocontrol import BaseOptoControl


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
        self._params: dict[str, Any] = {}
        self._instruments: dict[type[BaseInstrument], BaseInstrument] = {}
        self._opto_controls: list[BaseOptoControl] = []

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

    @abstractmethod
    def configure(
        self,
        params: dict[str, Any],
        instruments: dict[type[BaseInstrument], BaseInstrument],
        opto_controls: list[BaseOptoControl],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def acquire_once(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
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

    @abstractmethod
    def get_frame_limit(self) -> int | None:
        raise NotImplementedError

    def get_active_channel_labels(self) -> list[str]:
        return []

    def run_acquisition_threaded(
        self,
        on_frame: Callable[[np.ndarray], None],
        *,
        frame_limit: int | None = None,
        should_stop: Callable[[], bool],
        on_error: Callable[[Exception], None],
        on_finished: Callable[[int, Exception | None], None],
    ) -> threading.Thread:
        thread = threading.Thread(
            target=self._build_worker(on_frame, frame_limit, should_stop, on_error, on_finished),
            daemon=True,
        )
        thread.start()
        return thread

    def _build_worker(
        self,
        on_frame: Callable[[np.ndarray], None],
        frame_limit: int | None,
        should_stop: Callable[[], bool],
        on_error: Callable[[Exception], None],
        on_finished: Callable[[int, Exception | None], None],
    ) -> Callable[[], None]:
        """Returns the worker callable for the acquisition thread.
        Override in a subclass to change how acquisition is driven
        (e.g. event-driven, streaming, callback-based SDK).
        The returned callable must eventually call on_finished(count, error_or_none).
        """
        def worker() -> None:
            acquired = 0
            error_seen = None
            try:
                self.start()
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
                try:
                    self.stop()
                except Exception as stop_exc:
                    if error_seen is None:
                        error_seen = stop_exc
                        on_error(stop_exc)
                if on_finished is not None:
                    on_finished(acquired, error_seen)
        return worker
