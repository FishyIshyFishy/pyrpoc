"""``ProgramOutput``: one named part of a recording, of one type that owns its content and its
files, and ``program_output_registry``, every type by name."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .recording import Recording
from .registry import Registry

OutputT = TypeVar("OutputT", bound="ProgramOutput")


class ProgramOutput:
    """One named part of a recording. Each type is a subclass that holds its own content, says
    how programs add to it and panels read it, and writes it to files and reads it back."""

    # Set by ``program_output_registry``; a recording on disk names its outputs' types by it.
    type_key: str = "program_output"

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        self.name = name
        self.recording = recording
        # Where its files go, beside the recording's; None when it is not saved.
        self.root = root
        # What it wrote, keyed as ``load`` expects them back.
        self.paths: dict[str, Path] = {}
        # Content is added on a run's thread and read on the GUI thread.
        self.lock = threading.RLock()
        self._subscribers: list[Callable[[ProgramOutput], None]] = []

    @property
    def nbytes(self) -> int:
        """Memory held, in bytes."""
        raise NotImplementedError

    @classmethod
    def load(
        cls: type[OutputT], *, name: str, recording: Recording, files: dict[str, Path]
    ) -> OutputT:
        """The output written to ``files``, keyed as ``paths`` was."""
        raise NotImplementedError

    def close(self, error: Exception | None) -> None:
        """Finish its files: nothing more will be added."""
        del error

    def subscribe(self, callback: Callable[[ProgramOutput], None]) -> None:
        with self.lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[ProgramOutput], None]) -> None:
        with self.lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def changed(self) -> None:
        """Tell every subscriber its content changed."""
        # Called on the thread that changed it, so subscribers must not touch Qt.
        with self.lock:
            listeners = list(self._subscribers)
        for callback in listeners:
            callback(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}>"


# Every type of program output, by ``type_key``, so a recording on disk finds its classes again.
program_output_registry: Registry[ProgramOutput] = Registry(
    "ProgramOutputRegistry", ProgramOutput, stamp="type_key"
)
