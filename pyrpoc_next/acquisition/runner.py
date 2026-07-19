"""The shared async run engine.

Runs one configured modality, handing it an emit callback, a stop check, and (later)
a feedback channel. Owns the worker thread and lifecycle so no modality reimplements
threading. Also runs synchronously, which the tests use.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from attrs import define, field

from pyrpoc_next.structs.parcels import Parcel


@define
class RunContext:
    """What a modality's run receives: how to emit, whether to stop, how far to go."""

    emit: Callable[[Parcel], None]
    should_stop: Callable[[], bool]
    frame_limit: int | None = None
    feedback: object | None = field(default=None)  # reserved for display->modality events


class Runner:
    """Drives a modality on a worker thread (or synchronously) and pumps parcels to a sink."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def run_sync(self, modality, sink: Callable[[Parcel], None], frame_limit: int | None = None) -> None:
        """Run to completion on the calling thread. Used by tests and headless runs."""
        self.stop_event.clear()
        context = RunContext(emit=sink, should_stop=self.stop_event.is_set, frame_limit=frame_limit)
        modality.run(context)

    def start(self, modality, sink: Callable[[Parcel], None], frame_limit: int | None = None,
              on_finished: Callable[[Exception | None], None] | None = None) -> None:
        """Run on a daemon worker thread; report completion (and any error) via on_finished."""
        self.stop_event.clear()
        context = RunContext(emit=sink, should_stop=self.stop_event.is_set, frame_limit=frame_limit)

        def worker():
            error: Exception | None = None
            try:
                modality.run(context)
            except Exception as exc:
                error = exc
            finally:
                if on_finished is not None:
                    on_finished(error)

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()
