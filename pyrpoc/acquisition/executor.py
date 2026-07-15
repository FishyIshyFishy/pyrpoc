from __future__ import annotations

from collections.abc import Callable

from pyrpoc.structs.acquired_data import AcquiredData
from pyrpoc.acquisition.source import CommandSource
from pyrpoc.acquisition.setup import Setup
from pyrpoc.acquisition.handlers import HandlerRegistry, command_handler_registry


class Executor:
    """The one fixed acquisition loop.

    Runs setup once, then pulls commands from the active source, dispatches each
    to its handler, feeds the results back to the source (for closed-loop
    decorators) and out to ``on_results`` (save + display), until the source is
    exhausted or ``should_stop`` returns True. Backend-agnostic: it only ever
    calls ``next_command`` and looks up a handler by command type.
    """

    def __init__(self, handlers: HandlerRegistry | None = None):
        self.handlers = handlers or command_handler_registry

    def run(
        self,
        *,
        source: CommandSource,
        setup: Setup,
        on_results: Callable[[list[AcquiredData]], None],
        should_stop: Callable[[], bool],
        on_finished: Callable[[Exception | None], None],
    ) -> None:
        error: Exception | None = None
        setup.run()
        try:
            while not should_stop():
                command = source.next_command()
                if command is None:
                    break
                handler = self.handlers.handler_for(type(command))
                results = handler.run(command)
                source.observe_result(command, results)
                on_results(results)
        except Exception as exc:
            error = exc
        finally:
            try:
                setup.teardown()
            except Exception as teardown_exc:
                if error is None:
                    error = teardown_exc
            on_finished(error)
