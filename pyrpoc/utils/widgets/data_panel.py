"""``DataPanel``: a panel showing one open program output of the type it renders, and its
registry."""

from __future__ import annotations

from typing import Any, ClassVar, Generic

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from pyrpoc.core.library import Library
from pyrpoc.core.panels import Panel
from pyrpoc.core.program_input import ProgramInputs
from pyrpoc.core.registry import Registry
from pyrpoc.core.requests import Requests

from .source_picker import OutputT, SourcePicker


class DataPanel(Panel, Generic[OutputT]):
    """A panel showing one open program output of the type it ``renders``: a source
    picker above a ``body`` the subclass fills through ``draw``."""

    # The one type of program output this panel shows.
    renders: type[OutputT]

    # The heading it is listed under, and its place in the Add menu.
    group: str
    order: int
    # Listed under Add Misc. Panel rather than Add Display: a tool that works
    # on data, not a way to look at it.
    misc: ClassVar[bool] = False

    def __init__(self, library: Library, inputs: ProgramInputs, requests: Requests):
        super().__init__()
        self.inputs = inputs
        self.requests = requests
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self.body = QWidget(self)
        self.source = SourcePicker(self.renders, library, self)
        root.addWidget(self.source)
        root.addWidget(self.body, 1)
        self.shown_output: OutputT | None = None

    def connect_source(self) -> None:
        """Start redrawing on source changes."""
        # Called once the subclass's widgets exist, so a refresh never meets a
        # half-built panel.
        self.source.changed.connect(self.refresh)
        self.refresh()

    def detach(self) -> None:
        """Stop following the library."""
        self.library.unsubscribe(self.source.refresh_sources)

    @property
    def library(self) -> Library:
        return self.source.library

    def program_output(self) -> OutputT | None:
        return self.source.current()

    def refresh(self) -> None:
        program_output = self.program_output()
        self.shown_output = program_output
        if program_output is None:
            self.clear()
            return
        self.draw(program_output)

    def draw(self, program_output: OutputT) -> None:
        """Draw ``program_output``. One that holds nothing yet, as when a run has just started,
        should leave what is on screen, so levels and layout carry over."""
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


data_panel_registry: Registry[DataPanel[Any]] = Registry("DataPanelRegistry", DataPanel)
