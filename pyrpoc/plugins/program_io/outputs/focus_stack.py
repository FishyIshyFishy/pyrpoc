"""``FocusStackOutput``: the images a focus search took, each with where it was taken and how
sharp it was, and where the search settled."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pyrpoc.core.program_output import program_output_registry
from pyrpoc.core.recording import Recording

from .building_blocks.tiff import read_pages
from .image import ImageOutput

# The ``paths`` key of the search's steps and result, beside one key per channel.
SEARCH_KEY = "search"


@dataclass(frozen=True)
class FocusStep:
    """One image of the search: its height, its sharpness, and the step that got there (0 for
    the start)."""

    z_um: float
    metric: float
    step_um: float


@dataclass(frozen=True)
class FocusResult:
    best_z_um: float
    best_metric: float
    # Still sharpening at the edge of the search range, so the focus may lie beyond it.
    at_edge: bool


@program_output_registry.register("focus_stack")
class FocusStackOutput(ImageOutput):
    """An ``ImageOutput`` whose image ``i`` was taken at ``steps[i]``. The search is saved as
    ``<root>_<name>_search.json`` after every step, so a search cut short still says where it
    went."""

    def __init__(self, *, name: str, recording: Recording, root: Path | None):
        super().__init__(name=name, recording=recording, root=root)
        self.steps: list[FocusStep] = []
        self.result: FocusResult | None = None

    def add_step(self, step: FocusStep) -> None:
        with self.lock:
            self.steps.append(step)
            self.write_search()
        self.changed()

    def settle(self, result: FocusResult) -> None:
        with self.lock:
            self.result = result
            self.write_search()
        self.changed()

    def write_search(self) -> None:
        if self.root is None:
            return
        path = self.root.with_name(f"{self.root.name}_{self.name}_search.json")
        search = {
            "steps": [asdict(step) for step in self.steps],
            "result": None if self.result is None else asdict(self.result),
        }
        path.write_text(json.dumps(search, indent=2), encoding="utf-8")
        self.paths[SEARCH_KEY] = path

    @classmethod
    def load(cls, *, name: str, recording: Recording, files: dict[str, Path]) -> FocusStackOutput:
        output = cls(name=name, recording=recording, root=None)
        channels = {key: path for key, path in files.items() if key != SEARCH_KEY}
        if channels:
            output.fill(read_pages(channels), list(channels))
        if SEARCH_KEY in files:
            search = json.loads(files[SEARCH_KEY].read_text(encoding="utf-8"))
            output.steps = [
                FocusStep(float(s["z_um"]), float(s["metric"]), float(s["step_um"]))
                for s in search["steps"]
            ]
            result = search["result"]
            if result is not None:
                output.result = FocusResult(
                    float(result["best_z_um"]),
                    float(result["best_metric"]),
                    bool(result["at_edge"]),
                )
        return output
