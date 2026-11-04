"""Phase 2: the Library groups outputs into recordings and Experiments, and sorts them."""

from __future__ import annotations

import re
from pathlib import Path

from pyrpoc.core.saving import SaveTarget
from pyrpoc.engine.data_library.grouping import UNSAVED, SortBy, build_tree, sort_tree
from pyrpoc.engine.data_library.model import LibraryModel
from pyrpoc.engine.data_library.store import LibraryStore

from .helpers import FrameProgram, Harness, SpectrumProgram, saving_into
from .qt import qt_app

REPO = Path(__file__).resolve().parents[1]


def record(folder: Path, names: list[str]) -> None:
    """Save one run per name into ``folder``, alternating frames and spectra."""
    harness = Harness()
    for index, name in enumerate(names):
        program = FrameProgram() if index % 2 == 0 else SpectrumProgram()
        harness.run(program, f"test_{index}", saving_into(folder, name))
    assert harness.failures == []


def open_library() -> LibraryModel:
    return LibraryModel(LibraryStore(1 << 30), qt_app())


def test_opening_an_experiment_loads_all_its_recordings_under_one_group(tmp_path: Path) -> None:
    folder = tmp_path / "2026-11-03_beads"
    record(folder, ["alpha", "beta", "gamma"])
    library = open_library()
    library.load_experiment(folder)

    outputs = library.all()
    assert len(outputs) == 3
    assert {library.group_label(each) for each in outputs} == {"2026-11-03_beads"}
    [experiment] = build_tree(outputs, library.folders)
    assert experiment.folder == folder
    assert sorted(each.name for each in experiment.recordings) == ["alpha", "beta", "gamma"]


def test_opening_twice_does_not_duplicate(tmp_path: Path) -> None:
    folder = tmp_path / "exp"
    record(folder, ["alpha"])
    library = open_library()
    library.load_experiment(folder)
    library.load(folder / "alpha_meta.json")
    assert len(library.all()) == 1


def test_a_recording_opened_alone_appears_under_its_folder(tmp_path: Path) -> None:
    folder = tmp_path / "exp"
    record(folder, ["alpha", "beta"])
    library = open_library()
    library.load(folder / "beta_meta.json")
    [experiment] = build_tree(library.all(), library.folders)
    assert (experiment.label, [each.name for each in experiment.recordings]) == ("exp", ["beta"])


def test_an_empty_folder_still_shows(tmp_path: Path) -> None:
    library = open_library()
    library.load_experiment(tmp_path)
    [experiment] = build_tree(library.all(), library.folders)
    assert (experiment.folder, experiment.recordings) == (tmp_path, [])
    library.close_experiment(tmp_path)
    assert build_tree(library.all(), library.folders) == []


def test_unsaved_outputs_sit_in_their_own_group_on_top(tmp_path: Path) -> None:
    harness = Harness()
    harness.run(FrameProgram(), "test_frame", saving_into(tmp_path / "zeta", "saved"))
    harness.run(FrameProgram(), "test_frame", SaveTarget(name="loose", directory="", enabled=False))
    tree = sort_tree(build_tree(harness.library.all(), []), SortBy.NAME, False)
    assert [each.label for each in tree] == [UNSAVED, "zeta"]
    assert [each.name for each in tree[0].recordings] == ["loose"]
    assert harness.library.group_label(tree[0].recordings[0].outputs[0]) == UNSAVED


def test_sorting_by_name_and_by_date(tmp_path: Path) -> None:
    record(tmp_path / "b_second", ["m2", "a2"])
    record(tmp_path / "a_first", ["z1"])
    library = open_library()
    library.load_experiment(tmp_path / "b_second")
    library.load_experiment(tmp_path / "a_first")
    tree = build_tree(library.all(), library.folders)

    def shape(by: SortBy, descending: bool) -> list[tuple[str, list[str]]]:
        ordered = sort_tree(tree, by, descending)
        return [(e.label, [r.name for r in e.recordings]) for e in ordered]

    assert shape(SortBy.NAME, False) == [("a_first", ["z1"]), ("b_second", ["a2", "m2"])]
    assert shape(SortBy.NAME, True) == [("b_second", ["m2", "a2"]), ("a_first", ["z1"])]
    # Recorded in order m2, a2, z1: a_first holds the newest recording.
    assert shape(SortBy.DATE, True) == [("a_first", ["z1"]), ("b_second", ["a2", "m2"])]
    assert shape(SortBy.DATE, False) == [("b_second", ["m2", "a2"]), ("a_first", ["z1"])]


def test_core_and_plugins_know_nothing_of_folders_for_grouping() -> None:
    pattern = re.compile(r"meta_path|experiment_of|recordings_in|_meta\.json")
    offenders = [
        str(path.relative_to(REPO))
        for root in ("pyrpoc/core", "pyrpoc/plugins")
        for path in (REPO / root).rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
        # ProgramOutput carries where it was saved; nothing in core groups by it.
        and path.name != "program_output.py"
    ]
    assert offenders == []
