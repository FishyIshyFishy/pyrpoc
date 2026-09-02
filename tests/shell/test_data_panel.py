"""The data panel: the table of acquisitions, and the dock it shares.

Headless via QT_QPA_PLATFORM=offscreen (see tests/conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.core.streams import Cube3D, Image2D
from pyrpoc.data.dataset import Dataset, Provenance
from pyrpoc.shell.app import Application
from pyrpoc.shell.data_panel import FRAMES, NAME, STREAM, TIME, DataPanel


@pytest.fixture
def app(qapp):
    return Application()


def add(app, *, stream="intensity", spec=Image2D, name="cells", started="2026-09-02T14:32:07+00:00"):
    return app.library.add(
        Dataset(
            stream=stream,
            spec=spec,
            provenance=Provenance("simulation", started_at=started, run_id=1, name=name),
        )
    )


def cell(panel, row, column) -> str:
    return panel.table.item(row, column).text()


# --- the table --------------------------------------------------------------


def test_an_empty_library_shows_the_hint_and_no_table(app):
    """isHidden rather than isVisible: nothing here is inside a shown window."""
    panel = DataPanel(app)
    assert panel.table.isHidden() is True
    assert panel.empty_label.isHidden() is False
    assert panel.close_btn.isHidden() is True
    assert panel.table.rowCount() == 0


def test_a_row_carries_the_time_the_name_and_the_stream(app):
    dataset = add(app)
    panel = DataPanel(app)

    assert panel.table.rowCount() == 1
    assert cell(panel, 0, TIME) == dataset.started_time
    assert cell(panel, 0, NAME) == "cells"
    assert cell(panel, 0, STREAM) == "intensity"
    assert cell(panel, 0, FRAMES) == "0"


def test_the_time_column_carries_no_date(app):
    add(app)
    panel = DataPanel(app)
    assert cell(panel, 0, TIME).count(":") == 2
    assert "2026" not in cell(panel, 0, TIME)


def test_the_newest_acquisition_is_the_top_row(app):
    """So the one just taken is on screen without scrolling."""
    add(app, name="first")
    add(app, name="second")
    panel = DataPanel(app)
    assert [cell(panel, row, NAME) for row in range(2)] == ["second", "first"]


def test_one_run_with_two_streams_is_two_rows_told_apart_by_stream(app):
    add(app, stream="intensity")
    add(app, stream="histogram", spec=Cube3D)
    panel = DataPanel(app)
    assert {cell(panel, row, STREAM) for row in range(2)} == {"intensity", "histogram"}
    assert {cell(panel, row, NAME) for row in range(2)} == {"cells"}


def test_an_unsaved_run_still_has_a_name(app):
    """The name is not a path -- nothing has to be written for it to exist."""
    add(app, name="cells")
    panel = DataPanel(app)
    assert cell(panel, 0, NAME) == "cells"


# --- staying current --------------------------------------------------------


def test_a_new_acquisition_appears(app):
    panel = DataPanel(app)
    add(app)
    assert panel.table.rowCount() == 1
    assert panel.table.isHidden() is False
    assert panel.empty_label.isHidden() is True


def test_the_frame_count_follows_a_growing_run(app):
    dataset = add(app)
    panel = DataPanel(app)
    for _ in range(3):
        dataset.append(np.zeros((1, 2, 2), np.float32))
        app.bridge.dataset_changed.emit(dataset, len(dataset) - 1)
    assert cell(panel, 0, FRAMES) == "3"


def test_a_growing_run_does_not_clear_the_selection(app):
    """A rebuild per frame would take the row out from under a click."""
    dataset = add(app)
    panel = DataPanel(app)
    panel.table.selectRow(0)

    dataset.append(np.zeros((1, 2, 2), np.float32))
    app.bridge.dataset_changed.emit(dataset, 0)
    assert panel.selected_dataset() is dataset


def test_the_selection_survives_a_new_acquisition_arriving(app):
    first = add(app, name="first")
    panel = DataPanel(app)
    panel.table.selectRow(0)
    add(app, name="second")
    assert panel.selected_dataset() is first


# --- closing ----------------------------------------------------------------


def test_close_is_only_offered_for_a_selected_row(app):
    add(app)
    panel = DataPanel(app)
    assert panel.close_btn.isEnabled() is False
    panel.table.selectRow(0)
    assert panel.close_btn.isEnabled() is True


def test_closing_drops_the_acquisition_from_the_library(app):
    add(app)
    panel = DataPanel(app)
    panel.table.selectRow(0)
    panel.close_selected()

    assert len(app.library) == 0
    assert panel.table.rowCount() == 0
    assert panel.empty_label.isHidden() is False


def test_closing_with_nothing_selected_does_nothing(app):
    add(app)
    panel = DataPanel(app)
    panel.close_selected()
    assert len(app.library) == 1


# --- the dock it shares -----------------------------------------------------


def test_data_and_views_are_one_dock(app, qapp):
    """The Library dock is gone; both panels live in the dock Views had."""
    from pyrpoc.shell.theme.manager import ThemeController
    from pyrpoc.shell.window import PANELS, DockKey, MainWindow

    window = MainWindow(app, ThemeController(qapp))
    try:
        assert [spec.key for spec in PANELS] == [
            DockKey.ACQUISITION,
            DockKey.DEVICES,
            DockKey.DATA,
        ]
        assert set(window.dock_by_key) == {DockKey.ACQUISITION, DockKey.DEVICES, DockKey.DATA}
        assert window.dock_by_key[DockKey.DATA].objectName() == "dock.views"

        shared = window.dock_by_key[DockKey.DATA].widget()
        assert window.data_panel.isAncestorOf(window.data_panel.table)
        for panel in (window.data_panel, window.views_panel):
            assert shared.isAncestorOf(panel), f"{type(panel).__name__} is not in the dock"
    finally:
        window.deleteLater()


def test_both_halves_of_the_shared_dock_are_titled(app, qapp):
    """Two panels in one dock: the tab no longer says which is which."""
    from PyQt6.QtWidgets import QLabel

    from pyrpoc.shell.window import stacked

    data, views = DataPanel(app), DataPanel(app)
    splitter = stacked(("Data", data), ("Views", views))

    boxes = [splitter.widget(index) for index in range(splitter.count())]
    headings = [box.layout().itemAt(0).widget() for box in boxes]
    assert [heading.text() for heading in headings] == ["Data", "Views"]
    assert all(isinstance(heading, QLabel) for heading in headings)
    assert boxes[0].isAncestorOf(data) and boxes[1].isAncestorOf(views)
