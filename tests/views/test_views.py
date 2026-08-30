"""Views render datasets and never own arrays.

The properties this phase exists to deliver: data outlives the widget that
showed it, two views can share one run, and a closed view can be reopened
without losing anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrpoc.core.streams import Cube3D, Image2D, Samples4D
from pyrpoc.data.dataset import Dataset, Provenance
from pyrpoc.data.library import DatasetLibrary
from pyrpoc.views.image_2d import Image2DView
from pyrpoc.views.mask_editor import MaskEditorView
from pyrpoc.views.overlay import OverlayView
from pyrpoc.views.registry import view_registry

VIEW_CLASSES = [Image2DView, OverlayView, MaskEditorView]


def make_dataset(stream="intensity", spec=Image2D, run_id=1, labels=None) -> Dataset:
    return Dataset(
        stream=stream,
        spec=spec,
        provenance=Provenance("confocal", run_id=run_id),
        channel_labels=labels,
    )


def frame(value: float = 1.0, channels: int = 2) -> np.ndarray:
    return np.full((channels, 6, 8), value, np.float32)


# --- registry and contracts -------------------------------------------------


def test_all_three_views_are_registered(qapp):
    assert view_registry.keys() == ["image_2d", "mask_editor", "overlay"]


@pytest.mark.parametrize("cls", VIEW_CLASSES)
def test_every_view_declares_what_it_renders(cls, qapp):
    assert cls.renders == [Image2D]


@pytest.mark.parametrize("cls", VIEW_CLASSES)
def test_a_view_refuses_a_contract_it_does_not_render(cls, qapp):
    view = cls()
    with pytest.raises(TypeError, match="renders"):
        view.bind(make_dataset("histogram", Cube3D))


def test_no_view_renders_the_split_raw_stream(qapp):
    """Recorded: Samples4D and Cube3D are saved but have no view in v3.1."""
    for cls in VIEW_CLASSES:
        assert Samples4D not in cls.renders
        assert Cube3D not in cls.renders


# --- data outlives the widget ------------------------------------------------


def test_a_view_holds_no_array_of_its_own(qapp):
    dataset = make_dataset()
    dataset.append(frame(3.0))
    view = Image2DView()
    view.bind(dataset)
    view.refresh()
    assert view.dataset() is dataset
    assert not hasattr(view, "_data_chw"), "the v3.0 array-owning attribute is gone"


def test_closing_a_view_does_not_destroy_the_data(qapp):
    library = DatasetLibrary()
    dataset = library.add(make_dataset())
    dataset.append(frame(5.0))

    view = Image2DView()
    view.attach_library(library)
    view.deleteLater()
    del view

    np.testing.assert_array_equal(library.all()[0].latest(), frame(5.0))


def test_a_reopened_view_binds_to_the_data_that_is_still_there(qapp):
    library = DatasetLibrary()
    dataset = library.add(make_dataset())
    dataset.append(frame(2.0))

    reopened = Image2DView()
    reopened.attach_library(library)
    assert reopened.dataset() is dataset
    np.testing.assert_array_equal(reopened.dataset().latest(), frame(2.0))


def test_two_views_share_one_dataset(qapp):
    library = DatasetLibrary()
    dataset = library.add(make_dataset())
    dataset.append(frame(1.0))

    first, second = Image2DView(), OverlayView()
    first.attach_library(library)
    second.attach_library(library)

    assert first.dataset() is second.dataset() is dataset
    dataset.append(frame(9.0))
    first.refresh()
    second.refresh()
    np.testing.assert_array_equal(first.dataset().latest(), second.dataset().latest())


# --- the source picker -------------------------------------------------------


def test_latest_follows_the_newest_matching_run(qapp):
    library = DatasetLibrary()
    view = Image2DView()
    view.attach_library(library)
    assert view.dataset() is None

    first = library.add(make_dataset(run_id=1))
    assert view.dataset() is first

    second = library.add(make_dataset(run_id=2))
    assert view.dataset() is second, "Latest must follow the newest run"


def test_picking_a_run_explicitly_stops_it_following(qapp):
    library = DatasetLibrary()
    first = library.add(make_dataset(run_id=1))
    view = Image2DView()
    view.attach_library(library)

    view.source_combo.setCurrentIndex(view.source_combo.findData(first.id))
    library.add(make_dataset(run_id=2))
    assert view.dataset() is first, "an explicit choice must not be overridden"


def test_the_picker_only_offers_datasets_this_view_renders(qapp):
    library = DatasetLibrary()
    library.add(make_dataset("intensity", Image2D))
    library.add(make_dataset("histogram", Cube3D))
    library.add(make_dataset("raw_pixel_stream", Samples4D))

    view = Image2DView()
    view.attach_library(library)
    # One "Latest" entry plus the single Image2D dataset.
    assert view.source_combo.count() == 2


def test_an_empty_library_leaves_the_view_unbound(qapp):
    view = Image2DView()
    view.attach_library(DatasetLibrary())
    view.refresh()
    assert view.dataset() is None


# --- rendering ---------------------------------------------------------------


def test_image_view_builds_a_tile_per_channel(qapp):
    dataset = make_dataset(labels=["ai0", "ai1", "ai2"])
    dataset.append(frame(1.0, channels=3))
    view = Image2DView()
    view.bind(dataset)
    assert len(view._tiles) == 3


def test_image_view_names_tiles_after_the_acquisition_channels(qapp):
    dataset = make_dataset(labels=["ai0", "ai1"])
    dataset.append(frame(1.0))
    view = Image2DView()
    view.bind(dataset)
    assert view.get_channel_names() == ["ai0", "ai1"]


def test_a_hand_typed_tile_name_survives_the_next_frame(qapp):
    dataset = make_dataset(labels=["ai0", "ai1"])
    dataset.append(frame(1.0))
    view = Image2DView()
    view.bind(dataset)
    view._tiles[0].name_edit.setText("Transmission")

    dataset.append(frame(2.0))
    view.refresh()
    assert view.get_channel_names()[0] == "Transmission"


def test_tile_count_follows_a_changed_channel_count(qapp):
    dataset = make_dataset()
    dataset.append(frame(1.0, channels=2))
    view = Image2DView()
    view.bind(dataset)
    assert len(view._tiles) == 2

    other = make_dataset(run_id=2)
    other.append(frame(1.0, channels=4))
    view.bind(other)
    assert len(view._tiles) == 4


def test_binding_none_clears_the_view(qapp):
    dataset = make_dataset()
    dataset.append(frame(1.0))
    view = Image2DView()
    view.bind(dataset)
    view.bind(None)
    assert view._tiles == []


def test_overlay_composites_every_channel(qapp):
    dataset = make_dataset()
    dataset.append(frame(1.0, channels=2))
    view = OverlayView()
    view.bind(dataset)
    assert len(view._controls) == 2
    assert view.frame() is not None


def test_overlay_clear_leaves_no_controls(qapp):
    dataset = make_dataset()
    dataset.append(frame(1.0))
    view = OverlayView()
    view.bind(dataset)
    view.clear()
    assert view._controls == []


# --- persistence --------------------------------------------------------------


def test_image_view_state_round_trips(qapp):
    dataset = make_dataset(labels=["ai0", "ai1"])
    dataset.append(frame(1.0))
    view = Image2DView()
    view.bind(dataset)
    view._tiles[0].name_edit.setText("Transmission")
    view._tiles[0].autoscale_box.setChecked(False)
    state = view.export_persistence_state()

    restored = Image2DView()
    restored.bind(dataset)
    restored.import_persistence_state(state)
    assert restored.get_channel_names()[0] == "Transmission"
    assert restored._tiles[0].autoscale_box.isChecked() is False


def test_overlay_state_round_trips(qapp):
    dataset = make_dataset()
    dataset.append(frame(1.0))
    view = OverlayView()
    view.bind(dataset)
    view._controls[0].autoscale_box.setChecked(False)
    state = view.export_persistence_state()

    restored = OverlayView()
    restored.bind(dataset)
    restored.import_persistence_state(state)
    assert restored._controls[0].autoscale_box.isChecked() is False


def test_importing_junk_state_is_survivable(qapp):
    view = Image2DView()
    view.import_persistence_state({})
    view.import_persistence_state({"channels": "nope"})
    view.import_persistence_state({"channels": [1, 2, None]})


# --- the mask editor ----------------------------------------------------------


def test_the_mask_editor_reads_a_dataset_not_a_widget(qapp):
    """v3.0 reached into a display's get_normalized_data_3d."""
    dataset = make_dataset()
    dataset.append(np.linspace(0, 1, 2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8))
    editor = MaskEditorView()
    editor.bind(dataset)
    assert editor._data.shape == (2, 6, 8)
    assert editor._data.max() <= 255.0


def test_the_mask_editor_shows_nothing_when_unbound(qapp):
    """v3.0 synthesised a plausible microscope image here."""
    editor = MaskEditorView()
    editor.clear()
    assert float(editor._data.max()) == 0.0


def test_roi_editing_survives_a_same_shape_frame(qapp):
    """Rebuilding the table per frame would make it unusable during a run."""
    dataset = make_dataset()
    dataset.append(frame(1.0))
    editor = MaskEditorView()
    editor.bind(dataset)
    editor.add_roi([(1.0, 1.0), (5.0, 1.0), (5.0, 4.0)])
    assert editor.has_rois()

    dataset.append(frame(2.0))
    editor.refresh()
    assert editor.has_rois(), "the ROI was discarded on a new frame"


def test_a_different_shape_resets_the_editor(qapp):
    dataset = make_dataset()
    dataset.append(frame(1.0, channels=2))
    editor = MaskEditorView()
    editor.bind(dataset)
    editor.add_roi([(1.0, 1.0), (5.0, 1.0), (5.0, 4.0)])

    other = make_dataset(run_id=2)
    other.append(frame(1.0, channels=3))
    editor.bind(other)
    assert not editor.has_rois()


def test_generate_mask_needs_an_roi(qapp):
    editor = MaskEditorView()
    assert editor.generate_mask() is None


def test_saving_writes_a_loadable_mask(qapp, tmp_path, monkeypatch):
    from pyrpoc.core.modulation import load_mask

    dataset = make_dataset()
    dataset.append(frame(200.0))
    editor = MaskEditorView()
    editor.bind(dataset)
    editor.add_roi([(1.0, 1.0), (6.0, 1.0), (6.0, 4.0), (1.0, 4.0)])
    editor.low_spin.setValue(0)
    editor.high_spin.setValue(255)

    target = tmp_path / "mask.png"
    monkeypatch.setattr(
        "pyrpoc.views.mask_editor.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(target), ""),
    )
    saved = []
    editor.mask_saved.connect(lambda path, mask: saved.append(path))
    editor.save_mask()

    assert target.exists()
    assert saved == [str(target)]
    assert load_mask(target).max() == 255
    assert editor.is_dirty() is False


def test_the_editor_has_no_create_or_cancel_signals(qapp):
    """A mask is a preset written to a file, not something pushed into a control."""
    assert not hasattr(MaskEditorView, "create_mask_requested")
    assert not hasattr(MaskEditorView, "cancel_requested")
