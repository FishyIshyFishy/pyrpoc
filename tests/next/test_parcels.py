import numpy as np

from pyrpoc_next.structs.parcels import (
    HistogramCubeParcel,
    ImageFrameParcel,
    ImageParcel,
    Parcel,
    PartialImageParcel,
)


def test_image_frame_channel_count():
    parcel = ImageFrameParcel(data=np.zeros((3, 8, 8), dtype=np.float32), channel_labels=["a", "b", "c"])
    assert parcel.channel_count == 3


def test_image_parcels_share_base_but_differ_by_type():
    frame = ImageFrameParcel(data=np.zeros((1, 4, 4)), channel_labels=["x"])
    partial = PartialImageParcel(data=np.zeros((1, 4, 4)), channel_labels=["x"])
    assert isinstance(frame, ImageParcel) and isinstance(frame, Parcel)
    assert isinstance(partial, ImageParcel)
    # storage keys off the concrete type: only complete frames are persistent
    assert isinstance(frame, ImageFrameParcel)
    assert not isinstance(partial, ImageFrameParcel)


def test_histogram_cube_carries_timing():
    cube = HistogramCubeParcel(
        data=np.zeros((4, 4, 16), dtype=np.float32), bin_width_ps=100.0, laser_period_ps=12500.0
    )
    assert cube.data.shape == (4, 4, 16)
    assert cube.bin_width_ps == 100.0
