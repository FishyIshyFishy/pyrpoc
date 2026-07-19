from pyrpoc_next.structs.keys import DisplayKey, InstrumentKey, ModalityKey, ModifierKey
from pyrpoc_next.structs.manifest import DisplayManifest, ModalityManifest
from pyrpoc_next.structs.parcels import ImageFrameParcel, PartialImageParcel


def test_modality_manifest_declares_needs_and_output():
    manifest = ModalityManifest(
        key=ModalityKey.confocal,
        display_name="Confocal",
        emitted_parcels=(ImageFrameParcel, PartialImageParcel),
        required_instruments=(InstrumentKey.ni_daq,),
        realizable_modifiers=(ModifierKey.mask,),
    )
    assert ImageFrameParcel in manifest.emitted_parcels
    assert manifest.required_instruments == (InstrumentKey.ni_daq,)


def test_display_manifest_accepts_parcel_types():
    manifest = DisplayManifest(
        key=DisplayKey.streamed,
        display_name="Streamed",
        accepted_parcels=(ImageFrameParcel, PartialImageParcel),
    )
    assert PartialImageParcel in manifest.accepted_parcels
