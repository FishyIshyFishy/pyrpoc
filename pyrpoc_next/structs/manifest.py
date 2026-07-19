"""Manifests: hand-declared statements of what each plugin needs and produces.

The compatibility checker reads these before a run. Instruments have no manifest —
a modality names the instruments it requires and uses their methods directly.
"""

from __future__ import annotations

from attrs import define, field

from pyrpoc_next.structs.keys import DisplayKey, InstrumentKey, ModalityKey, ModifierKey
from pyrpoc_next.structs.parameters import ParameterGroups
from pyrpoc_next.structs.parcels import Parcel


@define
class ModalityManifest:
    """What a modality needs and produces."""

    key: ModalityKey
    display_name: str
    emitted_parcels: tuple[type[Parcel], ...]
    required_instruments: tuple[InstrumentKey, ...] = ()
    realizable_modifiers: tuple[ModifierKey, ...] = ()
    parameter_groups: ParameterGroups = field(factory=dict)


@define
class ModifierManifest:
    """What a modifier is and what configuration it carries."""

    key: ModifierKey
    display_name: str
    parameter_groups: ParameterGroups = field(factory=dict)


@define
class DisplayManifest:
    """Which parcel types a display can render."""

    key: DisplayKey
    display_name: str
    accepted_parcels: tuple[type[Parcel], ...]
