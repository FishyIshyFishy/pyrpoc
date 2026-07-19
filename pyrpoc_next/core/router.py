"""Parcel routing: deliver each emitted parcel to the displays that accept its type.

A display is anything satisfying DisplaySink — the gui's display widgets do — so the
core routes without knowing Qt exists.
"""

from __future__ import annotations

from typing import Protocol

from pyrpoc_next.structs.manifest import DisplayManifest
from pyrpoc_next.structs.parcels import Parcel


class DisplaySink(Protocol):
    """A display the router can deliver parcels to."""

    manifest: DisplayManifest

    def render(self, parcel: Parcel) -> None: ...


def route_parcel(parcel: Parcel, displays: list[DisplaySink]) -> None:
    """Send a parcel to every display whose manifest accepts its type."""
    for display in displays:
        if isinstance(parcel, display.manifest.accepted_parcels):
            display.render(parcel)
