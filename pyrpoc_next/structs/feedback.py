"""Display-to-modality feedback types.

Feedback (e.g. draw a box, reacquire that region) is deferred in the first cut, but
the types live here so the runner and modalities can carry a feedback channel from
the start without a later signature change. Concrete event kinds are added as needed.
"""

from __future__ import annotations

from attrs import define


@define
class Region:
    """A rectangular region in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int


@define
class FeedbackEvent:
    """Base for a display-to-modality event. Kept open — no fixed selection type yet."""
