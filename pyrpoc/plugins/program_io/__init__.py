"""What programs read and publish: the program inputs programs and panels share, and the kinds
of data programs publish. Importing this package declares and registers them all."""

from __future__ import annotations

from . import inputs, outputs

__all__ = ["inputs", "outputs"]
