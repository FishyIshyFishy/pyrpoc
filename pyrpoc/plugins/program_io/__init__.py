"""What programs read and fill: the program inputs programs and panels share, the requests
panels post to running programs, and the types of program output programs fill. Importing this
package declares and registers them all."""

from __future__ import annotations

from . import inputs, outputs, requests

__all__ = ["inputs", "outputs", "requests"]
