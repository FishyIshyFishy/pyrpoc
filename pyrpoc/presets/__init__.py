"""Presets: named compositions of a source (+ decorators), setup, and wiring.

A preset replaces the old hardcoded modality bundle. Importing this package
registers the command handlers and the built-in presets.
"""

from pyrpoc.acquisition import handlers  # noqa: F401  (registers command handlers)
from . import confocal, flim, split_confocal  # noqa: F401  (register presets)
