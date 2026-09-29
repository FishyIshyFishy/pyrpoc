"""The pieces programs are assembled from.

A program is a composition: it declares which parameter blocks it wants, which
runners it can be started by, and writes the loop that drives them. What it
composes lives here:

- ``param_groups/``: the parameter blocks, one per file, plus this
  instrument's own field types (masks, points).
- ``runners/``: the ways a program can be started, one per file.
- ``editors.py``: the widgets for those field types, handed to the form
  through ``Field.editor``. The one Qt module here, imported only when an
  editor is asked for, so programs import with no Qt in sight.

May import ``structs/`` and ``devices/``. Nothing here knows which program is
using it, and nothing here writes a dataset.
"""
