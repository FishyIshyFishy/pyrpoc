"""The nouns: every type that more than one folder has to agree on.

Each file is one abstraction and the generic vocabulary that goes with it --
``params`` (Field, the generic field types, Group), ``data`` (Data, the data
kinds, Dataset), ``device`` (Device), ``program`` (Program, RunContext),
``panel`` (Panel), and ``registries`` (where implementations register).
Implementations live in devices/, programs/ and panels/, and subclass what is
here.

Imports nothing from the rest of pyrpoc. No hardware and no file I/O; Qt only
in ``panel.py``, because a panel is a widget.
"""
