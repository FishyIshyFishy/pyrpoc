"""The application: constructs every part and connects them.

Not a set of abstractions and not an implementation of one -- the composition
root. It owns the machinery the parts run in: the runner and device claims,
the data library, saving, the session file, the run bridge onto the GUI thread,
the window and its menu. It is the only package that may import everything;
naming it explicitly is what makes it obvious when it grows too big, which a
smeared version never does.

Nothing imports app/ except main.py, with one exception: the acquisition and
devices panels reach in for the program catalog and the ``Application`` they
drive. See ``panels/__init__.py``.
"""
