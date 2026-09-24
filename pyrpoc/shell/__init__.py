"""Application chrome: the only module that may know everything.

Naming the promiscuous module explicitly is the trick -- it makes it obvious
when it grows too big, which a smeared version never does.

It is separate from panels/ despite both being Qt because they have different
import permissions -- with one exception. The four dataset-rendering panels
(image_2d, overlay, mask_editor, spectrum) may not touch run/ or programs/;
shell/ must. The acquisition and devices panels are the exception: they reach
back into shell/ for the program catalog, the run bridge and the parameter
form, the same as this package reaches into panels/ for the cards and the
range slider. See ``panels/__init__.py``.
"""
