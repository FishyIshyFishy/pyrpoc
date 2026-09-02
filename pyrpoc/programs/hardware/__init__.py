"""The hardware modules programs call. Not a layer -- a shared library.

Every program's ``run()`` is its own, and duplication between them is fine
because nothing imports ``programs/``. What may not be duplicated is the
arithmetic and the NI task setup, because three programs share it and the
arithmetic is pinned by ``tests/reference/phase0_references.npz``. That is the
only reason this package exists, and it is why it sits inside ``programs/``
rather than beside it: sharing code needs a module, not a peer folder.

The contract, which is what makes the reference pinning possible:

- plain functions -- no ``self``, no loop, no lifecycle
- arguments in, arrays out
- **never writes a dataset.** Turning an array into a dataset write is the
  program's job. Enforced by ``test_hardware_modules_never_import_data``.
- no Qt, and no knowledge of which program is calling

Two functions here open NI tasks -- ``raster.run_raster`` and
``tagger.run_flim_scan``. Both take a whole clock domain rather than a single
device: the raster drives AO, reads AI and clocks DO as one synchronised task
because they share a sample clock, and splitting it per device would
misrepresent the hardware. Everything else in this package is numpy.
"""
