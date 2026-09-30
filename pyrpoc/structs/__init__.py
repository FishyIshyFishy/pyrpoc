"""The nouns the rest of pyrpoc talks through. Nothing here imports the rest of pyrpoc.

The layout mirrors the package: a noun lives at the same path as the code that
implements it, so working in ``pyrpoc/X/`` means reading ``structs/X/``. Each
plugin kind's file says what a plugin must do and holds the registry it
registers in.

    registry.py        Registry: how plugins are found by key
    plugins/           what plugins implement
      devices.py         Device + device_registry
      programs/          Program, RunContext + program_registry; Runner and its
                         controls; Pick (a runner asks, a data panel answers)
      data_panels.py     Panel, DataPanel + data_panel_registry (the only Qt file)
      params.py          the parameter language plugins declare: fields, Group,
                         BlockStore; block_registry
    data_library/      what the data library holds
      data.py            kinds of data (Image2D, Spectrum1D, ...)
      dataset.py         Dataset, Provenance, Origin
      library.py         Library: read access to the open datasets
      saving.py          SaveTarget, Writer + writer_registry: file formats
"""

from __future__ import annotations
