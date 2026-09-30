"""The nouns the rest of pyrpoc talks through: one noun per file.

Everything else imports from here and nothing here imports the rest of pyrpoc.
Each plugin kind's file says what a plugin must do and holds the registry it
registers in, so one file answers "what does a new X need?".

    registry.py   Registry: how plugins are found by key
    data.py       kinds of data (Image2D, Spectrum1D, ...): shape contracts
    dataset.py    Dataset, Provenance, Origin: one output's frames and where they came from
    library.py    Library: read access to the open datasets
    saving.py     SaveTarget, and Writer + writer_registry: file formats
    params.py     the parameter language: fields, Group, BlockStore; block_registry
    device.py     Device + device_registry: hardware plugins
    program.py    Program, RunContext + program_registry: experiment plugins
    runner.py     Runner, RunnerContext, controls: the ways a program is started
    picks.py      Pick, PixelPick: a runner asks, a data panel answers
    panel.py      Panel, DataPanel + data_panel_registry: display plugins (the only Qt file)
"""
