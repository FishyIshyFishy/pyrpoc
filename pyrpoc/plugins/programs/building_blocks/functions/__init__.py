"""Behaviour more than one program runs, one topic per file:

galvo_raster.py         the raster waveform and one run's scan
daq_tasks.py            the NI tasks behind a galvo scan
mask_ttl.py             bound masks as TTL on the digital lines
run_loops.py            stepping through frames or tiles
synthetic_detector.py   keyed randomness and noise for faked data
"""

from __future__ import annotations
