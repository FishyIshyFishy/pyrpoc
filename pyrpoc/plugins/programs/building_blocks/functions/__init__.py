"""Behaviour more than one program runs, one topic per file:

galvo_raster.py         the raster waveform and one run's scan
galvo_park.py           parking the galvos at one point
daq_tasks.py            the NI tasks behind a galvo scan
mask_ttl.py             bound masks as TTL on the digital lines
run_loops.py            stepping through frames or tiles
ccd_read.py             one averaged spectrum off the Andor CCD
synthetic_detector.py   keyed randomness and noise for faked data
"""

from __future__ import annotations
