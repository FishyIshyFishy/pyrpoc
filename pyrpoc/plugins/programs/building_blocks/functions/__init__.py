"""Behaviour more than one program runs, one topic per file:

galvo_raster.py         the raster waveform and one run's scan
galvo_park.py           parking the galvos at one point
roi_scan.py             a region scan's rows and waveform, from a mask
daq_tasks.py            the NI tasks behind a galvo scan, finite or looping
mask_ttl.py             bound masks as TTL on the digital lines
hold_lines.py           digital lines held high until Stop, such as AOM gates
run_loops.py            stepping through frames or tiles
ccd_read.py             averaged spectra off the Andor CCD, one or a series
synthetic_detector.py   keyed randomness and noise for faked data
"""

from __future__ import annotations
