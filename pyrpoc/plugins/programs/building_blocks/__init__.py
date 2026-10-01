"""What programs are assembled from. A program declares its parameter groups
and runners from here, and calls the shared functions; programs never import
each other, so each one can be read, changed or removed on its own.

    parameter_groups/   every parameter block, one per file, with any field type and widget it needs
    runners/            the ways a program can be started
    functions/          behaviour more than one program runs: the galvo raster,
                        its NI tasks and mask TTL, run loops, the fake detector

Only what more than one program uses belongs here. Code one program needs
stays in that program's folder.
"""

from __future__ import annotations
