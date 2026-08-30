"""Bounded hardware actions. Plain functions: arguments in, arrays out.

No self, no loop, no saving, no Qt, and no knowledge of what they are used for.
The unit is a clock domain, not a device: the raster drives AO, AI and DO as one
synchronised NI task because they share a sample clock.

May import core/ and devices/. Must not import data/.
"""
