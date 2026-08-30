"""Application chrome: the only module that may know everything.

Naming the promiscuous module explicitly is the trick -- it makes it obvious
when it grows too big, which a smeared version never does.

It is separate from views/ despite both being Qt because they have different
import permissions: views/ may not touch run/, shell/ must.
"""
