from __future__ import annotations

from pyrpoc.core.registry import Registry
from pyrpoc.run.program import Program

#: stamp=False: a Program subclass must define nothing beyond uses, params,
#: emits and run (section 12), so the key stays at the registration site.
program_registry: Registry[Program] = Registry("ProgramRegistry", Program, stamp=False)
