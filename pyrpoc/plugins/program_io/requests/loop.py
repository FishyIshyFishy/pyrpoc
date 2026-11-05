"""``RUN_NOW``: run a unit's routines at once, ahead of everything due."""

from __future__ import annotations

from pyrpoc.core.requests import Request
from pyrpoc.utils.value_rules import IntField

RUN_NOW = Request[int]("loop_run_now", IntField("Unit", "The unit to run at once", 1, None, 1))
