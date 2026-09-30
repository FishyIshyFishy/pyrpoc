"""Acquisition: choosing a program, setting it up, and running it.

model.py       Acquisition: the selected program, its parameters, where it saves
host.py        RunnerHost: attaches the program's runners and routes their requests
events.py      RunEvents: starting runs and reporting them on the GUI thread
executor.py    Executor: runs programs on worker threads (no Qt)
recording.py   Recording, Series: what runs write into, and runs that continue one (no Qt)
claims.py      which devices a run needs and holds (no Qt)
panel.py       the Acquisition panel
"""

from __future__ import annotations
