"""Repo-root conftest: ensure the repo root is importable during the refactor.

The new package (``pyrpoc_next``) is built alongside the installed ``pyrpoc`` and
isn't exposed by the editable install's finder, so put the repo root on sys.path.
Removed once ``pyrpoc_next`` is swapped into place.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
