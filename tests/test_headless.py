"""Everything outside views/ and shell/ must import without Qt.

Section 12: "The test suite runs headless with no Qt application and no
hardware." This is what makes rule 4 mean something in practice -- device panels
have to be imported lazily inside ``Device.panel()`` rather than at module
scope, or this fails.

Runs in a subprocess so one stray import in another test cannot mask it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

PROBE = textwrap.dedent(
    """
    import importlib, sys

    folders = ["core", "devices", "data", "run", "programs", "session"]
    imported = []
    for name in folders:
        try:
            importlib.import_module(f"pyrpoc.{name}")
            imported.append(name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("pyrpoc"):
                continue          # folder does not exist yet
            raise

    leaked = sorted(m for m in sys.modules if m.startswith(("PyQt6", "pyqtgraph")))
    print("IMPORTED:" + ",".join(imported))
    print("LEAKED:" + ",".join(leaked))
    """
)


def test_headless_layers_do_not_import_qt():
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    lines = dict(line.split(":", 1) for line in result.stdout.strip().splitlines())
    imported = [name for name in lines["IMPORTED"].split(",") if name]
    leaked = [name for name in lines["LEAKED"].split(",") if name]

    assert "core" in imported and "programs" in imported, imported
    assert not leaked, (
        "importing the headless layers pulled in Qt: "
        + ", ".join(leaked)
        + " -- a device panel or view is being imported at module scope"
    )
