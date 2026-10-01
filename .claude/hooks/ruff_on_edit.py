from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Silent on success so the ~99% clean edits cost no context; only unfixable
# lint errors are reported back (exit 2 feeds stderr to Claude).


def run_ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args, "--force-exclude"],
        capture_output=True,
        text=True,
    )


def main() -> int:
    path = Path(json.load(sys.stdin)["tool_input"]["file_path"])
    if path.suffix not in {".py", ".pyi"}:
        return 0
    run_ruff("format", str(path))
    check = run_ruff("check", "--fix", "--quiet", str(path))
    if check.returncode == 0:
        return 0
    sys.stderr.write(check.stdout + check.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
