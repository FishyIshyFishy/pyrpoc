"""Enforce the dependency rules from section 7 and the invariants from section 12.

This catches a kind of regression the behaviour tests cannot: not wrong output,
but the structure quietly reverting to the tangle it replaced. Parses imports
rather than importing anything, so it is fast and has no side effects.

Folders that do not exist yet are skipped, so this can be written in phase 1 and
guard every phase after it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "pyrpoc"

# "may import" — read across. Anything not listed is forbidden.
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "devices": {"core"},
    "operations": {"core", "devices"},
    "data": {"core"},
    "run": {"core", "data", "operations", "devices"},
    "views": {"core", "data"},
    "programs": {"core", "data", "operations", "devices", "run"},
    "session": {"core"},
    "shell": {"core", "devices", "operations", "data", "run", "programs", "views", "session"},
}

# Folders from v3.0 that phase 9 deletes. While they still exist they are
# exempt: they are the old tree, not the new one.
LEGACY = {
    "backend_utils",
    "displays",
    "domain",
    "gui",
    "instruments",
    "modalities",
    "optocontrols",
    "persistence",
    "services",
}

QT_ALLOWED_PREFIXES = ("views/", "shell/")


def _modules() -> list[tuple[str, Path]]:
    """Every module under pyrpoc/, as (top-level folder, path)."""
    out: list[tuple[str, Path]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SOURCE_ROOT)
        if len(rel.parts) == 1:
            continue  # pyrpoc/main.py, pyrpoc/__init__.py
        out.append((rel.parts[0], path))
    return out


def _imported_pyrpoc_folders(path: Path) -> set[str]:
    """Top-level pyrpoc folders this module imports at module scope."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(SOURCE_ROOT)
    own_folder = rel.parts[0]
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "pyrpoc" and len(parts) > 1:
                    found.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative import: level 1 is this folder, deeper walks up.
                depth = node.level - 1
                base = list(rel.parts[:-1])
                target = base[: len(base) - depth] if depth else base
                if target:
                    found.add(target[0])
                continue
            if node.module and node.module.split(".")[0] == "pyrpoc":
                parts = node.module.split(".")
                if len(parts) > 1:
                    found.add(parts[1])

    found.discard(own_folder)
    return found


def _module_scope_imports(path: Path) -> set[str]:
    """Top-level (not function-body) imports, as dotted names."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` blocks do not run; skip them.
            continue
    return names


@pytest.mark.parametrize("folder,path", _modules(), ids=lambda v: str(v))
def test_module_only_imports_allowed_folders(folder, path):
    if folder in LEGACY:
        pytest.skip(f"{folder}/ is v3.0 code that phase 9 deletes")
    assert folder in ALLOWED, f"{folder}/ is not in the dependency table"

    imported = _imported_pyrpoc_folders(path)
    forbidden = {name for name in imported if name not in ALLOWED[folder] and name not in LEGACY}
    assert not forbidden, (
        f"{path.name} in {folder}/ imports {sorted(forbidden)}, "
        f"but {folder}/ may only import {sorted(ALLOWED[folder]) or ['nothing']}"
    )


def test_views_never_import_run_or_programs():
    """Rule 1: the display/acquisition separation, enforced by the import graph."""
    for folder, path in _modules():
        if folder != "views":
            continue
        imported = _imported_pyrpoc_folders(path)
        assert not (imported & {"run", "programs"}), f"{path} imports run/ or programs/"


def test_nothing_imports_programs_except_shell_and_itself():
    """Rule 2: any program can be deleted outright."""
    for folder, path in _modules():
        if folder in {"shell", "programs"} or folder in LEGACY:
            continue
        assert "programs" not in _imported_pyrpoc_folders(path), f"{path} imports programs/"


def test_operations_never_import_data():
    """Rule 3: operations return arrays; writing them into a dataset is the program's job."""
    for folder, path in _modules():
        if folder != "operations":
            continue
        assert "data" not in _imported_pyrpoc_folders(path), f"{path} imports data/"


def test_qt_only_in_views_shell_and_device_panels():
    """Rule 4, at module scope. Device panels are imported lazily inside panel()."""
    for folder, path in _modules():
        if folder in LEGACY:
            continue
        rel = path.relative_to(SOURCE_ROOT).as_posix()
        if rel.startswith(QT_ALLOWED_PREFIXES) or rel.endswith("/panel.py"):
            continue
        qt = {name for name in _module_scope_imports(path) if name.split(".")[0] in {"PyQt6", "PyQt6Ads"}}
        assert not qt, f"{rel} imports {sorted(qt)} at module scope"
