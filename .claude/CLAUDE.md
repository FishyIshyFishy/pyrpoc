# pyrpoc

## Architecture (enforced by `uv run lint-imports`; the map is `docs/architecture.md`)
- `structs/`: the vocabulary, one noun per file. Imports nothing else from pyrpoc; Qt only in `panel.py`.
- `plugins/{devices,programs,data_panels}/`: things you add more of. Each knows only `structs/` and `qt_components/`.
  - `programs/` may import `devices/`; `programs/` and `data_panels/` never import each other.
- `qt_components/`: generic widgets that know no feature.
- Subsystems `acquisition/` > `data_library/` | `device_inventory/`: each holds its no-Qt logic, its Qt model and its panel. They never import a plugin.
- `app/`: the composition root (`application.py` wires the subsystems; window, menus, theme, `session/`). The only place that knows about everything.
- New implementations register through the registry in the `structs/` file that defines their kind. Nothing else lists them.
- A whole new capability is a new subsystem folder, wired in `app/application.py`.

## Checks
Never silence a check (`noqa`, `ignore_imports`, `type: ignore`, `pyright: ignore`, config changes); stop and ask instead. Pre-commit (incl. slow pyright) runs only on `git commit`; never run it after ordinary edits. Fix every failure it reports. 

## Code style
- Comments explain why, in one or two lines. No history or narration of what the code plainly does.
- Plain `#` comments, never `#:`.
- Every file starts with `from __future__ import annotations`.
- Fix type errors by typing correctly (narrowing, protocols, stubs). Never `cast` or suppress.
- Keep functions under ~40 lines. Split by extracting a verb-named helper, not to hit a number.
- Validate only at boundaries (user input, files, session JSON, hardware). Inside the codebase, trust the annotations:
  - no `isinstance`/`hasattr`/`getattr(x, "y", default)` on declared types
  - no `try/except` around code that can't fail, and never `except Exception: pass`
  - no `None` checks on values that are never None, and no fallbacks that hide bugs (let it raise)
  - no unused parameters or options "for flexibility"
- no default parameter values unless the default is a real, common choice. Callers pass required values explicitly 