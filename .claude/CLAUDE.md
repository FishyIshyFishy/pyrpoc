# pyrpoc

## Architecture (enforced by `uv run lint-imports`; the pieces are defined in `pyrpoc/core/README.md`)
Imports go one way: `app` → `engine` → `plugins` → `utils` → `core`.
- `core/`: the abstract pieces and how they fit together, flat, one idea per file. Nothing specific: no particular input, output kind or value rule. Imports nothing else from pyrpoc; Qt only in `panels.py`.
- `utils/`: utilities used by many parts and belonging to none. `value_rules/` holds the common value rules and the shorthands that declare inputs and device settings with them; `widgets/` holds generic widgets that know no feature; `assets/` holds icons and vendor DLLs. Imports only `core/`.
- `plugins/{devices,program_io,programs,panels}/`: things you add more of.
  - `program_io/inputs/` declares the program inputs programs and panels share; `program_io/outputs/` holds the kinds of data programs publish and the writers that save them.
  - `programs/` and `panels/` may import `program_io/` and `devices/`, and never each other. `program_io/` and `devices/` know only `core/` and `utils/`.
- `engine/`: the app's working parts, one folder per collection: `acquisition/` > `data_library/` | `device_inventory/`. Each holds its no-Qt logic, its Qt model and its panel. They never import a plugin.
- `app/`: the composition root (`application.py` wires the engine; window, menus, theme, `session/`). The only place that knows about everything.
- New implementations register through the registry beside the base they build on: in `core/`, or in `utils/widgets/` for panels. Nothing else lists them.
- A whole new capability is a new folder in `engine/`, wired in `app/application.py`.

## Checks
Before finishing a change, check the files you touched:
`uv run ruff check --fix <files>`, `uv run ruff format <files>`, `uv run pyright <files>`, and `uv run lint-imports` if imports changed. The pre-commit hook re-runs ruff and import-linter on `git commit`, and CI (`.github/workflows/ci.yml`) runs everything on the whole project.

Never silence a check (`noqa`, `ignore_imports`, `type: ignore`, `pyright: ignore`, config changes); stop and ask instead.

## Code style
- Comments explain why, in one or two lines. No history or narration of what the code plainly does.
- Docstrings say what the module, class or function is or does, in one or two plain sentences. Not in a docstring: history, what other modules do, how callers use it, alternatives considered, or restatements of the type annotations. A "why" goes in a comment at the line that needs it.
- Every file starts with `from __future__ import annotations`.
- Fix type errors by typing correctly (narrowing, protocols, stubs). Never `cast` or suppress.
- Keep functions under ~40 lines. Split by extracting a verb-named helper, not to hit a number.
- Validate only at boundaries (user input, files, session JSON, hardware). Inside the codebase, trust the annotations:
  - no `isinstance`/`hasattr`/`getattr(x, "y", default)` on declared types
  - no `try/except` around code that can't fail, and never `except Exception: pass`
  - no `None` checks on values that are never None, and no fallbacks that hide bugs (let it raise)
  - no unused parameters or options "for flexibility"
- no default parameter values unless the default is a real, common choice. Callers pass required values explicitly 