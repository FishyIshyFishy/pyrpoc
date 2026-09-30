# pyrpoc

## Architecture (enforced by `uv run lint-imports`)
- `structs/`: the abstract vocabulary, which should import nothing else from pyrpoc.
- `devices/`, `programs/`, `panels/`: implementations. 
  - `programs/` may import `devices/`
  - `panels/` and `programs/` never import each other.
- `app/`: the composition root, which is the only place that knows about everything.
- New implementations register through `structs/registries.py`. Nothing else lists them.

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