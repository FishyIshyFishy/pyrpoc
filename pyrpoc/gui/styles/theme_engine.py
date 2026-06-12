"""Renders application themes at runtime.

A theme is a JSON file mapping token names (e.g. "highlight") to color values.
The QSS stylesheet and the SVG icons are stored once as templates containing
``^token^`` placeholders; rendering substitutes the theme's colors into both,
so any color scheme works without pregenerating stylesheets per theme.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

token_re = re.compile(r"\^([a-zA-Z0-9:._-]+)\^")
comment_line_re = re.compile(r"^\s*//.*$", re.MULTILINE)

styles_dir = Path(__file__).resolve().parent
templates_dir = styles_dir / "templates"
builtin_themes_dir = styles_dir / "themes"


class ThemeError(Exception):
    """A theme definition is missing, malformed, or incomplete."""


def strip_json_comments(text: str) -> str:
    """Remove lines whose first non-blank characters are ``//``."""
    return comment_line_re.sub("", text)


def load_theme_tokens(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThemeError(f"failed to read theme file {path}: {exc}") from exc
    if not isinstance(raw, dict) or any(
        not isinstance(value, str) for value in raw.values()
    ):
        raise ThemeError(f"theme file {path} must be a JSON object of string values")
    return raw


def resolve_theme_tokens(name: str, theme_files: dict[str, Path]) -> dict[str, str]:
    """Load a theme's tokens, following its optional ``"base"`` chain.

    A theme may set ``"base": "<other-theme>"`` to inherit that theme's tokens
    and override only some of them.
    """
    chain: list[str] = []
    tokens: dict[str, str] = {}
    current = name
    while True:
        if current in chain:
            raise ThemeError(f"theme inheritance cycle: {' -> '.join([*chain, current])}")
        chain.append(current)
        path = theme_files.get(current)
        if path is None:
            origin = f" (base of '{chain[0]}')" if len(chain) > 1 else ""
            raise ThemeError(f"unknown theme '{current}'{origin}")
        layer = load_theme_tokens(path)
        base = layer.pop("base", None)
        tokens = {**layer, **tokens}
        if base is None:
            return tokens
        current = base


def substitute_tokens(template_text: str, tokens: dict[str, str], source: str) -> str:
    missing: set[str] = set()

    def lookup(match: re.Match[str]) -> str:
        value = tokens.get(match.group(1))
        if value is None:
            missing.add(match.group(1))
            return match.group(0)
        return value

    result = token_re.sub(lookup, template_text)
    if missing:
        raise ThemeError(
            f"theme is missing tokens used by {source}: {', '.join(sorted(missing))}"
        )
    return result


def render_icons(tokens: dict[str, str], icon_dir: Path) -> None:
    """Write the theme-colored SVG icon set into icon_dir."""
    icon_dir.mkdir(parents=True, exist_ok=True)
    for template_path in sorted((templates_dir / "icons").glob("*.svg.in")):
        rendered = substitute_tokens(
            template_path.read_text(encoding="utf-8"), tokens, template_path.name
        )
        target = icon_dir / template_path.name.removesuffix(".in")
        target.write_text(rendered, encoding="utf-8")


def render_stylesheet(tokens: dict[str, str], icon_dir: Path) -> str:
    """Return the application QSS for a theme, referencing icons in icon_dir."""
    template = (templates_dir / "stylesheet.qss.in").read_text(encoding="utf-8")
    values = dict(tokens)
    values["icon_dir"] = icon_dir.resolve().as_posix()
    return substitute_tokens(template, values, "stylesheet.qss.in")


def render_theme(tokens: dict[str, str], icon_dir: Path) -> str:
    """Render a theme's icons to disk and return its stylesheet."""
    render_icons(tokens, icon_dir)
    return render_stylesheet(tokens, icon_dir)
