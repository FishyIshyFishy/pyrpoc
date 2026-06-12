from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyrpoc.gui.styles.theme_engine import (
    ThemeError,
    builtin_themes_dir,
    load_theme_tokens,
    render_icons,
    render_stylesheet,
    render_theme,
    resolve_theme_tokens,
    strip_json_comments,
    substitute_tokens,
    templates_dir,
    token_re,
)


def builtin_theme_names() -> list[str]:
    return sorted(path.stem for path in builtin_themes_dir.glob("*.json"))


def write_theme(path: Path, tokens: dict[str, str]) -> Path:
    path.write_text(json.dumps(tokens), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# JSON loading
# --------------------------------------------------------------------------- #

def test_strip_json_comments_removes_only_comment_lines():
    text = '// header\n{\n    // inline comment line\n    "a": "1"\n}\n'
    assert json.loads(strip_json_comments(text)) == {"a": "1"}


def test_load_theme_tokens_reads_commented_json(tmp_path):
    path = tmp_path / "t.json"
    path.write_text('// comment\n{"highlight": "#ff0000"}', encoding="utf-8")
    assert load_theme_tokens(path) == {"highlight": "#ff0000"}


def test_load_theme_tokens_missing_file_raises(tmp_path):
    with pytest.raises(ThemeError, match="failed to read"):
        load_theme_tokens(tmp_path / "nope.json")


def test_load_theme_tokens_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ThemeError, match="failed to read"):
        load_theme_tokens(path)


def test_load_theme_tokens_rejects_non_string_values(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"a": 3}', encoding="utf-8")
    with pytest.raises(ThemeError, match="string values"):
        load_theme_tokens(path)


# --------------------------------------------------------------------------- #
# token substitution
# --------------------------------------------------------------------------- #

def test_substitute_tokens_replaces_all_occurrences():
    out = substitute_tokens("a ^x^ b ^y^ c ^x^", {"x": "1", "y": "2"}, "test")
    assert out == "a 1 b 2 c 1"


def test_substitute_tokens_missing_token_lists_names():
    with pytest.raises(ThemeError, match=r"missing tokens.*bar.*foo"):
        substitute_tokens("^foo^ ^bar^", {}, "test")


# --------------------------------------------------------------------------- #
# inheritance
# --------------------------------------------------------------------------- #

def test_resolve_theme_tokens_follows_base_chain(tmp_path):
    files = {
        "base-theme": write_theme(tmp_path / "base-theme.json", {"a": "1", "b": "2"}),
        "child": write_theme(
            tmp_path / "child.json", {"base": "base-theme", "b": "20", "c": "30"}
        ),
    }
    assert resolve_theme_tokens("child", files) == {"a": "1", "b": "20", "c": "30"}


def test_resolve_theme_tokens_unknown_name_raises(tmp_path):
    with pytest.raises(ThemeError, match="unknown theme 'ghost'"):
        resolve_theme_tokens("ghost", {})


def test_resolve_theme_tokens_unknown_base_names_origin(tmp_path):
    files = {"child": write_theme(tmp_path / "child.json", {"base": "ghost"})}
    with pytest.raises(ThemeError, match=r"unknown theme 'ghost' \(base of 'child'\)"):
        resolve_theme_tokens("child", files)


def test_resolve_theme_tokens_detects_cycles(tmp_path):
    files = {
        "a": write_theme(tmp_path / "a.json", {"base": "b"}),
        "b": write_theme(tmp_path / "b.json", {"base": "a"}),
    }
    with pytest.raises(ThemeError, match="cycle"):
        resolve_theme_tokens("a", files)


# --------------------------------------------------------------------------- #
# built-in themes render completely
# --------------------------------------------------------------------------- #

def test_twelve_builtin_themes_ship():
    names = builtin_theme_names()
    assert len(names) == 12
    assert "dark-pink" in names
    assert any(name.startswith("light-") for name in names)


@pytest.mark.parametrize("name", builtin_theme_names())
def test_builtin_theme_renders_without_leftover_tokens(name, tmp_path):
    tokens = load_theme_tokens(builtin_themes_dir / f"{name}.json")
    icon_dir = tmp_path / name
    stylesheet = render_theme(tokens, icon_dir)

    assert token_re.search(stylesheet) is None
    assert tokens["highlight"] in stylesheet
    assert icon_dir.as_posix() in stylesheet

    icon_templates = list((templates_dir / "icons").glob("*.svg.in"))
    rendered_icons = list(icon_dir.glob("*.svg"))
    assert len(rendered_icons) == len(icon_templates) > 0
    for icon in rendered_icons:
        assert token_re.search(icon.read_text(encoding="utf-8")) is None


def test_builtin_themes_share_the_same_token_schema():
    schemas = {
        name: set(load_theme_tokens(builtin_themes_dir / f"{name}.json"))
        for name in builtin_theme_names()
    }
    reference = schemas["dark-pink"]
    assert all(schema == reference for schema in schemas.values())


def test_stylesheet_template_uses_only_known_tokens():
    tokens = set(load_theme_tokens(builtin_themes_dir / "dark-pink.json"))
    tokens.add("icon_dir")
    template = (templates_dir / "stylesheet.qss.in").read_text(encoding="utf-8")
    used = set(token_re.findall(template))
    assert used <= tokens


def test_partial_theme_over_builtin_base_renders(tmp_path):
    custom = {"base": "dark-pink", "highlight": "#00ff99"}
    files = {
        "dark-pink": builtin_themes_dir / "dark-pink.json",
        "my-green": write_theme(tmp_path / "my-green.json", custom),
    }
    tokens = resolve_theme_tokens("my-green", files)
    stylesheet = render_stylesheet(tokens, tmp_path / "icons")
    assert "#00ff99" in stylesheet
    render_icons(tokens, tmp_path / "icons")
    assert (tmp_path / "icons" / "checkbox_checked.svg").exists()
