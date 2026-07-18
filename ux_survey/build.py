"""Build a single, self-contained ``survey.html`` for easy distribution.

Inlines ``survey.css``, ``cases.js``, and ``survey.js`` into ``index.html`` so the
whole survey becomes ONE file a colleague can double-click (or you can email /
drop on a shared drive) — no folder, no zip-extract step, no missing stylesheet.

Keep editing the source files as usual, then run ``python build.py`` to
regenerate ``survey.html``.
"""
from __future__ import annotations

from pathlib import Path
import re

here = Path(__file__).parent
output_html = here / "survey.html"


def read(name: str) -> str:
    return (here / name).read_text(encoding="utf-8")


def inline_script(text: str) -> str:
    # A literal </script> inside inlined JS would close the tag early.
    return text.replace("</script>", "<\\/script>")


def warn_external_images(cases_js: str) -> None:
    # Only scan the actual case data, not the schema doc-comment at the top.
    marker = "window.SURVEY_CASES"
    body = cases_js[cases_js.index(marker):] if marker in cases_js else cases_js
    hits = re.findall(r"""image\s*:\s*["']((?!data:)[^"']+)["']""", body)
    for src in hits:
        print(
            f"  ! warning: cases.js references image '{src}'. Relative images do NOT "
            "travel in the single file - use a data: URI or previewText instead."
        )


def replace_once(html: str, anchor: str, replacement: str) -> str:
    if anchor not in html:
        raise SystemExit(f"build failed: could not find {anchor!r} in index.html")
    return html.replace(anchor, replacement, 1)


def build() -> None:
    html = read("index.html")
    warn_external_images(read("cases.js"))

    html = replace_once(
        html,
        '<link rel="stylesheet" href="survey.css" />',
        "<style>\n" + read("survey.css") + "\n</style>",
    )
    html = replace_once(
        html,
        '<script src="cases.js"></script>',
        "<script>\n" + inline_script(read("cases.js")) + "\n</script>",
    )
    html = replace_once(
        html,
        '<script src="survey.js"></script>',
        "<script>\n" + inline_script(read("survey.js")) + "\n</script>",
    )

    output_html.write_text(html, encoding="utf-8")
    print(f"Wrote {output_html} ({len(html):,} bytes) - send this ONE file to participants.")


if __name__ == "__main__":
    build()
