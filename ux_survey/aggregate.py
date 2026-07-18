"""Aggregate collected UX-survey result files into per-case tallies.

Drop every participant's downloaded ``.json`` into the results folder, then run
this script. It prints a per-case summary and writes ``summary.csv``.

Configure by editing the plain variables below — no command-line arguments.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

# --- configuration (edit these) -------------------------------------------
results_dir = Path(__file__).parent / "results"   # folder holding collected *.json
output_csv = Path(__file__).parent / "summary.csv"
# ---------------------------------------------------------------------------


def load_results(folder: Path) -> list[dict]:
    files = sorted(p for p in folder.glob("*.json") if p.name != "summary.json")
    out: list[dict] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  ! skipped {path.name}: {exc}")
            continue
        if isinstance(data, dict) and isinstance(data.get("responses"), list):
            data["__file"] = path.name
            out.append(data)
        else:
            print(f"  ! skipped {path.name}: not a survey result")
    return out


def main() -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(results_dir)
    print(f"Loaded {len(results)} result file(s) from {results_dir}\n")
    if not results:
        print("Nothing to aggregate yet — put participants' .json files in the results/ folder.")
        return

    # caseId -> {"operation", "prompt", "counts": Counter(label), "comments": [...]}
    cases: dict[str, dict] = defaultdict(
        lambda: {"operation": "", "prompt": "", "counts": Counter(), "comments": []}
    )
    order: list[str] = []

    for result in results:
        for resp in result["responses"]:
            cid = resp.get("caseId", "?")
            if cid not in cases:
                order.append(cid)
            entry = cases[cid]
            entry["operation"] = resp.get("operation", entry["operation"])
            entry["prompt"] = resp.get("prompt", entry["prompt"])
            label = resp.get("choiceLabel")
            entry["counts"][label if label else "(no answer)"] += 1
            comment = (resp.get("comment") or "").strip()
            if comment:
                entry["comments"].append((result.get("__file", "?"), comment))

    rows: list[list] = []
    for cid in order:
        entry = cases[cid]
        total = sum(entry["counts"].values())
        print(f"[{entry['operation'] or cid}] {entry['prompt']}")
        for label, count in entry["counts"].most_common():
            pct = (100.0 * count / total) if total else 0.0
            bar = "#" * int(round(pct / 5))
            print(f"    {label:<28} {count:>3}  {pct:5.1f}%  {bar}")
            rows.append([cid, entry["operation"], entry["prompt"], label, count, f"{pct:.1f}"])
        if entry["comments"]:
            print(f"    comments ({len(entry['comments'])}):")
            for fname, comment in entry["comments"]:
                print(f"      - ({fname}) {comment}")
        print()

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "operation", "prompt", "option", "count", "percent"])
        writer.writerows(rows)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
