"""Fail if a number shown in the README or in docs/design.md differs from the committed results.

Every generated table and the verdict sentence sit between a pair of markers:

    <!-- results:gate -->
    ...
    <!-- /results:gate -->

The text between each pair is regenerated from ``results/summary.json`` and compared with
what the document contains. ``--write`` replaces the fragments instead of comparing.

Usage:  python scripts/check_readme.py [--write]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from vqgate.report import readme_blocks

REPO = Path(__file__).resolve().parents[1]
DOCUMENTS = (REPO / "README.md", REPO / "docs" / "design.md")
SUMMARY = REPO / "results" / "summary.json"


def block_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!-- results:{name} -->\n)(.*?)(\n<!-- /results:{name} -->)", flags=re.DOTALL
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="update the fragments")
    args = parser.parse_args()
    expected = readme_blocks(json.loads(SUMMARY.read_text(encoding="utf-8")))
    texts = {path: path.read_text(encoding="utf-8") for path in DOCUMENTS}
    stale = []
    for name, fragment in expected.items():
        pattern = block_pattern(name)
        holders = [path for path, text in texts.items() if pattern.search(text)]
        if len(holders) != 1:
            print(f"'{name}' markers found in {len(holders)} documents, expected exactly one")
            return 1
        path = holders[0]
        if pattern.search(texts[path]).group(2) != fragment:
            stale.append(f"{name} ({path.name})")
            texts[path] = pattern.sub(
                lambda m, fragment=fragment: m.group(1) + fragment + m.group(3), texts[path]
            )
    if args.write:
        for path, text in texts.items():
            path.write_text(text, encoding="utf-8", newline="\n")
        print(f"updated: {', '.join(stale) or 'nothing to change'}")
        return 0
    if stale:
        print(f"out of date: {', '.join(stale)} (run with --write)")
        return 1
    print(f"README.md and docs/design.md match results/summary.json ({len(expected)} fragments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
