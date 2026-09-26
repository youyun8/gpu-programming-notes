#!/usr/bin/env python3
"""Regenerate the problem tables in README.md from each problem's front matter.

The tables are written between the markers
    <!-- BEGIN {PLATFORM} INDEX --> ... <!-- END {PLATFORM} INDEX -->
Run with --check in CI to fail when README.md is out of date.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
PLATFORMS = ["leetgpu", "tensara"]
STATUS_ICONS = {"solved": "✅", "wip": "🚧", "todo": "⬜"}
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def parse_front_matter(path: Path) -> dict:
    match = re.match(r"^---\n(.*?)\n---\n", path.read_text(), re.S)
    if not match:
        return {}
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def build_table(platform: str) -> str:
    rows = []
    for readme in sorted((ROOT / platform).glob("*/README.md")):
        meta = parse_front_matter(readme)
        if not meta:
            continue
        rel_dir = readme.parent.relative_to(ROOT).as_posix()
        tags = meta.get("tags", "[]").strip("[]")
        rows.append((
            readme.parent.name,
            f"| [{meta.get('title', readme.parent.name)}]({rel_dir}) "
            f"| {meta.get('difficulty', '?')} "
            f"| {tags or '-'} "
            f"| {STATUS_ICONS.get(meta.get('status', 'todo'), '?')} "
            f"| [link]({meta.get('url', '')}) |",
        ))
    if not rows:
        return "_No problems yet._"
    header = "| Problem | Difficulty | Tags | Status | Source |\n|---|---|---|---|---|"
    return header + "\n" + "\n".join(row for _, row in rows)


def main() -> int:
    check_only = "--check" in sys.argv
    original = README.read_text()
    updated = original
    for platform in PLATFORMS:
        tag = platform.upper()
        pattern = re.compile(rf"(<!-- BEGIN {tag} INDEX -->).*?(<!-- END {tag} INDEX -->)", re.S)
        updated = pattern.sub(lambda m: m.group(1) + "\n" + build_table(platform) + "\n" + m.group(2), updated)

    if updated == original:
        print("README.md index is up to date")
        return 0
    if check_only:
        print("README.md index is stale; run: python3 scripts/build_index.py", file=sys.stderr)
        return 1
    README.write_text(updated)
    print("README.md index updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
