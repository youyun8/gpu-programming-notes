#!/usr/bin/env python3
"""Scaffold a new problem directory from templates/problem.

Examples:
    python3 scripts/new_problem.py leetgpu "Matrix Transpose" --difficulty easy
    python3 scripts/new_problem.py tensara "ReLU" --difficulty easy --url https://tensara.org/problems/relu
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "templates" / "problem"

PLATFORMS = {
    "leetgpu": {
        "name": "LeetGPU",
        "url": "https://leetgpu.com/challenges/{slug}",
        "entry": 'extern "C" void solve(...)',
    },
    "tensara": {
        "name": "Tensara",
        "url": "https://tensara.org/problems/{slug}",
        "entry": 'extern "C" void solution(...)',
    },
}


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("platform", choices=sorted(PLATFORMS))
    parser.add_argument("title")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--url", help="problem URL (defaults to a guess based on the title)")
    args = parser.parse_args()

    platform = PLATFORMS[args.platform]
    slug = slugify(args.title)
    platform_dir = ROOT / args.platform

    if args.platform == "leetgpu":
        # Number LeetGPU problems in the order they are solved: 001-vector-addition
        existing = [p for p in platform_dir.iterdir() if p.is_dir() and re.match(r"\d{3}-", p.name)]
        dir_name = f"{len(existing) + 1:03d}-{slug}"
    else:
        dir_name = slug

    target = platform_dir / dir_name
    if target.exists() or any(p.name.endswith(slug) for p in platform_dir.iterdir()):
        print(f"error: a directory for '{slug}' already exists in {platform_dir}", file=sys.stderr)
        return 1

    replacements = {
        "{{TITLE}}": args.title,
        "{{PLATFORM}}": platform["name"],
        "{{URL}}": args.url or platform["url"].format(slug=slug),
        "{{DIFFICULTY}}": args.difficulty,
        "{{ENTRY_POINT}}": platform["entry"],
    }

    target.mkdir(parents=True)
    for template in TEMPLATE_DIR.iterdir():
        text = template.read_text()
        for key, value in replacements.items():
            text = text.replace(key, value)
        (target / template.name).write_text(text)

    print(f"created {target.relative_to(ROOT)}")
    print("next: paste the starter code into solution.cu, then run scripts/build_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
