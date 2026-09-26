#!/usr/bin/env python3
"""Generate the tutorial figures (SVG) from their Python descriptions.

    python3 scripts/build_figures.py            # writes tutorials/figures/*.svg
    python3 scripts/build_figures.py --check    # fails if a committed SVG is stale

Each module in scripts/figures/ (ch00.py, ch01.py, ..., gemm.py) defines
functions named ``fig_<name>`` that return an ``Svg``; the figure is written
to tutorials/figures/<module>-<name>.svg. The figures are committed so that
GitHub renders them too; the site builder inlines them (scripts/build_site.py).
"""
import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tutorials" / "figures"
MODULES = ["ch00", "ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "gemm"]


def collect():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    figures = {}
    for mod_name in MODULES:
        try:
            mod = importlib.import_module(f"figures.{mod_name}")
        except ModuleNotFoundError as e:
            if e.name == f"figures.{mod_name}":
                continue
            raise
        for attr in sorted(dir(mod)):
            if attr.startswith("fig_"):
                name = f"{mod_name}-{attr[4:].replace('_', '-')}"
                svg = getattr(mod, attr)(name)
                figures[name + ".svg"] = svg.render()
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check that the committed SVGs are current")
    args = parser.parse_args()
    figures = collect()
    stale = []
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, content in figures.items():
        dest = OUT / fname
        if dest.exists() and dest.read_text() == content:
            continue
        stale.append(fname)
        if not args.check:
            dest.write_text(content)
    orphans = sorted(p.name for p in OUT.glob("*.svg") if p.name not in figures)
    if args.check:
        for f in stale:
            print(f"stale figure: tutorials/figures/{f} (run scripts/build_figures.py)", file=sys.stderr)
        for f in orphans:
            print(f"orphan figure: tutorials/figures/{f}", file=sys.stderr)
        return 1 if stale or orphans else 0
    for f in orphans:
        (OUT / f).unlink()
    print(f"{len(figures)} figures, {len(stale)} written, {len(orphans)} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
