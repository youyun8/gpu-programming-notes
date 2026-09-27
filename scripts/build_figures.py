#!/usr/bin/env python3
"""Generate the tutorial figures (SVG) from their Python descriptions.

    python3 scripts/build_figures.py            # writes tutorials/figures/*.svg
    python3 scripts/build_figures.py --check    # fails if a committed SVG is stale

Each module in scripts/figures/ (ch00.py, ch01.py, ..., gemm.py) defines
functions named ``fig_<name>`` that return an ``Svg``; the figure is written
to tutorials/figures/<module>-<name>.svg. The problem modules (leetgpu.py,
tensara.py) are the exception: ``fig_<dir>`` is written next to the problem,
to <platform>/<dir>/figure.svg (underscores in <dir> become hyphens).
The figures are committed so that GitHub renders them too; the site builder
inlines them (scripts/build_site.py).
"""
import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tutorials" / "figures"
MODULES = ["overview", "ch00", "ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "ch09", "ch10", "ch11", "ch12",
           "ch13", "ch14", "gemm"]
PROBLEM_MODULES = ["leetgpu", "tensara"]  # one figure per problem folder
PROBLEM_FIGURE = "figure.svg"


def figure_path(module: str, name: str) -> Path:
    """Where the figure ``fig_<name>`` of ``module`` is written."""
    if module in PROBLEM_MODULES:
        return ROOT / module / name / PROBLEM_FIGURE
    return OUT / f"{module}-{name}.svg"


def committed_figures() -> list[Path]:
    """Every figure currently in the repository."""
    return sorted(OUT.glob("*.svg")) + sorted(p for m in PROBLEM_MODULES for p in (ROOT / m).glob(f"*/{PROBLEM_FIGURE}"))


def collect():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    figures = {}
    for mod_name in MODULES + PROBLEM_MODULES:
        try:
            mod = importlib.import_module(f"figures.{mod_name}")
        except ModuleNotFoundError as e:
            if e.name == f"figures.{mod_name}":
                continue
            raise
        for attr in sorted(dir(mod)):
            if attr.startswith("fig_"):
                short = attr[4:].replace('_', '-')
                if mod_name in PROBLEM_MODULES and not (ROOT / mod_name / short).is_dir():
                    raise SystemExit(f"figures/{mod_name}.py: {attr} has no problem folder {mod_name}/{short}")
                svg = getattr(mod, attr)(f"{mod_name}-{short}")
                figures[figure_path(mod_name, short)] = svg.render()
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check that the committed SVGs are current")
    args = parser.parse_args()
    figures = collect()
    stale = []
    OUT.mkdir(parents=True, exist_ok=True)
    for dest, content in figures.items():
        if dest.exists() and dest.read_text() == content:
            continue
        stale.append(dest)
        if not args.check:
            dest.write_text(content)
    orphans = [p for p in committed_figures() if p not in figures]
    if args.check:
        for f in stale:
            print(f"stale figure: {f.relative_to(ROOT)} (run scripts/build_figures.py)", file=sys.stderr)
        for f in orphans:
            print(f"orphan figure: {f.relative_to(ROOT)}", file=sys.stderr)
        return 1 if stale or orphans else 0
    for f in orphans:
        f.unlink()
    print(f"{len(figures)} figures, {len(stale)} written, {len(orphans)} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
