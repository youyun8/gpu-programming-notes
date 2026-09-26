#!/usr/bin/env python3
"""Create a folder for every upstream LeetGPU / Tensara problem that does not have one yet.

Each new folder gets a README.md with front matter and a solution.cu stub with
the exact entry-point signature. Existing folders are never overwritten.

    scripts/fetch_upstream.sh          # once
    python3 scripts/sync_problems.py   # add new problems
    python3 scripts/sync_problems.py --list-missing
"""
import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEETGPU = ROOT / ".upstream" / "leetgpu-challenges" / "challenges"
TENSARA = ROOT / ".upstream" / "tensara-problems" / "problems"

# Mirrors src/constants/datatypes.ts in tensara/tensara.
TENSARA_CPP_TYPES = {"float16": "__half", "float8": "uint8_t", "float4": "uint8_t"}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def leetgpu_problems():
    for d in sorted(LEETGPU.glob("*/*/challenge.py")):
        cdir = d.parent
        number, _, name = cdir.name.partition("_")
        title = re.search(r'^\s*name\s*=\s*"([^"]+)"', d.read_text(), re.M).group(1)
        cu = cdir / "starter" / "starter.cu"
        if cu.exists():
            starter = cu.read_text()
            sig = re.search(r'extern\s+"C"\s+void\s+solve\s*\(([^)]*)\)', starter, re.S).group(1)
            entry = f'extern "C" void solve({" ".join(sig.split())})'
            includes = [line for line in starter.splitlines() if line.startswith("#include")]
            language = "cuda"
        else:
            # A few challenges only offer Python frameworks; solve those in PyTorch.
            starter = (cdir / "starter" / "starter.pytorch.py").read_text()
            entry = re.search(r"^def solve\(.*?\):", starter, re.M | re.S).group(0)
            includes = [line for line in starter.splitlines() if line.startswith(("import", "from"))]
            language = "pytorch"
        yield {
            "dir": ROOT / "leetgpu" / f"{int(number):03d}-{name.replace('_', '-').lower()}",
            "title": title,
            "platform": "LeetGPU",
            "upstream": f"{cdir.parent.name}/{cdir.name}",
            "url": f"https://leetgpu.com/challenges/{slugify(title)}",
            "difficulty": cdir.parent.name,
            "tags": [],
            "includes": includes,
            "entry": entry,
            "language": language,
        }


def tensara_problems():
    for d in sorted(TENSARA.glob("*/def.py")):
        pdir = d.parent
        md = (pdir / "problem.md").read_text()
        title = re.search(r'^title:\s*"?([^"\n]+)"?', md, re.M).group(1)
        difficulty = re.search(r'^difficulty:\s*"?(\w+)', md, re.M).group(1).lower()
        tags_m = re.search(r"^tags:\s*\[(.*?)\]", md, re.M)
        tags = [t.strip().strip("\"'") for t in tags_m.group(1).split(",")] if tags_m else []
        params = None
        for node in ast.walk(ast.parse(d.read_text())):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "parameters" for t in node.targets):
                params = ast.literal_eval(node.value)
        args, types = [], set()
        for p in params:
            types.add(p["type"])
            ctype = TENSARA_CPP_TYPES.get(p["type"], p["type"])
            args.append(("const " if p.get("const") else "") + ctype + ("*" if p.get("pointer") else "") + " " + p["name"])
        includes = ["#include <cuda_runtime.h>"]
        if "float16" in types:
            includes.append("#include <cuda_fp16.h>")
        if types & {"uint8_t", "float8", "float4", "uint32_t", "uint64_t"}:
            includes.append("#include <cstdint>")
        yield {
            "dir": ROOT / "tensara" / pdir.name,
            "title": title,
            "platform": "Tensara",
            "upstream": pdir.name,
            "url": f"https://tensara.org/problems/{pdir.name}",
            "difficulty": difficulty,
            "tags": [t for t in tags if t],
            "includes": includes,
            "entry": f'extern "C" void solution({", ".join(args)})',
            "language": "cuda",
        }


def readme(p) -> str:
    return f"""---
title: {p['title']}
platform: {p['platform']}
upstream: {p['upstream']}
url: {p['url']}
difficulty: {p['difficulty']}
tags: [{', '.join(p['tags'])}]
status: todo
---

# {p['title']}

**Platform:** {p['platform']} · **Difficulty:** {p['difficulty']} · [Problem statement]({p['url']})

## Problem

<!-- Summarize the task in your own words: inputs, outputs, shapes,
     test sizes and tolerance. Never copy the upstream statement. -->

## Formulation

<!-- Every display formula is followed by a symbol table.
     No bare | inside math in a table cell: use \\lvert x \\rvert or \\mid. -->

$$
y_i = f(x_i)
$$

| Symbol | Meaning |
|---|---|
| $x_i$ | input element |
| $y_i$ | output element |

## Approach

<!-- Parallel decomposition, memory access pattern, why it is correct. -->

## Cost analysis

$$
Q = \\ldots\\ \\text{{bytes}}, \\qquad W = \\ldots, \\qquad T_{{\\min}} = \\max\\left(\\frac{{W}}{{F}},\\ \\frac{{Q}}{{\\beta}}\\right)
$$

| Symbol | Meaning |
|---|---|
| $Q$ | compulsory DRAM bytes |
| $W$ | useful flops |
| $F, \\beta$ | peak compute throughput and DRAM bandwidth |

## Pitfalls

## Verification

## Related
"""


def stub(p) -> str:
    includes = "\n".join(p["includes"])
    if p["language"] == "pytorch":
        return f"""# {p['title']} ({p['platform']})
# {p['url']}
{includes}


{p['entry']}
    pass
"""
    return f"""// {p['title']} ({p['platform']})
// {p['url']}
{includes}

{p['entry']} {{
}}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-missing", action="store_true")
    args = parser.parse_args()
    if not LEETGPU.exists() or not TENSARA.exists():
        print("error: run scripts/fetch_upstream.sh first", file=sys.stderr)
        return 1
    created = 0
    for p in [*leetgpu_problems(), *tensara_problems()]:
        if p["dir"].exists():
            continue
        if args.list_missing:
            print(p["dir"].relative_to(ROOT))
            continue
        p["dir"].mkdir(parents=True)
        (p["dir"] / "README.md").write_text(readme(p))
        name = "solution.py" if p["language"] == "pytorch" else "solution.cu"
        (p["dir"] / name).write_text(stub(p))
        created += 1
    if not args.list_missing:
        print(f"created {created} problem folder(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
