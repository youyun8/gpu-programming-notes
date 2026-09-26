#!/usr/bin/env python3
"""Translate a CUDA source file for the cuemu CPU emulator and build it as a shared library.

Usage:
    python3 tools/cuemu/cuemu.py build path/to/solution.cu -o /tmp/solution.so
    python3 tools/cuemu/cuemu.py translate path/to/solution.cu     # print translated source
    python3 tools/cuemu/cuemu.py run path/to/program.cu -- ARGS    # build a program with main() and run it
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADER = HERE / "cuemu.h"

DROPPED_INCLUDES = {
    "cuda_runtime.h", "cuda.h", "cuda_runtime_api.h", "cuda_fp16.h", "cuda_bf16.h", "cuda_fp8.h",
    "cuda_fp4.h", "device_launch_parameters.h", "math_constants.h", "device_functions.h",
    "vector_types.h", "sm_20_atomic_functions.h", "cuda_pipeline.h", "mma.h", "cstdint", "stdint.h",
    "cooperative_groups.h", "cooperative_groups/reduce.h", "cooperative_groups/scan.h",
}
UNSUPPORTED_INCLUDES = {"cub/cub.cuh", "cublas_v2.h", "cudnn.h"}


class TranslateError(Exception):
    pass


def _strip_includes(src: str) -> str:
    def repl(m):
        name = m.group(1)
        if name in DROPPED_INCLUDES:
            return f"// [cuemu] dropped #include <{name}>"
        if name in UNSUPPORTED_INCLUDES or name.startswith("thrust/") or name.startswith("cub/"):
            raise TranslateError(f"#include <{name}> is not supported by cuemu")
        return m.group(0)

    return re.sub(r'#\s*include\s*[<"]([^>"]+)[>"]', repl, src)


def _rewrite_extern_shared(src: str) -> str:
    pattern = re.compile(r"extern\s+__shared__\s+([^;\[\]]+?)\s*\b(\w+)\s*\[\s*\]\s*;")

    def repl(m):
        decl = re.sub(r"__align__\s*\(\s*\d+\s*\)|alignas\s*\(\s*\d+\s*\)", "", m.group(1)).strip()
        name = m.group(2)
        return f"{decl}* {name} = cuemu::dynamicSmem<{decl}>();"

    return pattern.sub(repl, src)


def _match_back_angle(src: str, end: int) -> int:
    """src[end] == '>'; return index of the matching '<'."""
    depth = 0
    i = end
    while i >= 0:
        if src[i] == ">":
            depth += 1
        elif src[i] == "<":
            depth -= 1
            if depth == 0:
                return i
        i -= 1
    raise TranslateError("unbalanced template arguments before <<<")


def _rewrite_launches(src: str) -> str:
    out = []
    pos = 0
    while True:
        start = src.find("<<<", pos)
        if start < 0:
            out.append(src[pos:])
            return "".join(out)
        # Kernel name: identifier (with ::) optionally followed by <template args>.
        j = start - 1
        while j >= 0 and src[j].isspace():
            j -= 1
        if src[j] == ">":
            j = _match_back_angle(src, j) - 1
        while j >= 0 and (src[j].isalnum() or src[j] in "_:"):
            j -= 1
        name_start = j + 1
        name = src[name_start:start].strip()
        if not name:
            raise TranslateError(f"cannot find kernel name before <<< at offset {start}")
        end = src.find(">>>", start)
        if end < 0:
            raise TranslateError("unterminated <<<")
        cfg = src[start + 3:end]
        k = end + 3
        while k < len(src) and src[k].isspace():
            k += 1
        if k >= len(src) or src[k] != "(":
            raise TranslateError("expected '(' after >>>")
        m = k + 1
        while m < len(src) and src[m].isspace():
            m += 1
        sep = "" if src[m] == ")" else ", "
        out.append(src[pos:name_start])
        out.append(f"cuemu::Launcher({cfg})({name}{sep}")
        pos = k + 1


def _inline_local_includes(src: str, base: Path, seen=None) -> str:
    """Paste `#include "header"` files found next to the source, so they are translated too."""
    seen = set() if seen is None else seen

    def repl(m):
        path = (base / m.group(1)).resolve()
        if not path.is_file():
            return m.group(0)
        if path in seen:
            return f"// [cuemu] already included {m.group(1)}"
        seen.add(path)
        text = _inline_local_includes(path.read_text(), path.parent, seen)
        text = re.sub(r"^\s*#\s*pragma\s+once\s*$", "", text, flags=re.M)
        return f'#line 1 "{path}"\n{text}\n// [cuemu] end of {m.group(1)}'

    return re.sub(r'^[ \t]*#\s*include\s*"([^"]+)"', repl, src, flags=re.M)


def translate(src: str, base: Path = None) -> str:
    if base is not None:
        src = _inline_local_includes(src, base)
    src = _strip_includes(src)
    src = _rewrite_extern_shared(src)
    src = _rewrite_launches(src)
    return src


def compiler() -> str:
    return os.environ.get("CUEMU_CXX", "clang++")


def build(cu_path: Path, out_path: Path, extra_flags=(), executable=False) -> Path:
    """Build a shared library (the default) or, with executable=True, a program with its own main()."""
    translated = translate(Path(cu_path).read_text(), Path(cu_path).resolve().parent)
    with tempfile.TemporaryDirectory() as tmp:
        cpp = Path(tmp) / (Path(cu_path).stem + ".cpp")
        cpp.write_text(f'#line 1 "{Path(cu_path).resolve()}"\n' + translated)
        kind = [] if executable else ["-fPIC", "-shared"]
        cmd = [
            compiler(), "-std=c++20", "-O2", *kind, "-fno-strict-aliasing", "-w",
            "-include", str(HEADER), str(cpp), "-o", str(out_path), *extra_flags,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise TranslateError("compilation failed:\n" + result.stderr[-6000:])
    return out_path


def cached_build(cu_path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(Path(cu_path).read_bytes() + HEADER.read_bytes() + Path(__file__).read_bytes()).hexdigest()
    out = cache_dir / f"{Path(cu_path).parent.name}-{digest[:12]}.so"
    if not out.exists():
        build(cu_path, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("source")
    b.add_argument("-o", "--output", required=True)
    t = sub.add_parser("translate")
    t.add_argument("source")
    r = sub.add_parser("run")
    r.add_argument("source")
    r.add_argument("args", nargs=argparse.REMAINDER, help="program arguments (after --)")
    args = parser.parse_args()
    try:
        if args.cmd == "build":
            build(Path(args.source), Path(args.output))
        elif args.cmd == "run":
            with tempfile.TemporaryDirectory() as tmp:
                exe = build(Path(args.source), Path(tmp) / "program", executable=True)
                prog_args = args.args[1:] if args.args[:1] == ["--"] else args.args
                return subprocess.run([str(exe), *prog_args]).returncode
        else:
            print(translate(Path(args.source).read_text(), Path(args.source).resolve().parent))
    except TranslateError as e:
        print(f"cuemu: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
