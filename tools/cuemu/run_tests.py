#!/usr/bin/env python3
"""Run solutions against the official LeetGPU / Tensara reference implementations on the CPU.

Each solution.cu is compiled for the cuemu CPU emulator and executed on the
platform's own functional test cases (LeetGPU) or on the test cases that are
small enough to emulate (Tensara, plus hand-picked scaled-down cases from
tensara_small_cases.py). Results are compared with the platform's reference
implementation and tolerance.

Usage:
    python3 tools/cuemu/run_tests.py leetgpu/001-vector-add tensara/relu
    python3 tools/cuemu/run_tests.py --all
    python3 tools/cuemu/run_tests.py --all --platform tensara --jobs 8

Upstream problem definitions are looked up in (override with env vars):
    LEETGPU_CHALLENGES  default: .upstream/leetgpu-challenges
    TENSARA_PROBLEMS    default: .upstream/tensara-problems
    TENSARA_ENGINE      default: .upstream/tensara/engine
Run scripts/fetch_upstream.sh to clone them.
"""
import argparse
import concurrent.futures as cf
import ctypes
import importlib.util
import json
import mmap
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import cuemu  # noqa: E402

UPSTREAM = ROOT / ".upstream"
LEETGPU_DIR = Path(os.environ.get("LEETGPU_CHALLENGES", UPSTREAM / "leetgpu-challenges"))
TENSARA_PROBLEMS = Path(os.environ.get("TENSARA_PROBLEMS", UPSTREAM / "tensara-problems"))
TENSARA_ENGINE = Path(os.environ.get("TENSARA_ENGINE", UPSTREAM / "tensara" / "engine"))
BUILD_CACHE = ROOT / "build" / "cuemu"

MAX_ELEMENTS = int(os.environ.get("CUEMU_MAX_ELEMENTS", 1 << 22))
TEST_TIMEOUT = float(os.environ.get("CUEMU_TIMEOUT", 300))

PAGE = mmap.PAGESIZE
_libc = ctypes.CDLL(None, use_errno=True)
_libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


class TooLarge(Exception):
    pass


# When True, oversized factory calls return shape-only "meta" tensors instead of
# raising, so one huge case does not abort a whole test-case list.
META_MODE = False


# ---------------------------------------------------------------------------
# Guarded device buffers: data ends right before a PROT_NONE page, and the
# alignment padding in between is filled with a canary so out-of-bounds writes
# are detected and out-of-bounds reads see NaNs.
# ---------------------------------------------------------------------------
class GuardedBuffer:
    CANARY = 0xFF

    def __init__(self, tensor):
        self.tensor = tensor
        self.nbytes = tensor.numel() * tensor.element_size()
        self.padded = max(256, (self.nbytes + 255) // 256 * 256)
        total = (self.padded + PAGE - 1) // PAGE * PAGE + PAGE
        self.map = mmap.mmap(-1, total, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.map))
        self.guard = base + total - PAGE
        if _libc.mprotect(self.guard, PAGE, 0) != 0:
            raise OSError("mprotect failed")
        self.addr = self.guard - self.padded
        ctypes.memset(self.addr, self.CANARY, self.padded)
        if self.nbytes:
            ctypes.memmove(self.addr, tensor.contiguous().data_ptr(), self.nbytes)

    def canary_ok(self) -> bool:
        tail = ctypes.string_at(self.addr + self.nbytes, self.padded - self.nbytes)
        return all(b == self.CANARY for b in tail)

    def read_into(self, tensor):
        if self.nbytes:
            ctypes.memmove(tensor.data_ptr(), self.addr, self.nbytes)

    def same_as_original(self) -> bool:
        return ctypes.string_at(self.addr, self.nbytes) == ctypes.string_at(self.tensor.contiguous().data_ptr(), self.nbytes)


# ---------------------------------------------------------------------------
# Problem metadata
# ---------------------------------------------------------------------------
def read_front_matter(readme: Path) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", readme.read_text(), re.S)
    meta = {}
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta


def discover(paths, platform=None):
    found = []
    for p in paths:
        p = Path(p)
        if (p / "solution.cu").exists() or (p / "solution.py").exists():
            found.append(p)
        else:
            found += sorted({d.parent for d in p.glob("*/solution.*") if d.suffix in (".cu", ".py")})
    if platform:
        found = [d for d in found if d.parent.name == platform]
    return found


# ---------------------------------------------------------------------------
# Torch helpers (imported lazily so `--help` works without torch)
# ---------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def size_guard():
    """Make torch factory functions raise TooLarge for tensors above MAX_ELEMENTS."""
    torch = _torch()
    if getattr(torch, "_cuemu_guarded", False):
        return
    names = ["rand", "randn", "randint", "zeros", "ones", "empty", "full", "randperm"]

    def numel_of(args, kwargs, name):
        shape = kwargs.get("size")
        if shape is None:
            pos = list(args)
            if name == "randint":
                pos = [a for a in pos if isinstance(a, (tuple, list, torch.Size))]
            elif name == "full":
                pos = pos[:1]
            elif name == "randperm":
                pos = pos[:1]
            if len(pos) == 1 and isinstance(pos[0], (tuple, list, torch.Size)):
                shape = pos[0]
            else:
                shape = [a for a in pos if isinstance(a, int)]
        n = 1
        for d in shape:
            n *= int(d)
        return n

    for name in names:
        orig = getattr(torch, name)

        def wrapper(*args, __orig=orig, __name=name, **kwargs):
            if numel_of(args, kwargs, __name) > MAX_ELEMENTS:
                if META_MODE:
                    kwargs = {k: v for k, v in kwargs.items() if k != "generator"}
                    kwargs["device"] = "meta"
                    return __orig(*args, **kwargs)
                raise TooLarge(f"tensor larger than {MAX_ELEMENTS} elements")
            return __orig(*args, **kwargs)

        setattr(torch, name, wrapper)
    torch._cuemu_guarded = True


def clone_arg(v):
    torch = _torch()
    return v.clone() if isinstance(v, torch.Tensor) else v


def compare(expected, actual, atol, rtol):
    torch = _torch()
    if expected.shape != actual.shape:
        return False, f"shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}"
    e = expected.double() if expected.is_floating_point() else expected.long()
    a = actual.double() if actual.is_floating_point() else actual.long()
    if expected.is_floating_point():
        ok = torch.allclose(a, e, atol=atol, rtol=rtol, equal_nan=True)
    else:
        ok = torch.equal(a, e) if atol == 0 else torch.allclose(a.double(), e.double(), atol=atol, rtol=rtol)
    if ok:
        return True, ""
    diff = (a.double() - e.double()).abs()
    diff = torch.nan_to_num(diff, nan=float("inf"))
    idx = int(diff.argmax())
    return False, (f"max abs diff {float(diff.max()):.3g} at flat index {idx}: "
                   f"got {a.flatten()[idx].item()}, expected {e.flatten()[idx].item()}")


def call_solution(lib_path, entry, argtypes, values):
    """values: list of torch tensors / python scalars. Returns list of GuardedBuffer (or None)."""
    torch = _torch()
    lib = ctypes.CDLL(str(lib_path))
    fn = getattr(lib, entry)
    fn.restype = None
    call_args, buffers = [], []
    for v, t in zip(values, argtypes):
        if isinstance(v, torch.Tensor):
            buf = GuardedBuffer(v)
            buffers.append(buf)
            call_args.append(ctypes.c_void_p(buf.addr))
        else:
            buffers.append(None)
            call_args.append(t(v))
    fn.argtypes = [ctypes.c_void_p if b is not None else t for b, t in zip(buffers, argtypes)]
    fn(*call_args)
    return buffers


# ---------------------------------------------------------------------------
# LeetGPU
# ---------------------------------------------------------------------------
def load_leetgpu(upstream_id):
    challenge_dir = LEETGPU_DIR / "challenges" / upstream_id
    sys.path.insert(0, str(LEETGPU_DIR / "challenges"))
    spec = importlib.util.spec_from_file_location("challenge_" + upstream_id.replace("/", "_"), challenge_dir / "challenge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Challenge(device="cpu")


def materialize(v):
    torch = _torch()
    from core import challenge_base as cb  # type: ignore

    def dt(name):
        return getattr(torch, name)

    if isinstance(v, cb.RandTensor):
        t = torch.empty(v.shape, dtype=dt(v.dtype))
        return t.uniform_(v.low, v.high) if t.is_floating_point() else t.random_(int(v.low), int(v.high))
    if isinstance(v, cb.RandnTensor):
        return torch.empty(v.shape, dtype=dt(v.dtype)).normal_(v.mean, v.std)
    if isinstance(v, cb.RandIntTensor):
        return torch.randint(v.low, v.high, v.shape, dtype=dt(v.dtype))
    if isinstance(v, cb.FullTensor):
        return torch.full(v.shape, v.value, dtype=dt(v.dtype))
    if isinstance(v, cb.OutTensor):
        t = torch.empty(v.shape, dtype=dt(v.dtype))
        return t.fill_(float("nan")) if t.is_floating_point() else t.fill_(-12345 if t.dtype.is_signed else 123)
    return v


def leetgpu_cases(ch):
    global META_MODE
    torch = _torch()
    torch.manual_seed(0)
    META_MODE = True
    try:
        example = ch.generate_example_test()
        functional = ch.generate_functional_test()
    finally:
        META_MODE = False
    cases = [("example", example)] + [(f"functional[{i}]", c) for i, c in enumerate(functional)]
    return cases


def is_meta_case(case):
    torch = _torch()
    return any(isinstance(v, torch.Tensor) and v.device.type == "meta" for v in case.values())


def _top_p_nucleus_check(case, outputs):
    """torch.multinomial's RNG stream cannot be reproduced from CUDA, so for
    top-p sampling we check that the sampled token lies in the nucleus."""
    torch = _torch()
    logits, p = case["logits"].double(), float(case["p"][0])
    probs = torch.softmax(logits, dim=0)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cutoff = min(int(torch.searchsorted(torch.cumsum(sorted_probs, 0), p)) + 1, len(probs))
    threshold = float(sorted_probs[cutoff - 1])
    allowed = set(int(i) for i in torch.nonzero(probs >= threshold * (1 - 1e-6)).flatten())
    token = int(outputs["sampled_token"][0])
    if token in allowed:
        return True, ""
    return False, f"sampled token {token} is not in the nucleus {sorted(allowed)[:10]}"


# Challenges whose reference output cannot be reproduced bit-for-bit.
CUSTOM_LEETGPU_CHECKS = {"medium/60_top_p_sampling": _top_p_nucleus_check}


def run_leetgpu_case(lib_path, upstream_id, index):
    torch = _torch()
    size_guard()
    ch = load_leetgpu(upstream_id)
    name, case = leetgpu_cases(ch)[index]
    if is_meta_case(case):
        return None, f"{name}: skipped (too large to emulate)"
    sig = ch.get_solve_signature()
    case = {k: materialize(v) for k, v in case.items()}
    if any(isinstance(v, _torch().Tensor) and v.numel() > MAX_ELEMENTS for v in case.values()):
        return None, f"{name}: skipped (too large to emulate)"
    ref_args = {k: clone_arg(v) for k, v in case.items()}
    ch.reference_impl(**ref_args)
    names = list(sig.keys())
    argtypes = [sig[n][0] for n in names]
    values = [case[n] for n in names]
    buffers = call_solution(lib_path, "solve", argtypes, values)
    custom = CUSTOM_LEETGPU_CHECKS.get(upstream_id)
    outputs = {}
    for n, buf in zip(names, buffers):
        if buf is None:
            continue
        if not buf.canary_ok():
            return False, f"{name}: wrote past the end of '{n}'"
        direction = sig[n][1]
        if direction == "in":
            if not buf.same_as_original():
                return False, f"{name}: modified read-only input '{n}'"
            continue
        out = torch.empty_like(case[n])
        buf.read_into(out)
        outputs[n] = out
        if custom:
            continue
        ok, msg = compare(ref_args[n], out, ch.atol, ch.rtol)
        if not ok:
            return False, f"{name}: output '{n}' mismatch: {msg}"
    if custom:
        ok, msg = custom(case, outputs)
        if not ok:
            return False, f"{name}: {msg}"
    return True, name


def run_leetgpu_py_case(solution_py, upstream_id, index):
    """Challenges that only offer Python frameworks: call solve() on CPU tensors."""
    torch = _torch()
    size_guard()
    ch = load_leetgpu(upstream_id)
    name, case = leetgpu_cases(ch)[index]
    if is_meta_case(case):
        return None, f"{name}: skipped (too large to emulate)"
    sig = ch.get_solve_signature()
    case = {k: materialize(v) for k, v in case.items()}
    ref_args = {k: clone_arg(v) for k, v in case.items()}
    ch.reference_impl(**ref_args)
    spec = importlib.util.spec_from_file_location("solution_py", solution_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sol_args = {k: clone_arg(v) for k, v in case.items()}
    mod.solve(**sol_args)
    for n, (_, direction) in sig.items():
        if direction == "in" or not isinstance(sol_args[n], torch.Tensor):
            continue
        ok, msg = compare(ref_args[n], sol_args[n], ch.atol, ch.rtol)
        if not ok:
            return False, f"{name}: output '{n}' mismatch: {msg}"
    return True, name


def count_leetgpu_cases(upstream_id):
    size_guard()
    return len(leetgpu_cases(load_leetgpu(upstream_id)))


# ---------------------------------------------------------------------------
# Tensara
# ---------------------------------------------------------------------------
def load_tensara(slug):
    sys.path.insert(0, str(TENSARA_ENGINE))
    import problem as problem_base  # type: ignore
    import lowp_reference  # CPU stand-ins for flashinfer / swizzled scaled_mm

    lowp_reference.install()

    path = TENSARA_PROBLEMS / "problems" / slug / "def.py"
    src = path.read_text()
    src = re.sub(r"device\s*=\s*['\"]cuda['\"]", 'device="cpu"', src)
    src = src.replace('autocast("cuda"', 'autocast("cpu"').replace(".cuda()", ".cpu()")
    src = re.sub(r"""\.to\(\s*['"]cuda['"]\s*\)""", ".to('cpu')", src)
    src = re.sub(r"""torch\.device\(\s*['"]cuda['"]\s*\)""", "torch.device('cpu')", src)
    src = re.sub(r"""\.type\s*==\s*['"]cuda['"]""", ".type == 'cpu'", src)
    src = src.replace("torch.cuda.synchronize()", "None")
    mod = types.ModuleType("tensara_" + slug.replace("-", "_"))
    exec(compile(src, str(path), "exec"), mod.__dict__)
    classes = [v for v in mod.__dict__.values()
               if isinstance(v, type) and issubclass(v, problem_base.Problem) and v is not problem_base.Problem]
    return classes[0]()


def tensara_cases(prob, slug):
    import tensara_small_cases  # type: ignore

    cases = [("sample", prob.generate_sample())]
    for tc in prob.generate_test_cases():
        cases.append((tc.get("name", "test"), tc))
    cases += [(f"small:{tc.get('name', i)}", tc) for i, tc in enumerate(tensara_small_cases.extra_cases(prob, slug))]
    return cases


def run_tensara_case(lib_path, slug, index):
    import warnings

    warnings.filterwarnings("ignore")
    torch = _torch()
    size_guard()
    prob = load_tensara(slug)
    name, tc = tensara_cases(prob, slug)[index]
    try:
        inputs = tc["create_inputs"]()
    except TooLarge:
        return None, f"{name}: skipped (too large to emulate)"
    inputs = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    ref_inputs = tuple(clone_arg(v) for v in inputs)
    with torch.no_grad():
        ref = prob.reference_solution(*ref_inputs)
    expected = (ref,) if isinstance(ref, torch.Tensor) else tuple(ref)
    actual = [torch.zeros_like(e).contiguous() for e in expected]
    extra = list(prob.get_extra_params(tc))
    argtypes = prob.get_function_signature()["argtypes"]
    values = list(inputs) + actual + extra
    if len(values) != len(argtypes):
        return False, f"{name}: argument count mismatch ({len(values)} values vs {len(argtypes)} parameters)"
    buffers = call_solution(lib_path, "solution", argtypes, values)
    n_in = len(inputs)
    for i, buf in enumerate(buffers):
        if buf is None:
            continue
        if not buf.canary_ok():
            return False, f"{name}: wrote past the end of argument {i}"
        if i < n_in and not buf.same_as_original():
            return False, f"{name}: modified read-only input argument {i}"
    for j, out in enumerate(actual):
        buffers[n_in + j].read_into(out)
    if len(expected) == 1:
        ok, info = prob.verify_result(expected[0], actual[0])
    else:
        ok, info = prob.verify_result(expected, tuple(actual))
    if not ok:
        return False, f"{name}: verify_result failed: {json.dumps(info, default=str)[:300]}"
    return True, name


def count_tensara_cases(slug):
    return len(tensara_cases(load_tensara(slug), slug))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _child(conn, fn, args, max_elements):
    global MAX_ELEMENTS
    MAX_ELEMENTS = max_elements
    try:
        conn.send(fn(*args))
    except TooLarge:
        conn.send((None, "skipped (too large to emulate)"))
    except Exception:
        conn.send((False, traceback.format_exc(limit=4)[-1500:]))
    finally:
        conn.close()


def isolated(fn, *args, timeout=TEST_TIMEOUT, max_elements=None):
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, fn, args, max_elements or MAX_ELEMENTS))
    proc.start()
    child.close()
    if parent.poll(timeout):
        try:
            result = parent.recv()
        except EOFError:
            result = (False, "crashed")
    else:
        proc.kill()
        result = (False, f"timed out after {timeout:.0f}s")
    proc.join()
    if proc.exitcode not in (0, None) and result == (False, "crashed"):
        result = (False, f"crashed (signal {-proc.exitcode})" if proc.exitcode < 0 else "crashed")
    return result


def test_problem(problem_dir: Path, reverse: bool):
    meta = read_front_matter(problem_dir / "README.md")
    platform = problem_dir.parent.name
    upstream = meta.get("upstream")
    # Problems whose inputs are large even in their smallest tests (e.g. packed
    # transformer weights) can raise the emulation limit in their front matter.
    limit = int(meta.get("cuemu_max_elements", MAX_ELEMENTS))
    if not upstream:
        return problem_dir, [(False, "README front matter has no 'upstream' key")], 0.0
    t0 = time.time()
    if (problem_dir / "solution.py").exists():
        n = isolated(count_leetgpu_cases, upstream, max_elements=limit)
        if not isinstance(n, int):
            return problem_dir, [(False, f"could not load test cases: {n[1]}")], time.time() - t0
        results = [isolated(run_leetgpu_py_case, problem_dir / "solution.py", upstream, i, max_elements=limit) for i in range(n)]
        return problem_dir, results, time.time() - t0
    try:
        lib = cuemu.cached_build(problem_dir / "solution.cu", BUILD_CACHE)
    except cuemu.TranslateError as e:
        return problem_dir, [(False, str(e))], time.time() - t0
    if reverse:
        os.environ["CUEMU_REVERSE"] = "1"
    if platform == "leetgpu":
        n = isolated(count_leetgpu_cases, upstream, max_elements=limit)
        runner = run_leetgpu_case
    else:
        n = isolated(count_tensara_cases, upstream, max_elements=limit)
        runner = run_tensara_case
    if not isinstance(n, int):
        return problem_dir, [(False, f"could not load test cases: {n[1]}")], time.time() - t0
    results = [isolated(runner, lib, upstream, i, max_elements=limit) for i in range(n)]
    return problem_dir, results, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--all", action="store_true", help="test every problem in leetgpu/ and tensara/")
    parser.add_argument("--platform", choices=["leetgpu", "tensara"])
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count())
    parser.add_argument("--reverse", action="store_true", help="schedule threads in reverse order (race detection)")
    parser.add_argument("--json", help="write a JSON report to this path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Import torch once in the parent so forked test processes start fast.
    torch = _torch()
    torch.set_num_threads(1)
    import warnings
    warnings.filterwarnings("ignore")

    paths = args.paths or []
    if args.all:
        paths += [ROOT / "leetgpu", ROOT / "tensara"]
    problems = discover(paths, args.platform)
    if not problems:
        parser.error("no problems selected")

    report, failed = {}, 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(test_problem, p, args.reverse) for p in problems]
        for fut in cf.as_completed(futures):
            pdir, results, dt = fut.result()
            passed = sum(1 for ok, _ in results if ok)
            skipped = sum(1 for ok, _ in results if ok is None)
            bad = [msg for ok, msg in results if ok is False]
            status = "PASS" if not bad and passed else ("FAIL" if bad else "SKIP")
            failed += status != "PASS"
            rel = pdir.relative_to(ROOT).as_posix() if pdir.is_relative_to(ROOT) else str(pdir)
            print(f"{status}  {rel:55} {passed} passed, {len(bad)} failed, {skipped} skipped  ({dt:.1f}s)", flush=True)
            for msg in bad[:3] if not args.verbose else bad:
                print("      " + msg.strip().replace("\n", "\n      "))
            report[rel] = {"status": status, "passed": passed, "failed": len(bad), "skipped": skipped, "errors": bad}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\n{len(problems) - failed}/{len(problems)} problems passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
