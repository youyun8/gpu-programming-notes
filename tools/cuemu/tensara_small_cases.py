"""Scaled-down Tensara test cases.

Tensara's official test cases are benchmark-sized (e.g. 8192 x 8192), far too
large to emulate on a CPU. Most generators close over their sizes through
lambda default arguments that mirror integer fields of the test-case dict
(rows/cols/dims/...), so a smaller but otherwise identical case can be built by
mapping every "large" integer through the same function in both places:

  * scaled: v -> v / 32 (or v / 1024 above 65536) - keeps divisibility;
  * odd:    scaled + 3 - exercises tails and boundary handling.

Problems whose shapes must stay divisible (block-scaled formats, pooling with
exact windows, ...) only get the scaled variant. Problems whose generator does
not follow the pattern can register a custom builder in CUSTOM.
"""
import inspect

SMALLEST_SCALED = 64  # integers below this are parameters (kernel size, dim, ...), not sizes

# Problems where "+3" would violate a shape constraint of the problem itself.
NO_ODD = {
    "mxfp4-dequantize", "mxfp4-quantize", "mxfp4-gemm", "mxfp8-dequantize", "mxfp8-quantize", "mxfp8-gemm",
    "nvfp4-dequantize", "nvfp4-quantize", "nvfp4-gemm", "nvfp4-gemv", "poly-multiply-ff", "vector-multiply-ff",
    "ecc-point-negation", "array-sort", "conv-1d", "conv-2d",
}


def _scale(v):
    if v < SMALLEST_SCALED:
        return v
    return max(v // 1024, 16) if v > 65536 else max(v // 32, 2)


def _odd(v):
    s = _scale(v)
    return s + 3 if v >= SMALLEST_SCALED else v


def _map(value, fn):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map(v, fn) for v in value)
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    return value


def shrink(tc, fn, label):
    create = tc["create_inputs"]
    try:
        params = inspect.signature(create).parameters
    except (TypeError, ValueError):
        return None
    overrides = {}
    for name, p in params.items():
        if name in ("seed", "dtype", "g") or p.default is inspect.Parameter.empty:
            continue
        mapped = _map(p.default, fn)
        if mapped != p.default:
            overrides[name] = mapped
    if not overrides:
        return None
    new = {k: (_map(v, fn) if k not in ("name", "create_inputs") else v) for k, v in tc.items()}
    new["name"] = f"{label}: {tc.get('name', 'case')}"
    new["create_inputs"] = lambda: create(**overrides)
    return new


CUSTOM = {}


def extra_cases(prob, slug):
    if slug in CUSTOM:
        return CUSTOM[slug](prob)
    cases = []
    for tc in prob.generate_test_cases():
        for fn, label in ((_scale, "scaled"), (_odd, "odd")):
            if label == "odd" and slug in NO_ODD:
                continue
            small = shrink(tc, fn, label)
            if small is not None:
                cases.append(small)
    return cases
