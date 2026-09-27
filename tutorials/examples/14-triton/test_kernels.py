"""Checks every chapter 14 kernel against PyTorch, including masked edges.

    python3 test_kernels.py            # on a GPU; without one, runs in the Triton interpreter
    python3 test_kernels.py --bench    # also times the kernels (GPU only)

Without a CUDA device the kernels run in Triton's interpreter
(TRITON_INTERPRET=1), which executes them with NumPy on the CPU: slow, but it
checks the indexing, masks and results exactly like the CPU emulator does for
the CUDA examples.
"""
import os
import sys

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")   # must be set before triton is imported

import triton  # noqa: E402

from flash_attention import flash_attention  # noqa: E402
from histogram import histogram  # noqa: E402
from layer_norm import layer_norm  # noqa: E402
from matmul import matmul  # noqa: E402
from softmax import softmax  # noqa: E402
from vector_add import add  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
failures = 0


def check(name, got, ref, rtol=1e-4, atol=1e-5):
    global failures
    ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
    err = (got - ref).abs().max().item() if got.numel() else 0.0
    print(f"{'ok  ' if ok else 'FAIL'} {name} (max abs error {err:.3g})")
    failures += not ok


def attention_reference(q, k, v, causal):
    s = (q.double() @ k.double().T) / q.shape[1] ** 0.5
    if causal:
        s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1), float("-inf"))
    return (torch.softmax(s, dim=1) @ v.double()).float()


def main():
    gen = torch.Generator().manual_seed(0)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(DEVICE)

    for n in (0, 1, 17, 1000, 4097):
        x, y = rand(n), rand(n)
        check(f"add n={n}", add(x, y), x + y)

    for rows, cols in ((0, 7), (1, 1), (3, 37), (3, 100), (7, 1024), (2, 3001)):
        x = rand(rows, cols, scale=5.0)
        check(f"softmax {rows}x{cols}", softmax(x), torch.softmax(x.double(), dim=1).float())
    x = rand(4, 42, scale=5.0)[:, :37]  # contiguous columns, padded row stride
    check("softmax padded stride", softmax(x), torch.softmax(x.double(), dim=1).float())

    # On a GPU tl.dot uses TF32 for float32 inputs (10-bit mantissa): looser tolerance.
    tol = dict(rtol=2e-2, atol=2e-2) if DEVICE == "cuda" else dict(rtol=1e-4, atol=1e-4)
    for m, n, k in ((0, 7, 3), (3, 0, 5), (3, 5, 0), (1, 1, 1),
                    (33, 47, 29), (64, 96, 80)):
        a, b = rand(m, k), rand(k, n)
        check(f"matmul {m}x{n}x{k}", matmul(a, b), (a.double() @ b.double()).float(), **tol)
    a, b = rand(70, 50), rand(50, 90)
    check("matmul, GROUP_M=2 BLOCK_M=16", matmul(a, b, BLOCK_M=16, GROUP_M=2),
          (a.double() @ b.double()).float(), **tol)
    a, b = rand(23, 17).T, rand(19, 23).T
    check("matmul strided 17x19x23", matmul(a, b),
          (a.double() @ b.double()).float(), **tol)

    for n, d, causal in ((0, 16, False), (1, 16, False), (37, 32, False),
                         (100, 32, False), (129, 64, True), (64, 16, True)):
        q, k, v = rand(n, d), rand(n, d), rand(n, d)
        got = flash_attention(q, k, v, causal=causal, block_q=32, block_kv=32)
        check(f"flash_attention n={n} d={d} causal={causal}", got, attention_reference(q, k, v, causal), **tol)
    q, k, v = (rand(33, 37)[:, :32] for _ in range(3))
    check("flash_attention padded stride", flash_attention(q, k, v, causal=True, block_q=32, block_kv=32),
          attention_reference(q, k, v, True), **tol)

    norm_tol = dict(rtol=5e-4, atol=5e-4)
    for rows, cols in ((0, 7), (1, 1), (3, 37), (5, 513)):
        x = rand(rows, cols, scale=3.0)
        weight, bias = rand(cols), rand(cols)
        got = layer_norm(x, weight, bias)
        ref = torch.nn.functional.layer_norm(
            x.double(), (cols,), weight.double(), bias.double(), 1e-5,
        ).float()
        check(f"layer_norm {rows}x{cols}", got, ref, **norm_tol)
    x, weight, bias = rand(3, 42)[:, :37], rand(37), rand(37)
    ref = torch.nn.functional.layer_norm(
        x.double(), (37,), weight.double(), bias.double(), 1e-5,
    ).float()
    check("layer_norm padded stride", layer_norm(x, weight, bias), ref, **norm_tol)

    histogram_cases = (
        torch.empty(0, dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([-3, -1, 0, 1, 1, 6, 7, 99], dtype=torch.int32),
        torch.tensor([(i * 17) % 13 for i in range(513)], dtype=torch.int64),
    )
    for i, values in enumerate(histogram_cases):
        values = values.to(DEVICE)
        n_bins = 7 if i < 3 else 13
        valid = values[(values >= 0) & (values < n_bins)].to(torch.int64)
        ref = torch.bincount(valid, minlength=n_bins).to(torch.int32)
        check(f"histogram case={i} n={values.numel()}", histogram(values, n_bins), ref)

    if "--bench" in sys.argv and DEVICE == "cuda":
        bench()
    print(f"14-triton: {'FAILED' if failures else 'all checks passed'}")
    return 1 if failures else 0


def bench():
    x = torch.randn(1 << 26, device="cuda")
    ms = triton.testing.do_bench(lambda: add(x, x))
    print(f"add            {ms:8.3f} ms  {3 * x.numel() * 4 / ms * 1e-6:7.1f} GB/s")
    s = torch.randn(4096, 4096, device="cuda")
    ms = triton.testing.do_bench(lambda: softmax(s))
    print(f"softmax        {ms:8.3f} ms  {2 * s.numel() * 4 / ms * 1e-6:7.1f} GB/s")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: matmul(a, a))
    print(f"matmul (fp16)  {ms:8.3f} ms  {2 * 4096 ** 3 / ms * 1e-9:7.1f} TFLOP/s")
    q = torch.randn(8192, 64, device="cuda", dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: flash_attention(q, q, q, causal=True))
    print(f"flash (causal) {ms:8.3f} ms  {2 * 2 * 8192 ** 2 * 64 / 2 / ms * 1e-9:7.1f} TFLOP/s")


if __name__ == "__main__":
    sys.exit(main())
