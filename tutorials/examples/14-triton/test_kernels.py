"""Checks the chapter 14 kernels against PyTorch, and optionally benchmarks them.

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

    for n in (1, 1000, 4097):
        x, y = rand(n), rand(n)
        check(f"add n={n}", add(x, y), x + y)

    for rows, cols in ((1, 1), (3, 100), (7, 1024), (2, 3000)):
        x = rand(rows, cols, scale=5.0)
        check(f"softmax {rows}x{cols}", softmax(x), torch.softmax(x.double(), dim=1).float())

    # On a GPU tl.dot uses TF32 for float32 inputs (10-bit mantissa): looser tolerance.
    tol = dict(rtol=2e-2, atol=2e-2) if DEVICE == "cuda" else dict(rtol=1e-4, atol=1e-4)
    for m, n, k in ((1, 1, 1), (33, 47, 29), (64, 96, 80)):
        a, b = rand(m, k), rand(k, n)
        check(f"matmul {m}x{n}x{k}", matmul(a, b), (a.double() @ b.double()).float(), **tol)
    a, b = rand(70, 50), rand(50, 90)
    check("matmul, GROUP_M=2 BLOCK_M=16", matmul(a, b, BLOCK_M=16, GROUP_M=2),
          (a.double() @ b.double()).float(), **tol)

    for n, d, causal in ((1, 16, False), (100, 32, False), (130, 64, True), (64, 16, True)):
        q, k, v = rand(n, d), rand(n, d), rand(n, d)
        got = flash_attention(q, k, v, causal=causal, block_q=32, block_kv=32)
        check(f"flash_attention n={n} d={d} causal={causal}", got, attention_reference(q, k, v, causal), **tol)

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
