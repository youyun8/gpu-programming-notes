"""Chapter 14, kernel 1: vector addition, the smallest complete Triton kernel.

Each program instance (the Triton analogue of a CUDA block) handles BLOCK
consecutive elements; `mask` guards the tail, like `if (i < n)` in CUDA.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # blockIdx.x
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # a vector of BLOCK indices
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    assert x.shape == y.shape and x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block),)              # gridDim.x
    add_kernel[grid](x, y, out, n, BLOCK=block)
    return out
