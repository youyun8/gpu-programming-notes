"""Chapter 14, kernel 2: row-wise softmax, fused into one pass over memory.

One program per row; the whole row is loaded into registers as a vector of
BLOCK = next_power_of_2(n_cols) elements (masked lanes read -inf), so the
maximum, the exponentials, the sum and the division all happen on-chip and
the row is read once and written once. Compare with the three kernels of
chapter 13, section 2.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(in_ptr, out_ptr, n_cols, in_stride, out_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)          # a block-wide reduction, written as one call
    e = tl.exp(x)                      # masked lanes: exp(-inf) = 0
    tl.store(out_ptr + row * out_stride + cols, e / tl.sum(e, axis=0), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.stride(1) == 1
    rows, cols = x.shape
    block = triton.next_power_of_2(cols)
    # More warps for longer rows, so each thread holds a bounded number of elements.
    num_warps = 4 if block <= 2048 else (8 if block <= 8192 else 16)
    out = torch.empty_like(x)
    softmax_kernel[(rows,)](x, out, cols, x.stride(0), out.stride(0), BLOCK=block, num_warps=num_warps)
    return out
