"""Readable MXFP4-style block quantize/dequantize for teaching.

The kernel uses 32-value groups, an E8M0-like power-of-two scale, and the
E2M1 finite magnitudes. It returns dequantized values instead of Quark's packed
checkpoint representation, so it is useful for numerical experiments but is
not a checkpoint converter.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def mxfp4_qdq_kernel(x_ptr, out_ptr, n, GROUP: tl.constexpr):
    group_id = tl.program_id(0)
    offsets = group_id * GROUP + tl.arange(0, GROUP)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    max_abs = tl.max(tl.abs(x), axis=0)
    # The largest finite E2M1 value is 6. Clamp the all-zero group to scale 1.
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(max_abs / 6.0, 2.0 ** -126))))
    scale = tl.where(max_abs == 0.0, 1.0, scale)
    magnitude = tl.abs(x) / scale

    # Nearest of {0, .5, 1, 1.5, 2, 3, 4, 6}; thresholds are midpoints.
    q = tl.where(magnitude < 0.25, 0.0, 0.5)
    q = tl.where(magnitude >= 0.75, 1.0, q)
    q = tl.where(magnitude >= 1.25, 1.5, q)
    q = tl.where(magnitude >= 1.75, 2.0, q)
    q = tl.where(magnitude >= 2.5, 3.0, q)
    q = tl.where(magnitude >= 3.5, 4.0, q)
    q = tl.where(magnitude >= 5.0, 6.0, q)
    q = tl.where(x < 0.0, -q, q)
    tl.store(out_ptr + offsets, q * scale, mask=mask)


def mxfp4_qdq(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    assert x.is_contiguous() and group_size == 32
    out = torch.empty_like(x)
    mxfp4_qdq_kernel[(triton.cdiv(x.numel(), group_size),)](
        x, out, x.numel(), GROUP=group_size,
    )
    return out
