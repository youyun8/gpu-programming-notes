"""Chapter 14: fused row-wise LayerNorm.

Each program owns one row.  It computes mean and variance with block
reductions, then applies the affine transform without materialising any
intermediate tensors.  The mask makes non-power-of-two row widths safe.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr,
                      n_cols, x_stride, out_stride, eps,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    normalized = centered * tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = normalized * weight + bias
    tl.store(out_ptr + row * out_stride + cols, out, mask=mask)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    """Apply affine LayerNorm over the last dimension of a 2-D tensor."""
    assert x.dim() == 2 and x.stride(1) == 1
    rows, cols = x.shape
    assert cols > 0
    assert weight.shape == (cols,) and bias.shape == (cols,)
    assert weight.device == x.device and bias.device == x.device
    assert weight.is_contiguous() and bias.is_contiguous()

    out = torch.empty_like(x)
    if rows == 0:
        return out
    block = triton.next_power_of_2(cols)
    num_warps = 4 if block <= 2048 else (8 if block <= 8192 else 16)
    layer_norm_kernel[(rows,)](
        x, weight, bias, out, cols, x.stride(0), out.stride(0), eps,
        BLOCK=block, num_warps=num_warps,
    )
    return out
