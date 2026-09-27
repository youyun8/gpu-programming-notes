"""Chapter 14: an atomic histogram with masked tails and invalid-bin handling.

Every program reads a block of input values.  Valid values atomically add one
to their bin; negative and out-of-range values are deliberately ignored.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def histogram_kernel(values_ptr, histogram_ptr, n, n_bins,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < n
    values = tl.load(values_ptr + offsets, mask=in_bounds, other=-1)
    valid = in_bounds & (values >= 0) & (values < n_bins)
    # Keep even masked pointer arithmetic in range; only valid lanes update.
    bins = tl.where(valid, values, 0)
    tl.atomic_add(histogram_ptr + bins, 1, mask=valid)


def histogram(values: torch.Tensor, n_bins: int,
              block: int = 256) -> torch.Tensor:
    """Count values in ``[0, n_bins)`` and ignore all other values."""
    assert values.dim() == 1 and values.is_contiguous()
    assert values.dtype in (torch.int32, torch.int64)
    assert n_bins > 0
    out = torch.zeros(n_bins, device=values.device, dtype=torch.int32)
    if values.numel() == 0:
        return out
    grid = (triton.cdiv(values.numel(), block),)
    histogram_kernel[grid](
        values, out, values.numel(), n_bins, BLOCK=block,
    )
    return out
