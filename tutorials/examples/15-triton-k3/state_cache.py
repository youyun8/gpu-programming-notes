"""Indexed state-cache writes and reads, the core data movement in continuous batching."""
import torch
import triton
import triton.language as tl


@triton.jit
def scatter_rows_kernel(source_ptr, slots_ptr, cache_ptr, width,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    slot = tl.load(slots_ptr + row)
    values = tl.load(source_ptr + row * width + cols, mask=mask, other=0.0)
    tl.store(cache_ptr + slot * width + cols, values, mask=mask)


@triton.jit
def gather_rows_kernel(cache_ptr, slots_ptr, out_ptr, width,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    slot = tl.load(slots_ptr + row)
    values = tl.load(cache_ptr + slot * width + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * width + cols, values, mask=mask)


def scatter_rows(source: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor) -> None:
    """Write each row to its cache slot. Slots must be unique within this launch."""
    assert source.ndim == cache.ndim == 2 and source.shape[1] == cache.shape[1]
    assert slots.shape == (source.shape[0],) and slots.dtype == torch.int64
    assert source.is_contiguous() and slots.is_contiguous() and cache.is_contiguous()
    assert torch.unique(slots).numel() == slots.numel(), "duplicate slots would race"
    block = triton.next_power_of_2(source.shape[1])
    scatter_rows_kernel[(source.shape[0],)](source, slots, cache, source.shape[1], BLOCK=block)


def gather_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    assert cache.ndim == 2 and slots.ndim == 1 and slots.dtype == torch.int64
    assert cache.is_contiguous() and slots.is_contiguous()
    out = torch.empty((slots.numel(), cache.shape[1]), device=cache.device, dtype=cache.dtype)
    block = triton.next_power_of_2(cache.shape[1])
    gather_rows_kernel[(slots.numel(),)](cache, slots, out, cache.shape[1], BLOCK=block)
    return out
