"""A small, readable Triton kernel for one recurrent Kimi Delta Attention step.

Production K3 kernels fuse more preprocessing and use architecture-specific
layouts. This version keeps the recurrence visible and updates an FP32 state.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def kda_step_kernel(q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, state_ptr, out_ptr,
                    K: tl.constexpr, V, BK: tl.constexpr, BV: tl.constexpr):
    bh = tl.program_id(0)
    value_block = tl.program_id(1)
    ik = tl.arange(0, BK)
    iv = value_block * BV + tl.arange(0, BV)
    mask_k = ik < K
    mask_v = iv < V

    q = tl.load(q_ptr + bh * K + ik, mask=mask_k, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + bh * K + ik, mask=mask_k, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + bh * V + iv, mask=mask_v, other=0.0).to(tl.float32)
    alpha = tl.load(alpha_ptr + bh * K + ik, mask=mask_k, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + bh).to(tl.float32)

    state_offsets = bh * K * V + ik[:, None] * V + iv[None, :]
    state_mask = mask_k[:, None] & mask_v[None, :]
    state = tl.load(state_ptr + state_offsets, mask=state_mask, other=0.0).to(tl.float32)
    decayed = state * alpha[:, None]
    residual = v - tl.sum(decayed * k[:, None], axis=0)
    updated = decayed + k[:, None] * (beta * residual)[None, :]
    out = tl.sum(updated * q[:, None], axis=0)

    tl.store(state_ptr + state_offsets, updated, mask=state_mask)
    tl.store(out_ptr + bh * V + iv, out, mask=mask_v)


def kda_step(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
             beta: torch.Tensor, state: torch.Tensor, value_block: int = 32) -> torch.Tensor:
    """Update ``state`` in place and return one KDA output.

    Leading dimensions are flattened as batch × head. ``state`` must be FP32,
    because recurrent rounding error accumulates over the generated sequence.
    """
    assert q.shape == k.shape == alpha.shape
    assert q.shape[:-1] == v.shape[:-1] == beta.shape
    assert state.shape == (*q.shape, v.shape[-1]) and state.dtype == torch.float32
    assert all(t.is_contiguous() for t in (q, k, v, alpha, beta, state))
    k_size, v_size = q.shape[-1], v.shape[-1]
    bh = q.numel() // k_size
    out = torch.empty_like(v)
    kda_step_kernel[(bh, triton.cdiv(v_size, value_block))](
        q, k, v, alpha, beta, state, out,
        K=k_size, V=v_size, BK=triton.next_power_of_2(k_size), BV=value_block,
    )
    return out
