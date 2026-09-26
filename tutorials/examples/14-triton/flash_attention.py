"""Chapter 14, kernel 4: FlashAttention forward (one head), as in chapter 13, section 5.

O = softmax(Q K^T * scale [+ causal mask]) V for Q, K, V of shape (N, D). Each
program owns BLOCK_Q query rows and streams K and V past them in tiles of
BLOCK_KV rows, keeping the running maximum m, the running sum l and the
unnormalised output acc (the online-softmax state) in registers. The N x N
score matrix never exists in memory. D must be a power of two (16..128).
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, n, scale,
                           stride_q, stride_k, stride_v, stride_o,
                           D: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, CAUSAL: tl.constexpr):
    q_rows = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + q_rows[:, None] * stride_q + dims[None, :], mask=q_rows[:, None] < n, other=0.0)
    q = q * scale                                          # fold the scale into Q once

    m = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, D), dtype=tl.float32)
    # Causal: key tiles entirely after the last query row of this block contribute nothing.
    kv_end = tl.minimum(n, (tl.program_id(0) + 1) * BLOCK_Q) if CAUSAL else n
    for start in range(0, kv_end, BLOCK_KV):
        kv_rows = start + tl.arange(0, BLOCK_KV)
        k = tl.load(k_ptr + kv_rows[:, None] * stride_k + dims[None, :], mask=kv_rows[:, None] < n, other=0.0)
        v = tl.load(v_ptr + kv_rows[:, None] * stride_v + dims[None, :], mask=kv_rows[:, None] < n, other=0.0)
        s = tl.dot(q, tl.trans(k))                         # (BLOCK_Q, BLOCK_KV) scores, on-chip
        valid = kv_rows[None, :] < n
        if CAUSAL:
            valid = valid & (kv_rows[None, :] <= q_rows[:, None])
        s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        # Rows with no valid key yet keep m_new = -inf; subtracting 0 avoids inf - inf = NaN.
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m - m_safe)                         # rescales the old state (0 at the start)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + q_rows[:, None] * stride_o + dims[None, :], out.to(o_ptr.dtype.element_ty),
             mask=q_rows[:, None] < n)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
                    block_q: int = 64, block_kv: int = 64) -> torch.Tensor:
    n, d = q.shape
    assert k.shape == (n, d) and v.shape == (n, d) and d == triton.next_power_of_2(d) and d >= 16
    assert q.stride(1) == k.stride(1) == v.stride(1) == 1
    o = torch.empty_like(q)
    grid = (triton.cdiv(n, block_q),)
    flash_attention_kernel[grid](q, k, v, o, n, 1.0 / math.sqrt(d),
                                 q.stride(0), k.stride(0), v.stride(0), o.stride(0),
                                 D=d, BLOCK_Q=block_q, BLOCK_KV=block_kv, CAUSAL=causal)
    return o
