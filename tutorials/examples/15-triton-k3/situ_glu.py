"""A teaching implementation of Kimi K3's fused SiTU-GLU activation."""
import torch
import triton
import triton.language as tl


@triton.jit
def situ_glu_kernel(x_ptr, out_ptr, n, BETA1: tl.constexpr, BETA2: tl.constexpr,
                    BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    gate = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + n + offsets, mask=mask, other=0.0).to(tl.float32)
    # tanh(x) = 2 sigmoid(2x) - 1 keeps the kernel portable across backends.
    bounded_gate = BETA1 * (2.0 / (1.0 + tl.exp(-2.0 * gate / BETA1)) - 1.0)
    bounded_up = BETA2 * (2.0 / (1.0 + tl.exp(-2.0 * up / BETA2)) - 1.0)
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(out_ptr + offsets, bounded_gate * sigmoid_gate * bounded_up, mask=mask)


def situ_glu(x: torch.Tensor, beta1: float = 4.0, beta2: float = 25.0) -> torch.Tensor:
    """Apply SiTU-GLU to a contiguous ``[gate, up]`` vector."""
    assert x.is_contiguous() and x.numel() % 2 == 0
    n = x.numel() // 2
    out = torch.empty(n, device=x.device, dtype=x.dtype)
    block = 256
    situ_glu_kernel[(triton.cdiv(n, block),)](
        x, out, n, BETA1=beta1, BETA2=beta2, BLOCK=block,
    )
    return out
