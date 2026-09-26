"""Check the chapter 15 teaching kernels against direct PyTorch references."""
import os
import sys

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from kda_step import kda_step  # noqa: E402
from mxfp4_qdq import mxfp4_qdq  # noqa: E402
from situ_glu import situ_glu  # noqa: E402
from state_cache import gather_rows, scatter_rows  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
failures = 0


def check(name, got, expected, rtol=1e-4, atol=1e-5):
    global failures
    try:
        torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
        print(f"ok   {name}")
    except AssertionError as error:
        failures += 1
        print(f"FAIL {name}\n{error}")


def mxfp4_reference(x):
    padded = torch.nn.functional.pad(x.flatten(), (0, (-x.numel()) % 32))
    groups = padded.reshape(-1, 32).float()
    max_abs = groups.abs().amax(1, keepdim=True)
    scale = torch.pow(2.0, torch.ceil(torch.log2(torch.clamp(max_abs / 6.0, min=2.0 ** -126))))
    scale = torch.where(max_abs == 0, 1.0, scale)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    indices = (groups.abs().unsqueeze(-1) - levels).abs().argmin(-1)
    q = levels[indices].copysign(groups)
    return (q * scale).flatten()[:x.numel()].to(x.dtype).reshape_as(x)


def main():
    generator = torch.Generator().manual_seed(7)

    x = torch.randn(2 * 777, generator=generator, device=DEVICE)
    gate, up = x.chunk(2)
    expected = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(up / 25))
    check("SiTU-GLU, non-power-of-two length", situ_glu(x), expected, rtol=2e-4, atol=2e-5)

    shape, k_size, v_size = (2, 3), 13, 19
    q = torch.randn(*shape, k_size, generator=generator, device=DEVICE)
    k = torch.nn.functional.normalize(torch.randn(*shape, k_size, generator=generator, device=DEVICE), dim=-1)
    v = torch.randn(*shape, v_size, generator=generator, device=DEVICE)
    alpha = torch.sigmoid(torch.randn(*shape, k_size, generator=generator, device=DEVICE))
    beta = torch.sigmoid(torch.randn(*shape, generator=generator, device=DEVICE))
    state = torch.randn(*shape, k_size, v_size, generator=generator, device=DEVICE)
    expected_state = state * alpha[..., :, None]
    residual = v - torch.einsum("...k,...kv->...v", k, expected_state)
    expected_state = expected_state + beta[..., None, None] * k[..., :, None] * residual[..., None, :]
    expected_out = torch.einsum("...k,...kv->...v", q, expected_state)
    got_state = state.clone()
    got_out = kda_step(q, k, v, alpha, beta, got_state)
    check("KDA recurrent output", got_out, expected_out, rtol=3e-4, atol=3e-5)
    check("KDA in-place FP32 state", got_state, expected_state, rtol=3e-4, atol=3e-5)

    quant_input = torch.cat((torch.zeros(32), torch.randn(69, generator=generator) * 8)).to(DEVICE)
    check("MXFP4-style QDQ", mxfp4_qdq(quant_input), mxfp4_reference(quant_input))

    source = torch.randn(4, 37, generator=generator, device=DEVICE)
    slots = torch.tensor([5, 1, 7, 3], device=DEVICE)
    cache = torch.zeros(9, 37, device=DEVICE)
    scatter_rows(source, slots, cache)
    check("indexed state-cache round trip", gather_rows(cache, slots), source)

    print(f"15-triton-k3: {'FAILED' if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
