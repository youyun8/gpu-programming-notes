"""CPU stand-ins for the GPU-only low-precision references used by Tensara.

Tensara's NVFP4 problems call flashinfer (CUDA only), and the MX / NVFP4 GEMMs
call torch.nn.functional.scaled_mm with swizzled scales (unsupported on CPU).
This module re-implements exactly those pieces in plain PyTorch so the test
runner can check solutions without a GPU:

  * the 128 x 4 "32_4_4" swizzled scale layout (verified against
    torchao.prototype.mx_formats.utils.to_blocked);
  * FP4 (E2M1) packing: element 2i in the low nibble (verified against torchao);
  * flashinfer.fp4_quantization.nvfp4_quantize / e2m1_and_ufp8sf_scale_to_float
    (16-element blocks, FP8 E4M3 block scales, fp32 global scale);
  * scaled_mm for BlockWise1x32 (E8M0) and BlockWise1x16 + TensorWise (NVFP4).

These are faithful re-implementations of the documented semantics, not the
original libraries; bit-level rounding differences are possible in rare ties.
"""
import sys
import types

import torch

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def swizzled_index(r, c, cols):
    """Offset of scale (r, c) of a rows x cols scale matrix in the 128x4 swizzled layout."""
    ncb = (cols + 3) // 4
    rb, cb, ri, ci = r // 128, c // 4, r % 128, c % 4
    return (rb * ncb + cb) * 512 + (ri % 32) * 16 + (ri // 32) * 4 + ci


def swizzle(scales):
    """(rows, cols) -> flat swizzled tensor padded to (roundup(rows,128), roundup(cols,4))."""
    rows, cols = scales.shape
    prow, pcol = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    out = torch.zeros(prow * pcol, dtype=scales.dtype)
    r = torch.arange(rows).view(-1, 1).expand(rows, cols)
    c = torch.arange(cols).view(1, -1).expand(rows, cols)
    out[swizzled_index(r, c, cols).flatten()] = scales.flatten()
    return out


def unswizzle(flat, rows, cols):
    flat = flat.flatten()
    r = torch.arange(rows).view(-1, 1).expand(rows, cols)
    c = torch.arange(cols).view(1, -1).expand(rows, cols)
    return flat[swizzled_index(r, c, cols).flatten()].view(rows, cols)


def unpack_e2m1(q):
    """uint8 (rows, k/2) -> float32 (rows, k); low nibble first."""
    q = q.to(torch.int64)
    codes = torch.stack([q & 0xF, q >> 4], dim=-1).flatten(-2)
    mag = E2M1_VALUES[codes & 7]
    return torch.where((codes & 8) != 0, -mag, mag)


def quantize_e2m1(x):
    """float32 -> e2m1 code (0..15): round to nearest, ties to even code, saturate at 6.

    Midpoints between {0, .5, 1, 1.5, 2, 3, 4, 6} are .25, .75, 1.25, 1.75, 2.5, 3.5, 5;
    '>' vs '>=' at each midpoint sends ties to the even code.
    """
    mag = x.abs()
    code = torch.zeros_like(mag, dtype=torch.int64)
    for threshold, value, inclusive in ((0.25, 1, False), (0.75, 2, True), (1.25, 3, False), (1.75, 4, True),
                                        (2.5, 5, False), (3.5, 6, True), (5.0, 7, False)):
        hit = mag >= threshold if inclusive else mag > threshold
        code = torch.where(hit, torch.full_like(code, value), code)
    return code | ((x < 0).to(torch.int64) << 3)


def pack_e2m1(codes):
    codes = codes.to(torch.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def fp8_e4m3_to_float(b):
    return b.contiguous().view(torch.float8_e4m3fn).to(torch.float32)


def e8m0_to_float(b):
    b = b.to(torch.int64)
    val = torch.pow(2.0, (b - 127).to(torch.float64)).to(torch.float32)
    return torch.where(b == 255, torch.tensor(float("nan")), val)


# ----- flashinfer stand-ins -----------------------------------------------------
def nvfp4_quantize(a, global_sf, sfLayout=None, do_shuffle=False, sf_vec_size=16, **_):
    """a: (m, k) fp16/bf16/fp32 -> (q uint8 (m, k/2), scale uint8 swizzled (pm, pk16))."""
    a = a.float()
    m, k = a.shape
    sf_g = float(global_sf.flatten()[0]) if torch.is_tensor(global_sf) else float(global_sf)
    sf_g32 = torch.tensor(sf_g, dtype=torch.float32)
    blocks = a.view(m, k // sf_vec_size, sf_vec_size)
    vmax = blocks.abs().amax(-1)
    sf_val = sf_g32 * (vmax * torch.tensor(1.0 / 6.0, dtype=torch.float32))
    sf8 = sf_val.to(torch.float8_e4m3fn)
    sf_f = sf8.to(torch.float32)
    out_scale = torch.where(sf_f != 0, 1.0 / (sf_f * (1.0 / sf_g32)), torch.zeros_like(sf_f))
    scaled = blocks * out_scale.unsqueeze(-1)
    q = pack_e2m1(quantize_e2m1(scaled.view(m, k)))
    sf_bytes = sf8.view(torch.uint8)
    pm, pc = (m + 127) // 128 * 128, (k // sf_vec_size + 3) // 4 * 4
    return q, swizzle(sf_bytes).view(pm, pc)


def e2m1_and_ufp8sf_scale_to_float(q, sf, global_scale_inv, sf_vec_size=16, ufp8_type=1, is_sf_swizzled_layout=True):
    m, half_k = q.shape
    k = half_k * 2
    vals = unpack_e2m1(q)
    cols = k // sf_vec_size
    sf_bytes = sf.contiguous().view(torch.uint8).flatten()
    sf_mat = unswizzle(sf_bytes, m, cols) if is_sf_swizzled_layout else sf_bytes.view(m, cols)
    scale = fp8_e4m3_to_float(sf_mat) * float(global_scale_inv.flatten()[0])
    return (vals.view(m, cols, sf_vec_size) * scale.unsqueeze(-1)).view(m, k)


# ----- scaled_mm stand-in ------------------------------------------------------------
def _dequant_operand(q, scales, recipes, swizzles):
    recipes = recipes if isinstance(recipes, (list, tuple)) else [recipes]
    scales = scales if isinstance(scales, (list, tuple)) else [scales]
    raw = q
    transposed = False
    if raw.dim() == 2 and not raw.is_contiguous():  # b.t() view
        raw = raw.t()
        transposed = True
    if raw.dtype == torch.float4_e2m1fn_x2:
        vals = unpack_e2m1(raw.view(torch.uint8))
    else:
        vals = raw.to(torch.float32)
    rows, k = vals.shape
    block = 32 if "1x32" in str(recipes[0]) else 16
    cols = k // block
    s0 = scales[0].contiguous()
    sbytes = s0.view(torch.uint8).flatten()
    smat = unswizzle(sbytes, rows, cols)
    if s0.dtype == torch.float8_e8m0fnu:
        sf = e8m0_to_float(smat)
    else:
        sf = fp8_e4m3_to_float(smat)
    out = (vals.view(rows, cols, block) * sf.unsqueeze(-1)).view(rows, k)
    if len(scales) > 1:  # TensorWise global scale
        out = out * float(scales[1].flatten()[0])
    return out.t() if transposed else out


def scaled_mm(a, b, scale_a=None, scale_recipe_a=None, scale_b=None, scale_recipe_b=None, swizzle_a=None, swizzle_b=None,
              output_dtype=torch.float32, **_):
    a_f = _dequant_operand(a, scale_a, scale_recipe_a, swizzle_a).double()
    b_f = _dequant_operand(b, scale_b, scale_recipe_b, swizzle_b).double()
    return (a_f @ b_f).to(output_dtype)


def install():
    """Register the stand-ins (flashinfer module, F.scaled_mm) in this process."""
    fq = types.ModuleType("flashinfer.fp4_quantization")
    fq.nvfp4_quantize = nvfp4_quantize
    fq.e2m1_and_ufp8sf_scale_to_float = e2m1_and_ufp8sf_scale_to_float
    fi = types.ModuleType("flashinfer")
    fi.fp4_quantization = fq
    sys.modules["flashinfer"] = fi
    sys.modules["flashinfer.fp4_quantization"] = fq
    import torch.nn.functional as F
    F.scaled_mm = scaled_mm
