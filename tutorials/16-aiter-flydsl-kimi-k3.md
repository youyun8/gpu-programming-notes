# 16 – AITER and FlyDSL for Kimi K3 on AMD GPUs

> **Part V · AMD Production Kernels** · Prerequisites:
> [05 – CDNA3 and MFMA](05-amd-cdna3-mfma.md),
> [06 – AITER Assembly GEMM](06-aiter-asm-gemm.md), and
> [15 – Triton in Model Systems](15-triton-model-systems.md) ·
> Next: [08 – Deploying This Site](08-deploying-this-site.md)

This chapter connects the AMD kernel stack to a complete model. Kimi K3 is a
useful case because it combines low-precision weights, sparse experts, Kimi
Delta Attention (KDA), gated MLA, and a new activation.

The tools have separate roles:

```text
Quark checkpoint conversion
    → SGLang, vLLM, or ATOM runtime
        → AITER operator dispatch
            → FlyDSL, Triton/Gluon, CK, HIP, assembly, or Opus kernel
```

Quark changes model data. AITER chooses and packages operators. FlyDSL writes
and compiles kernels. None of the three is a complete serving runtime.

**You will learn**

- the FlyDSL layout and launch model, from vector addition to MFMA;
- how AITER dispatches tuned kernels;
- every K3-related AITER/FlyDSL operator family that is public today;
- the full MoE data path, including MXFP4/MXFP8 and SiTUv2;
- where K3 still uses Triton instead of FlyDSL;
- how to test one kernel and how to reproduce the eight-GPU K3 recipe.

## 1. Reproducible Environment

This chapter uses:

- [AITER v0.1.23](https://github.com/ROCm/aiter/tree/v0.1.23);
- [FlyDSL v0.3.4.1](https://github.com/ROCm/FlyDSL/tree/v0.3.4.1);
- [Quark release/0.12](https://github.com/amd/Quark/tree/release/0.12).

AITER v0.1.23 pins FlyDSL 0.3.4.1. The online FlyDSL documentation may describe
a newer API, so use tag-pinned examples when reproducing this chapter.

Check the machine first:

```bash
rocminfo | rg 'gfx'
python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__, "ROCm:", torch.version.hip)
print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name())
PY
```

ROCm intentionally uses PyTorch's `torch.cuda` namespace. Do not replace it
with `torch.rocm`.

| Target | Status to assume |
|---|---|
| `gfx942` (MI300X/MI325X) | broad AITER support; useful for individual kernels |
| `gfx950` (MI350X/MI355X) | target of the published full K3 recipe |
| RDNA targets | experimental for many AITER paths |

Support for one kernel does not prove that the full 2.8-trillion-parameter
model fits or runs on that target.

## 2. FlyDSL from the Smallest Kernel

FlyDSL is a Python DSL built around explicit tensors, layouts, copy atoms, and
MMA atoms. A kernel has two levels:

```python
@flyc.kernel
def vector_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                      tiled_copy: fx.TiledCopy):
    # Device program: partition tiles among threads and copy fragments.
    ...

@flyc.jit
def vector_add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
    # Host program: construct layouts and launch the device kernel.
    ...
```

Run the official first example:

```bash
git clone --branch v0.3.4.1 https://github.com/ROCm/FlyDSL.git
cd FlyDSL
pip install flydsl==0.3.4.1 pytest pandas
python3 examples/01-vectorAdd.py
python3 -m pytest tests/kernels/test_vec_add.py
```

The example builds a `UniversalCopy128b` copy atom, maps a \(128\)-thread
layout over the output tile, partitions source and destination tensors, and
predicates border copies. These are the FlyDSL equivalents of `float4`,
thread-to-element mapping, and bounds checks from chapters 01 and 02.

### 2.1 The Learning Ladder

| Step | Tagged example | New idea |
|---|---|---|
| 1 | `examples/01-vectorAdd.py` | tensors, layouts, predicated 128-bit copies |
| 2 | `examples/02-tiledCopy.py` | global/shared-memory tiling and partitioning |
| 3 | `examples/03-tiledMma.py` | register fragments and tiled MFMA |
| 4 | `examples/04-preshuffle_gemm.py` | production weight layout and GEMM |

The key question is always: **which logical values does this thread own at
this point?** A layout answers that question. A copy atom or MMA atom says
what instruction moves or consumes those values.

## 3. AITER Is a Dispatcher, Not One Kernel Language

AITER exposes PyTorch-callable operators. On the first call it may JIT-compile
a backend, then cache the result. Dispatch considers:

- GPU architecture;
- data types and quantization format;
- matrix or attention shape;
- alignment and layout;
- environment overrides;
- entries in tuned CSV configuration files.

The selected implementation may be FlyDSL, Triton/Gluon, Composable Kernel,
HIP, assembly, or another generator. Therefore, “using AITER” does not imply
“using FlyDSL.”

K3-specific tuning files under
[`aiter/configs/model_configs`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/configs/model_configs)
cover:

- BF16 and A4W4 block-scaled GEMM;
- A8W8 preshuffled GEMM;
- A16W4, A8W4, A4W4, and I4 fused MoE;
- FP8 FMHA.

Tuning is shape- and architecture-specific. An unknown shape may use a generic
configuration or another backend.

## 4. Quantization and Data Layout

K3's native QAT representation uses MXFP4 weights and MXFP8 activations.
Another AMD checkpoint combines MXFP4 with per-token-per-channel FP8 in
attention. The names describe arithmetic *and* storage.

| Format | Main idea | Kernel work |
|---|---|---|
| FP8 | one eight-bit floating value per element | amax, scale, convert |
| INT8/INT4 | integer value plus scale | reduce, round, clamp, pack |
| MXFP8 | FP8 values sharing a scale over a small block | block reduction and scale layout |
| MXFP4 | E2M1 values sharing an E8M0 scale, usually 32 values per block | block reduction, nibble packing, scale swizzle |

Preshuffling reorganizes weights before inference so each MFMA wave can load
the fragments it needs with contiguous or conflict-free accesses. It is not a
mathematical transpose that can be added or removed casually. The converter,
GEMM, and scale layout must agree.

Start with isolated tests:

```bash
cd aiter
python3 op_tests/test_quant_mxfp4.py
python3 op_tests/test_gemm_a8w8.py
```

Compare dequantized output with an FP32 reference before measuring speed.
Include all-zero blocks, maximum finite values, partial groups, and values at
rounding midpoints.

## 5. The K3 MoE Pipeline

K3 has 896 routed experts, selects 16 per token, and also has shared experts.
The production path is a pipeline:

```text
router logits
  → grouped top-k
  → route + quantize + scatter
  → expert GEMM 1 (gate/up)
  → SiTUv2 + activation quantization
  → expert GEMM 2 (down)
  → weighted reduce/combine
```

### 5.1 Routing

Routing selects experts and produces weights. The scatter stage then groups
token rows by expert and pads/alines work for the GEMM. AITER's public code
includes strided grouped top-k for K3 and
[`moe_fused_route_quant_scatter.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py)
for fused movement and quantization.

Routing is small in FLOPs but sensitive to:

- stable token/expert ordering;
- expert capacity and empty experts;
- duplicate experts for one token;
- quantization scale ownership;
- expert-parallel destination rank.

### 5.2 Expert GEMM 1

The first expert GEMM produces gate and up vectors. Public FlyDSL paths include
[`mxfp4_gemm1.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm1.py)
and a
[`two-stage mixed MoE GEMM`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py).
The tile must balance MFMA use, expert size, and the often-small number of
tokens assigned to one expert.

### 5.3 SiTUv2 and Requantization

The activation is the formula from chapter 15:

$$
y =
\left[4\tanh(g/4)\sigma(g)\right]
\left[25\tanh(u/25)\right].
$$

AITER's `situv2_and_mul_quant` fuses that expression with row-scale
calculation and FP8 conversion:

```python
from aiter.ops.activation import situv2_and_mul_quant

out = torch.empty((tokens, width), device="cuda", dtype=aiter.dtypes.fp8)
scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)
situv2_and_mul_quant(out, x, scale, width, 4.0, 25.0)
```

Validate the quantized result by comparing `out.float() * scale` with an FP32
reference. The official tagged test also checks zero rows and an empty batch.

```bash
python3 op_tests/test_situv2_and_mul_quant.py -d bf16 -m 1 -n 768
```

### 5.4 Expert GEMM 2 and Combine

[`mxfp4_gemm2.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm2.py)
projects expert activations back to model width. The combine stage multiplies
each result by its routing weight and reduces the selected experts for each
token. With expert parallelism, communication may sit between dispatch and
combine.

Test each boundary separately: routed row order, quantized activation and
scale, GEMM output, then final weighted sum.

## 6. Attention and State-Space Operators

### 6.1 Gated MLA

AITER and FlyDSL provide building blocks for K3's gated MLA layers:

- FMHA and paged attention;
- QK normalization, RoPE, and quantization;
- KV gather and B-projection;
- MLA split reduction;
- FP8 attention configurations.

Relevant entry points include
[`fmha_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/fmha_kernels.py),
[`qk_norm_rope_quant.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/qk_norm_rope_quant.py),
and
[`mla_reduce_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/mla_reduce_kernels.py).

### 6.2 KDA

KDA is the important boundary: AITER has K3 KDA implementations under its
Triton/Gluon tree, including `kimi_delta_attn`, `gated_delta_net`, and chunk
delta attention. No equivalent public FlyDSL KDA path is present in the
versions used here.

The published AMD SGLang recipe enables AITER/FlyDSL but still selects
`--attention-backend triton`. It is accurate to say that FlyDSL accelerates
the K3 MoE/GEMM/support path; it is not accurate to say that every K3
attention kernel is FlyDSL.

## 7. Communication and Multi-GPU Execution

The full model needs tensor and/or expert parallelism. AITER/FlyDSL families
include custom all-reduce, reduce-scatter/all-gather, dispatch/combine, and
MegaMoE communication.

Communication correctness depends on more than one kernel:

- every rank must use the same token/expert metadata;
- send and receive counts must match, including empty experts;
- reduction dtype and order affect error;
- streams and events must protect buffers;
- graph capture needs stable addresses and launch topology.

Run single-GPU operator tests first. FlyDSL keeps multi-GPU all-reduce tests
separate in `tests/kernels/test_allreduce.py`.

## 8. Test Ladder

### 8.1 FlyDSL smoke tests

```bash
python3 examples/01-vectorAdd.py
python3 -m pytest tests/kernels/test_vec_add.py
python3 -m pytest tests/kernels/test_preshuffle_gemm.py -m "not large_shape"
python3 -m pytest tests/kernels/test_quant.py
python3 -m pytest tests/kernels/test_moe_gemm.py
python3 -m pytest tests/kernels/test_flash_attn_fwd.py
```

FlyDSL separates tests that need no device, portable compilation, ROCDL
lowering, and real GPU execution. Read the marker before assuming a passing
compile test proves numerical correctness.

### 8.2 AITER operator tests

Install the v0.1.23 wheel that matches ROCm and Python, or build the tagged
repository recursively:

```bash
git clone --recursive --branch v0.1.23 https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
python3 op_tests/test_rmsnorm2d.py
python3 op_tests/test_gemm_a8w8.py
python3 op_tests/test_mla.py
python3 op_tests/test_moeTopkSoftmax.py
python3 op_tests/test_moe_2stage.py
```

For every benchmark table, keep a correctness column. JIT compilation must be
excluded from steady-state timing.

### 8.3 Full K3 recipe

The published full-model recipe requires eight MI350X or MI355X GPUs and about
1.56 TB of checkpoint storage:

```bash
madengine run --tags pyt_sglang_kimi-k3 --keep-model-dir --live-output
```

Its key environment choices include:

```bash
SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1
AITER_FLYDSL_FORCE=1
AITER_SITUV2_A8W4=1
```

Use the
[ROCm MAD K3 recipe](https://github.com/ROCm/MAD/blob/develop/benchmark/kimi_k3/README.md)
as the source of truth for complete launch flags. A successful small-kernel
test is not a substitute for this end-to-end validation.

## 9. Quark-to-Runtime Workflow

The AMD K3 checkpoint recipe uses direct conversion so the full model does not
need to be loaded conventionally:

```python
template = LLMTemplate.get("kimi_k3")
quant_config = template.get_config(
    scheme="mxfp4",
    layer_config={"*self_attn*": "ptpc_fp8"},
)
ModelQuantizer(quant_config).direct_quantize_checkpoint(
    pretrained_model_path="moonshotai/Kimi-K3",
    save_path=output_dir,
)
```

This is a data-format boundary. Record the Quark version, template, layer
overrides, runtime image, and checkpoint identifier together. A runtime flag
that emulates a format is usually a version-specific reproduction workaround,
not a general requirement.

## 10. Debugging and Tuning

Use this order:

1. **Reference:** compare with an FP32 PyTorch implementation.
2. **Edges:** test zero tokens, one token, odd sizes, partial groups, and empty
   experts.
3. **Dispatch:** log architecture, selected backend, and tuning row.
4. **Compiler:** inspect FlyDSL/MLIR/ROCDL output and resource use.
5. **Profiler:** measure waves, MFMA use, LDS conflicts, cache misses, and
   launch gaps with `rocprofv3`.
6. **End to end:** measure scheduler, communication, and model output quality.

If `AITER_FLYDSL_FORCE=1` makes a shape fail, first check whether that shape has
a supported FlyDSL kernel. Forcing a backend removes fallback; it does not add
coverage.

## Key Takeaways

1. FlyDSL makes layouts, copies, and MFMA ownership explicit in Python.
2. AITER dispatches several backend families; inspect the chosen one.
3. K3's main FlyDSL path is quantized MoE: route, scatter, two GEMMs,
   SiTUv2/requantize, and combine.
4. Gated MLA has FlyDSL building blocks, while public KDA paths are currently
   Triton/Gluon.
5. Exact quantized layouts and tuning rows are part of correctness.
6. Full K3 validation needs the supported eight-GPU recipe; individual kernels
   can be learned and checked on smaller supported AMD systems.

## Exercises

1. Trace the thread and value layouts in FlyDSL's vector-add example.
2. Run preshuffled GEMM with an odd \(M\) and identify which predicates protect
   the edge tile.
3. Compare AITER's SiTUv2 FP8 output with the chapter 15 Triton FP32 output.
4. Log AITER dispatch for one K3 MoE shape with and without
   `AITER_FLYDSL_FORCE`.
5. Draw the buffers and scales between every stage of the two-stage MoE test.
