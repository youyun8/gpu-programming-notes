# 16 – FlyDSL in AITER – Kimi K3 on AMD GPUs

> **Part VI · Kimi K3 Case Studies** · Prerequisites:
> [Matrix Multiplication 1 – Foundations](04-tiled-matmul.md) and
> [Matrix Multiplication 8 – Tensor Cores](gemm/07-tensor-cores.md),
> [Triton – From First Kernel to Production](14-triton.md), and
> [CDNA3 and MFMA](05-amd-cdna3-mfma.md) ·
> Previous: [15 – Triton in SGLang: Serving Kimi K3](15-triton-model-systems.md) ·
> Next: [08 – Deploying This Site](08-deploying-this-site.md)

This chapter follows one production path from a Kimi K3 checkpoint to kernels
on AMD GPUs. It focuses on FlyDSL kernels reached through AITER, but it keeps
the software boundaries explicit:

```text
Quark converts checkpoint data
    → SGLang, vLLM, or ATOM schedules a request
        → AITER exposes and dispatches operators
            → FlyDSL, Triton/Gluon, CK, HIP, assembly, or Opus runs a kernel
```

Quark is not a serving runtime. AITER is not one kernel language. FlyDSL does
not implement every AITER operator. In particular, the public K3 recipes
covered here keep Kimi Delta Attention (KDA) on Triton/Gluon paths.

**You will learn**

- the roles of Kimi K3, Quark, the serving runtime, AITER, and FlyDSL;
- the FlyDSL tensor, layout, copy, and MMA model;
- how AITER combines dispatch, JIT and AOT packaging, and tuned tables;
- how quantization and preshuffled storage become kernel contracts;
- the complete K3 MoE path from routing to the weighted combine;
- which public K3-facing AITER families use FlyDSL and which do not;
- how MLA, expert parallelism, and communication fit around the MoE path;
- how to test one operator before running the eight-GPU MAD recipe.

## 1. Pin the Public Stack

The examples and source links in this chapter use:

- [AITER v0.1.23](https://github.com/ROCm/aiter/tree/v0.1.23);
- [FlyDSL v0.3.4.1](https://github.com/ROCm/FlyDSL/tree/v0.3.4.1);
- [Quark release/0.12](https://github.com/amd/Quark/tree/release/0.12).

AITER v0.1.23 pins FlyDSL 0.3.4.1. FlyDSL changes quickly, so use the tagged
examples instead of copying an API from current online documentation.

Check the machine before building:

```bash
rocminfo | rg 'gfx'
python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__, "ROCm:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
PY
```

ROCm deliberately uses PyTorch's `torch.cuda` namespace. Do not change it to
`torch.rocm`.

| Target | Safe expectation for this chapter |
|---|---|
| `gfx942` (MI300X/MI325X) | Many AITER operators and useful local kernel tests |
| `gfx950` (MI350X/MI355X) | Target of the published eight-GPU K3 recipe |
| RDNA targets | Experimental or unsupported for many paths discussed here |

Support for an individual operator does not mean the full
2.8-trillion-parameter model fits or that the published recipe supports the
same GPU.

## 2. Give Each Layer One Job

| Layer | Job | What it does not promise |
|---|---|---|
| Kimi K3 model | Defines the architecture, weights, 896 routed experts, top-16 routing, KDA, and gated MLA | A particular AMD kernel backend |
| Quark | Quantizes and converts checkpoint tensors | Request scheduling or runtime dispatch |
| SGLang, vLLM, or ATOM | Owns batches, caches, graph capture, parallel ranks, and backend selection | That every selected AITER operator is FlyDSL |
| AITER | Presents PyTorch-callable AMD operators and chooses implementations | One implementation language for all operators |
| FlyDSL | Describes layouts, copies, MMA work, and launches in Python | A complete model server |

This separation is also a debugging tool. A wrong scale in a converted
checkpoint is not fixed by changing an MFMA tile. A correct local GEMM does
not prove that expert-parallel metadata is exchanged correctly.

## 3. FlyDSL: From Values to MFMA

FlyDSL is a Python DSL with an explicit device program and host-side launch
program:

```python
@flyc.kernel
def vector_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    tiled_copy: fx.TiledCopy,
):
    # Device program: partition tensors and move each thread's values.
    ...

@flyc.jit
def vector_add(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    # Host program: build layouts, specialize, and launch the kernel.
    ...
```

The exact API evolves, but the reasoning model is stable.

### 3.1 Tensors and layouts

An `fx.Tensor` is a view of storage. Its layout maps a logical coordinate to a
physical offset. A layout can describe a global-memory tile, an LDS tile, or a
register fragment. It can also split and compose dimensions.

At every stage, ask:

> Which logical values does this thread or wave own, and where are they
> stored?

That question replaces hand-written index arithmetic. A layout is part of
correctness, not only a performance hint.

### 3.2 Copy atoms and tiled copies

A copy atom describes one legal movement, such as a vectorized 128-bit copy.
A tiled copy repeats that atom over a larger logical tile and assigns pieces
to threads. The source and destination partitions must agree on value
ownership.

Copies still need predicates at an edge tile. A wide vector operation is safe
only when alignment, valid lanes, and the storage layout all satisfy its
contract.

### 3.3 MMA atoms and tiled MMA

An MMA atom represents one hardware matrix instruction. On CDNA this usually
means an MFMA operation. A tiled MMA expands the atom over a larger output
tile and defines:

- which lanes hold A and B fragments;
- which accumulator values each lane owns;
- how the wave repeats work across M, N, and K;
- how register fragments connect to global or LDS copies.

The kernel normally pipelines copies with repeated MMA steps, then applies an
epilogue and stores the result. This is the same matrix path developed in
[Tiled Matrix Multiplication](04-tiled-matmul.md), now expressed through
layout algebra.

### 3.4 Learn in four tagged steps

```bash
git clone --branch v0.3.4.1 https://github.com/ROCm/FlyDSL.git
cd FlyDSL
pip install flydsl==0.3.4.1 pytest pandas
python3 examples/01-vectorAdd.py
```

| Step | Tagged example | New idea |
|---|---|---|
| 1 | [`examples/01-vectorAdd.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/01-vectorAdd.py) | tensors, layouts, predicates, and 128-bit copies |
| 2 | [`examples/02-tiledCopy.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/02-tiledCopy.py) | global/LDS tiles and thread partitioning |
| 3 | [`examples/03-tiledMma.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/03-tiledMma.py) | register fragments and tiled MFMA |
| 4 | [`examples/04-preshuffle_gemm.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/04-preshuffle_gemm.py) | a production weight layout and GEMM |

Do not start with the full MoE kernel. First trace one tile through these four
examples: global memory → LDS or registers → MMA fragments → output.

## 4. AITER Dispatch, Compilation, and Tuning

AITER gives a runtime a stable operator interface. The selected implementation
can be FlyDSL, Triton/Gluon, Composable Kernel, HIP, assembly, Opus, or another
generator.

Dispatch can depend on:

- GPU architecture;
- input, weight, accumulator, and output types;
- quantization and scale format;
- M, N, K, token count, page size, and attention mode;
- strides, alignment, preshuffle state, and other layout facts;
- environment overrides;
- a matching tuned configuration.

Therefore:

> “The runtime uses AITER” does not mean “the runtime uses FlyDSL.”

### 4.1 JIT

A FlyDSL wrapper can specialize and compile a kernel on its first call. AITER
then caches the result. The cold call includes Python dispatch, code
generation, compilation, and loading. Warm calls should reuse the compiled
specialization.

Always warm up before timing. Record cache state when comparing runs.

### 4.2 AOT

AITER also has AOT manifests, including public FlyDSL manifests for MoE
families. These describe kernels that can be built and packaged before model
execution. AOT reduces startup work, but only for exported specializations.
An uncovered shape can still need JIT compilation or a different backend.

See the tagged
[`aiter/aot/flydsl`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/aot/flydsl)
directory. A manifest is not proof that the runtime chose that kernel.

### 4.3 Tuned and untuned tables

K3-specific files in
[`aiter/configs/model_configs`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/configs/model_configs)
cover these public format families:

- BF16 GEMM;
- A8W8 B-preshuffled GEMM;
- A4W4 block-scaled GEMM;
- A16W4, A8W4, A4W4, and I4 fused MoE;
- FP8 FMHA AOT configurations.

The directory has tuned and untuned variants. A tuned row is a measured choice
for a shape and architecture. It is not a general statement that one tile is
best everywhere.

`AITER_FLYDSL_FORCE=1` removes some fallback choices. It does not create a
FlyDSL implementation for an unsupported operator, shape, or architecture.

## 5. Public K3-Facing Operator Catalog

The table below catalogs the K3-facing FlyDSL/AITER use cases visible in the
pinned public source. “Available” means that a public operator or tuning path
exists. It does not mean every serving framework selects it for every K3
request.

| K3 use case | Public AITER/FlyDSL family | Boundary |
|---|---|---|
| Dense projections | BF16 GEMM, A8W8 B-preshuffled GEMM, and A4W4 block-scaled GEMM tuning | AITER may select FlyDSL, assembly, or another backend |
| Top-k and expert metadata | top-k, MoE sorting, route maps, and group/local lookup helpers | Routing policy belongs to the model/runtime |
| Quantize and scatter routed rows | [`moe_fused_route_quant_scatter.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py), scatter-copy, scale-preshuffle helpers | Row order, destination, data, and scales form one contract |
| Expert GEMM 1 | [`mxfp4_gemm1.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm1.py) and MoE stage-1 wrappers | Produces gate/up data; supported formats and shapes are dispatch-specific |
| Mixed two-stage MoE | [`mixed_moe_gemm_2stage.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py) | Covers public mixed-format paths; it is not every fused-MoE path |
| SiTUv2 plus requantization | AITER `situv2_and_mul_quant` | Public AITER operator; do not label it FlyDSL without checking the selected build |
| Expert GEMM 2 | [`mxfp4_gemm2.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm2.py) and MoE stage-2 wrappers | Consumes the activation and its scale |
| Expert output combine | FlyDSL gather/reduce, MoE reduce, dispatch/combine, and MegaMoE helpers | Local combine and inter-rank combine are different cases |
| MLA preparation | [`qk_norm_rope_quant.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/qk_norm_rope_quant.py), KV gather/B projection | Preparation does not imply the attention core uses FlyDSL |
| MLA attention and merge | [`fmha_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/fmha_kernels.py), paged/FP8 MLA paths, and [`mla_reduce_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/mla_reduce_kernels.py) | AITER also has assembly, Triton, and Opus MLA paths |
| Expert-parallel movement | intra-node dispatch/combine, communication-fused MoE, MegaMoE, and quick all-reduce | Requires matching rank metadata, IPC setup, and topology |
| KDA | AITER Triton/Gluon KDA families | No public K3 FlyDSL KDA path is claimed here |

Other generic FlyDSL kernels in AITER—convolution, HSTU, unrelated GEMMs, and
other model-specific operators—are outside this K3 catalog. Conversely, this
catalog does not claim that all of K3 runs in FlyDSL.

## 6. Quantization and Preshuffling

K3's native QAT representation uses MXFP4 weights and MXFP8 activations.
Another AMD checkpoint path combines MXFP4 with per-token-per-channel FP8 in
attention. A format name describes arithmetic and storage together.

| Format | Stored values | Scale work |
|---|---|---|
| FP8 | One eight-bit floating value per element | Find an amax, derive a scale, convert |
| INT8/INT4 | Signed integer plus a scale | Reduce, round, clamp, and sometimes pack |
| MXFP8 | FP8 values sharing a scale over a small block | Reduce each block and arrange block scales |
| MXFP4 | Packed E2M1 values with a shared E8M0 block scale | Reduce, encode, nibble-pack, and swizzle scales |

The precise block size, scale orientation, and byte order come from the
operator contract. Do not infer them from “4-bit” alone.

### 6.1 Why preshuffle exists

A mathematical weight matrix is indexed as \(W[k,n]\), but one MFMA wave does
not consume it in simple row-major order. Preshuffling stores the weights and,
when required, scales in the order used by the copy and MMA layouts.

This can:

- make wave loads contiguous;
- avoid repeated runtime permutations;
- reduce LDS bank conflicts;
- put packed nibbles and scales next to the fragments that use them.

Preshuffling is not a free transpose. The checkpoint converter or preparation
step, selected GEMM, scale layout, and architecture must agree. A kernel can
run and still produce plausible but wrong values when one of these contracts
does not match.

### 6.2 Validate the format before the GEMM

Start with:

```bash
cd aiter
python3 op_tests/test_quant_mxfp4.py
python3 op_tests/test_gemm_a8w8.py
```

Test all-zero blocks, largest finite values, rounding midpoints, partial
groups, odd row counts, and empty input. Compare dequantized values with an
FP32 reference before measuring GEMM speed.

## 7. Complete K3 MoE Path

K3 has 896 routed experts, selects 16 experts per token, and also uses shared
experts. The routed path is:

```text
router logits
  → grouped top-k and routing weights
  → route + quantize + scatter
  → expert GEMM 1 (gate/up)
  → SiTUv2 + activation requantization
  → expert GEMM 2 (down)
  → weighted reduce/combine
```

Each arrow is a testable boundary.

### 7.1 Route

The router produces selected expert IDs and weights. Runtime policy may also
renormalize weights and map global experts to local ranks. AITER's routing and
sorting helpers turn this sparse choice into metadata that grouped GEMMs can
consume.

Check:

- stable token/expert order;
- the meaning of global and local expert IDs;
- duplicate selections, if the caller can produce them;
- empty experts and zero-token batches;
- capacity, padding, and alignment;
- routing-weight dtype and normalization.

Routing is light in FLOPs but can corrupt the complete layer when one index is
wrong.

### 7.2 Quantize and scatter

The fused route/quant/scatter kernel reads a token row, computes the required
activation scale, converts values, and writes the row into expert-grouped
storage. In expert parallelism, it also prepares data and metadata for another
rank.

Validate four outputs together:

1. destination expert or rank;
2. destination row;
3. quantized data;
4. the scale used to recover that row.

Testing only the quantized bytes misses routing errors. Testing only the row
map misses a scale attached to the wrong token.

### 7.3 Expert GEMM 1

GEMM 1 applies each selected expert's gate and up projections. The effective
M for one expert can be small and uneven, even when the global batch is large.
The kernel must balance MFMA efficiency with padding and launch overhead.

The public MXFP4 and mixed two-stage kernels provide FlyDSL paths for supported
format and shape combinations. Dispatch still decides whether a given call
uses them.

At this boundary, compare output in the scattered row order. Do not combine
rows back to token order yet; that can hide an ordering bug.

### 7.4 SiTUv2 and activation requantization

For gate input \(g\) and up input \(u\), K3 uses:

$$
y =
\left[\beta_1\tanh(g/\beta_1)\sigma(g)\right]
\left[\beta_2\tanh(u/\beta_2)\right],
\qquad \beta_1=4,\quad \beta_2=25.
$$

| Symbol | Meaning |
|---|---|
| \(g,u\) | Gate and up halves from GEMM 1 |
| \(\sigma\) | Sigmoid |
| \(\beta_1,\beta_2\) | K3 bounds |
| \(y\) | Fused expert activation |

AITER's `situv2_and_mul_quant` fuses this expression with a row-scale
calculation and FP8 conversion:

```python
from aiter.ops.activation import situv2_and_mul_quant

out = torch.empty((tokens, width), device="cuda", dtype=aiter.dtypes.fp8)
scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)
situv2_and_mul_quant(out, x, scale, width, 4.0, 25.0)
```

Compare `out.float() * scale` with an FP32 implementation of the formula.
Keep zero rows and empty batches in the test:

```bash
python3 op_tests/test_situv2_and_mul_quant.py -d bf16 -m 1 -n 768
```

This is an AITER operator boundary. The public API alone does not prove that
its implementation is FlyDSL.

### 7.5 Expert GEMM 2

GEMM 2 consumes the quantized activation and its row scale, then applies each
expert's down projection. It must read rows in exactly the order produced by
scatter and GEMM 1.

Compare its output before applying routing weights. A mismatch here should not
be debugged through the final combined tensor.

### 7.6 Weighted combine

The combine stage restores token order, multiplies each selected expert output
by its routing weight, and reduces the top-16 contributions. It can also add
the shared-expert result at the model-defined point.

With expert parallelism, combine includes a communication problem:

```text
local expert outputs
  → send or reduce across ranks
  → gather contributions for each original token
  → multiply by routing weights
  → sum in token order
```

Check one token and one expert contribution first. Then check all 16. Finally
compare the complete local or distributed combine with a high-precision
reference.

## 8. MLA Support and the KDA Boundary

K3 mixes gated Multi-head Latent Attention (MLA) layers with KDA layers. They
do not have to use the same backend.

### 8.1 Public FlyDSL/AITER MLA building blocks

The pinned AITER source includes public paths for:

- Q/K normalization, RoPE, and quantization;
- KV gather and B projection;
- FP8 FMHA;
- paged MLA work for supported page layouts;
- split attention output and log-sum-exp reduction.

These are real FlyDSL/AITER use cases, but coverage is constrained by
architecture, dtype, head shape, page size, and decode or prefill mode. AITER
also ships non-FlyDSL MLA implementations. Inspect dispatch rather than
crediting all MLA time to FlyDSL.

### 8.2 KDA remains Triton/Gluon

KDA is the hard backend boundary for this chapter. AITER's public K3 KDA work
lives in its tagged
[`kimi_delta_attn`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/ops/triton/kimi_delta_attn),
[`gated_delta_net`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/ops/triton/gated_delta_net),
and
[`chunk_delta_attn`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/ops/triton/_triton_kernels/chunk_delta_attn)
Triton/Gluon families. The tree also contains generic FlyDSL gated-delta
helper code; that helper is not evidence of a complete K3 FlyDSL KDA backend.
This chapter does not claim one.

The published SGLang K3 command enables AITER and forces supported FlyDSL
paths, yet it still passes:

```bash
--attention-backend triton
```

That is not a contradiction. One model request can use FlyDSL for MoE and
support operators while Triton/Gluon handles KDA and the selected attention
backend.

## 9. Communication and Expert Parallelism

Tensor parallelism splits dense work. Expert parallelism places experts on
different ranks. K3 needs consistent routing metadata and data movement around
the grouped GEMMs.

Public FlyDSL/AITER communication families relevant to this design include:

- intra-node dispatch/combine;
- communication-fused MoE;
- MegaMoE dispatch, GEMM, and combine stages;
- reduce-scatter/all-gather style support;
- quick all-reduce paths for supported formats.

Availability in source does not show that a particular K3 recipe selected a
family. Runtime flags, world size, topology, data type, and shape still decide.

For correctness:

- all ranks must agree on token and expert metadata;
- send and receive counts must include empty experts correctly;
- local and global expert IDs must not be mixed;
- streams and events must protect reusable buffers;
- IPC handles and peer access must be valid;
- graph capture requires stable addresses and launch topology;
- reduction order and dtype can change numerical error.

Start on one GPU. Then use two ranks with a hand-written route map. Move to the
full rank count only after empty and imbalanced expert cases pass.

## 10. Quark Ends at the Checkpoint Boundary

The AMD K3 conversion path can quantize a checkpoint directly, without first
loading the complete model in the conventional way:

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

Quark writes data and metadata. It does not choose the runtime kernel. AITER
does not reinterpret a checkpoint whose packing or scale layout is wrong.

Record these items as one reproducibility unit:

- source checkpoint identifier and revision;
- Quark release and K3 template;
- quantization scheme and layer overrides;
- output checkpoint identifier;
- serving image and AITER/FlyDSL versions;
- target GPU architecture.

A runtime compatibility flag that emulates an older layout may be useful for
one release. It should not be presented as a permanent property of MXFP4.

## 11. Local Operator Test Ladder

Do not begin with eight GPUs. Climb this ladder and keep a correctness result
beside every timing result.

### 11.1 FlyDSL language smoke tests

```bash
cd FlyDSL
python3 examples/01-vectorAdd.py
python3 -m pytest tests/kernels/test_vec_add.py
python3 -m pytest tests/kernels/test_preshuffle_gemm.py -m "not large_shape"
python3 -m pytest tests/kernels/test_quant.py
python3 -m pytest tests/kernels/test_moe_gemm.py
python3 -m pytest tests/kernels/test_flash_attn_fwd.py
```

A compile-only or lowering test does not prove GPU numerical correctness.
Read the test marker and target before interpreting a pass.

### 11.2 AITER operator tests

Install a wheel that matches ROCm and Python, or build the pinned source:

```bash
git clone --recursive --branch v0.1.23 https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

Then test one boundary at a time:

```bash
python3 op_tests/test_quant_mxfp4.py
python3 op_tests/test_gemm_a8w8.py
python3 op_tests/test_situv2_and_mul_quant.py -d bf16 -m 1 -n 768
python3 op_tests/test_moeTopkSoftmax.py
python3 op_tests/test_moe_2stage.py
python3 op_tests/test_mla.py
python3 op_tests/test_rmsnorm2d.py
```

Use the test's supported arguments to add:

1. zero tokens;
2. one token;
3. odd and edge dimensions;
4. empty and heavily imbalanced experts;
5. an exact K3 production shape;
6. cold and warm launch measurements.

### 11.3 Multi-GPU tests

Run communication tests only after local MoE passes. FlyDSL keeps all-reduce
coverage in `tests/kernels/test_allreduce.py`; AITER has separate multi-GPU
operator tests. Confirm that the test's expected world size and topology match
the machine before launching it.

## 12. Eight-GPU MAD Recipe

The public
[ROCm MAD K3 recipe](https://github.com/ROCm/MAD/blob/develop/benchmark/kimi_k3/README.md)
is the source of truth for full-model commands and images. It requires:

- eight MI350X or MI355X GPUs (`gfx950`);
- tensor parallel size 8 in the published launches;
- about 1.56 TB for the checkpoint.

MAD publishes K3 runs for three frameworks:

```bash
# vLLM
madengine run --tags pyt_vllm_kimi-k3 --keep-model-dir --live-output

# SGLang
madengine run --tags pyt_sglang_kimi-k3 --keep-model-dir --live-output

# ATOM
madengine run --tags pyt_atom_kimi-k3 --keep-model-dir --live-output
```

For the FlyDSL-focused SGLang path, the published container uses:

```bash
SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1
AITER_FLYDSL_FORCE=1
AITER_SITUV2_A8W4=1
```

and starts the server with the important backend boundary visible:

```bash
sglang serve --model-path /model_weights \
  --trust-remote-code --tp-size 8 \
  --attention-backend triton --dtype bfloat16 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 256 \
  --host 127.0.0.1 --port 30000 \
  --disable-radix-cache \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```

Use the image and complete command from MAD rather than combining flags from
different releases. A successful local FlyDSL test is necessary evidence, but
it is not a substitute for loading the checkpoint, serving requests, and
checking model output.

## 13. Profiling and Debugging

Use this order:

1. **Reference.** Compare one boundary with an FP32 PyTorch implementation.
2. **Format.** Verify packed values, scale shape, scale ownership, and
   preshuffle version.
3. **Metadata.** Check token rows, expert IDs, rank IDs, and empty experts.
4. **Dispatch.** Log architecture, operator, selected backend, and tuning row.
5. **Compilation.** Separate cold JIT time from warm execution and retain the
   compiler error for the failing specialization.
6. **Kernel profile.** Use `rocprofv3` to inspect launch gaps, duration, memory
   traffic, MFMA use, occupancy, and LDS behavior.
7. **Communication profile.** Look for rank imbalance, serialization, missing
   overlap, and excess copies.
8. **End to end.** Measure serving latency and throughput, then check output
   quality.

Common failure patterns:

| Symptom | First check |
|---|---|
| Correct BF16, wrong quantized output | Scale orientation, block size, packing, and preshuffle |
| One expert is wrong | Route map, local expert ID, and that expert's scale/weight offset |
| Only the final result is wrong | Routing weights and combine order |
| First call is very slow | JIT compilation and cache path |
| Forced FlyDSL fails | Whether that shape and architecture have a supported FlyDSL path |
| One rank hangs | Send/receive counts, empty experts, stream events, and collective order |
| Small tests pass, graph capture fails | Stable buffers, warm-up coverage, and capture-safe communication |

## 14. Production Checklist

Before calling a K3 path production-ready, record:

- [ ] exact checkpoint, Quark, runtime image, AITER, and FlyDSL revisions;
- [ ] GPU architecture, firmware/driver, ROCm, PyTorch, and world size;
- [ ] selected backend and tuning row for each critical operator;
- [ ] quantization, scale, packing, and preshuffle contracts;
- [ ] local reference results for route, scatter, both GEMMs, activation, and
      combine;
- [ ] empty, odd, imbalanced, and maximum supported shapes;
- [ ] cold-start and steady-state measurements;
- [ ] graph-capture behavior and fallback behavior;
- [ ] multi-rank timeout, error propagation, and health checks;
- [ ] memory headroom for weights, caches, JIT/AOT code, and communication
      buffers;
- [ ] end-to-end accuracy or task-quality checks, not only kernel tolerances;
- [ ] profile traces at representative prompt lengths and concurrency.

## Key Takeaways

1. FlyDSL makes value ownership, copies, and MFMA layouts explicit.
2. AITER dispatches several backend families; an AITER call is not proof of a
   FlyDSL kernel.
3. The central public FlyDSL K3 path is quantized MoE: route, quantize/scatter,
   GEMM 1, SiTUv2/requantize, GEMM 2, and combine.
4. AITER exposes useful FlyDSL MLA building blocks, while KDA remains a
   Triton/Gluon boundary in the public K3 path described here.
5. Quantized layout, scales, preshuffling, tuning, and communication metadata
   are part of correctness.
6. Local operator tests and the eight-GPU MAD run answer different questions;
   both are required for production evidence.

## Exercises

1. Trace the value ownership in FlyDSL vector addition. Mark the logical
   coordinate, physical offset, thread, copy atom, and edge predicate.
2. Draw one preshuffled GEMM tile from global weights to the MFMA B fragment.
   Identify which facts must also be known by the checkpoint converter.
3. Log AITER dispatch for one K3 GEMM shape with and without
   `AITER_FLYDSL_FORCE=1`. Explain the result without assuming the forced path
   supports every shape.
4. Build an FP32 reference for SiTUv2 and compare it with dequantized
   `situv2_and_mul_quant` output for zeros, bounds, and rounding midpoints.
5. Make a buffer diagram for route → quant/scatter → GEMM 1 → SiTUv2/requant
   → GEMM 2 → combine. Label row order, dtype, scale shape, and owner.
6. Create a two-rank expert-parallel test with one empty expert and one hot
   expert. State the send and receive counts before running it.
7. From an AITER trace, classify every K3 operator as FlyDSL, Triton/Gluon,
   assembly/Opus, HIP/CK, or unknown. Leave unknown entries unknown until
   dispatch evidence identifies them.
