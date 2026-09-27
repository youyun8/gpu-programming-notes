# 15 – Triton in SGLang – Serving Kimi K3

> **Part VI · Kimi K3 Case Studies** · Prerequisite:
> [14 – Triton: From First Kernel to Production](14-triton.md) ·
> Programs: [`examples/15-triton-k3/`](examples/15-triton-k3/test_model_kernels.py) ·
> Next: [16 – FlyDSL in AITER: Kimi K3 on AMD GPUs](16-aiter-flydsl-kimi-k3.md)

This chapter moves from a small Triton program to a complete serving path.
SGLang is the system, and Kimi K3 is the traced end-to-end case. We begin with
one fused activation, then follow state and tokens through attention, experts,
sampling, communication, and production dispatch.

Three projects still have distinct roles. **Kimi K3** defines the open-weight
model. **AMD Quark** converts and quantizes checkpoints. **SGLang** owns
serving: scheduling, caches, distributed execution, and backend selection.
SGLang combines Triton with CUDA, CuTe DSL, AITER, FlashInfer, and vendor
libraries. A kernel called by SGLang is not automatically a Triton kernel.

The examples here are small enough to run in Triton's CPU interpreter. They
teach the same data flow as the production kernels without copying
version-specific implementation details.

**You will learn**

- the full map of SGLang Triton use-case families;
- how SiTU-GLU, MXFP4-style quantization, indexed caches, and recurrent KDA
  work;
- why decode, chunkwise prefill, and speculative decoding need different state
  algorithms;
- where MLA, MoE, sampling, and communication fit in a K3 request;
- why production dispatch sometimes selects CUDA, CuTe DSL, AITER, or a vendor
  library instead of Triton.

## 1. Source Boundaries and Pinned Versions

| Layer | Project | Responsibility |
|---|---|---|
| Model definition | [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) | Architecture, weights, and technical report |
| Model optimization | [AMD Quark](https://github.com/amd/Quark/tree/release/0.12) | Quantization, calibration, checkpoint conversion |
| Serving runtime | [SGLang](https://github.com/sgl-project/sglang) | Batching, cache ownership, scheduling, backend dispatch |
| Portable kernels | Triton / FLA | Attention, state updates, routing, and data movement |
| Hardware-specific kernels | AITER, CuTe DSL, FlashKDA, FlashInfer | Faster paths for supported devices and shapes |

The [official K3 repository](https://github.com/MoonshotAI/Kimi-K3) contains
the architecture, configuration, and model code, but **it has no Triton
source**. Public Triton implementations used to serve K3 live mainly in
[SGLang's KDA path](https://github.com/sgl-project/sglang/tree/fc9e1c8d296216ff1e216dfbe7286ef392448d28/python/sglang/srt/layers/attention/linear)
and [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention/tree/fa06b39153e09b2347387fff990d7d2d07ddd04b/fla/ops/kda).
[FlashKDA](https://github.com/MoonshotAI/FlashKDA) is **CUDA/CUTLASS, not
Triton**. SGLang can use a Triton KDA path as a portable alternative, but that
does not turn FlashKDA itself into Triton.

!!! note "Versions used in this chapter"

    Source links are pinned to Quark release/0.12-era code, SGLang commit
    `fc9e1c8`, and FLA commit `fa06b39`. These projects move quickly. Check
    signatures against the version installed in your environment.

## 2. Trace One K3 Request Through SGLang

A prompt enters SGLang and follows this path:

```text
schedule and allocate cache slots
  → preprocess projections, SiTU, and norms
  → run KDA or MLA attention
  → route tokens to MoE experts
  → quantize, move, and combine data as required
  → sample the next token
  → commit accepted cache and recurrent state
```

Prefill processes many prompt tokens. Decode processes one new token for each
active request. Speculative decoding processes several candidates, but commits
only the accepted prefix. Those modes share model weights; they do not always
share the same kernel.

K3 has 93 layers: 69 KDA layers and 24 gated Multi-head Latent Attention
(MLA) layers. It is also a sparse MoE with 896 routed experts and 16 selected
experts per token. This creates more work than one attention kernel.

### 2.1 SGLang Triton use-case families

The hierarchy below covers the kernel families that matter to K3 and the
related Triton machinery in SGLang.

| Family | SGLang work | Why Triton helps |
|---|---|---|
| Preprocessing, activation, and norms | short convolution, SiTU-GLU, Q/K L2 normalization, RMSNorm, gated output norm, gate/decay/beta transforms | fuse bandwidth-bound elementwise work and reductions |
| Cache and state movement | paged KV writes, recurrent-state gather/scatter, page movement, accepted-token maps | express masks, strides, and pointer indirection directly |
| KDA decode | recurrent state update and output projection | fixed-size state tiles map well to block tensors |
| KDA chunkwise prefill | local QK/KK products, triangular solves, chunk state passing | regular matrix tiles, scans, and variable-length masks |
| Speculative state | snapshots, parent-state selection, ReplaySSM ring-buffer replay | move or replay only state that may be accepted |
| MLA attention | RoPE, Q/K norm, paged cache, split-KV attention, output merge | fuse custom cache layouts and online reductions |
| MoE routing and movement | top-k, expert alignment, token permutation, grouped expert GEMM, weighted combine | handle irregular metadata around regular matrix work |
| Quantization | per-token/group FP8 or INT8, AWQ dequantization, MXFP4/MXFP8/NVFP4 helpers | combine reductions, scaling, conversion, and packing |
| Sampling | top-p/min-p renormalization, rejection sampling, tree reconstruction | avoid round trips between many short tensor operations |
| Communication | fused all-reduce/residual work, symmetric-memory helpers, sequence-parallel metadata | fuse local transforms with distributed data movement |

SGLang may choose a different backend for each row. Its kernel namespace also
contains CUDA JIT, CuTe DSL, AITER, Helion, FlashInfer, and library calls.
Always inspect the selected backend and dispatch condition before attributing
performance to Triton.

## 3. Stage 1: Preprocessing, SiTU, and Norms

The first useful lesson is fusion. K3 preprocessing includes projections,
normalization, gate transforms, and bounded activations. Each operation is
simple, but writing every intermediate tensor is expensive.

### 3.1 Fuse SiTU-GLU

K3 replaces the usual SwiGLU with a bounded SiTU activation. For gate input
\(g\) and up input \(u\), the teaching kernel computes

$$
y =
\left[\beta_1 \tanh(g/\beta_1)\sigma(g)\right]
\left[\beta_2 \tanh(u/\beta_2)\right],
\qquad \beta_1=4,\quad \beta_2=25.
$$

| Symbol | Meaning |
|---|---|
| \(g,u\) | The two halves of the projected input |
| \(\sigma\) | Sigmoid |
| \(\beta_1,\beta_2\) | Bounds used by the K3 configuration |
| \(y\) | Fused activation output |

[`situ_glu.py`](examples/15-triton-k3/situ_glu.py) loads both halves and writes
the result once:

```python
gate = tl.load(x_ptr + offsets, mask=mask, other=0.0)
up = tl.load(x_ptr + n + offsets, mask=mask, other=0.0)
bounded_gate = BETA1 * (2 / (1 + tl.exp(-2 * gate / BETA1)) - 1)
bounded_up = BETA2 * (2 / (1 + tl.exp(-2 * up / BETA2)) - 1)
out = bounded_gate * (1 / (1 + tl.exp(-gate))) * bounded_up
```

This is a useful Triton kernel because separate PyTorch operations would read
and write large temporary tensors. SGLang currently also has non-Triton SiTU
paths. The formula is K3-specific; the choice of implementation is runtime-
and hardware-specific.

### 3.2 Fuse norms only when the data flow allows it

KDA preprocesses Q, K, gates, decay, and beta before the recurrent update.
Q/K L2 normalization reduces over a head dimension, while RMSNorm reduces over
a hidden dimension. Both use the same basic Triton shape:

1. load one row in tiles;
2. accumulate squares in FP32;
3. multiply by an inverse square root;
4. apply any weight, gate, or output transform;
5. store once.

Fusion saves memory traffic, but it can also increase register use. Keep a
separate norm kernel when the fused tile spills, when another backend already
produces the normalized value, or when a dispatch boundary needs the
intermediate tensor. The first exercise adds Q/K normalization to the tested
KDA kernel.

## 4. Format Context: Quark MXFP4

K3 serving often starts from converted low-precision weights. Quark is useful
context because it defines the checkpoint representation; it is not the
SGLang scheduler or the KDA implementation.

Quark's public Triton code covers OCP microscaling and FP8 conversion. Its
[MX implementation](https://github.com/amd/Quark/blob/f7d8cefc7a6c973ff90cb87a6b154cbe3cc9aef2/quark/torch/kernel/mx/triton.py)
uses 32-value blocks for formats such as MXFP4:

1. reduce the block to its maximum absolute value;
2. derive a shared E8M0 power-of-two scale;
3. divide each value by that scale;
4. round to an E2M1 value;
5. pack two four-bit values into one byte;
6. swizzle scales into the layout required by the consumer.

[`mxfp4_qdq.py`](examples/15-triton-k3/mxfp4_qdq.py) implements steps 1–4 and
returns dequantized values. It uses the finite E2M1 magnitudes
\(\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}\).

!!! warning "Teaching QDQ is not a checkpoint converter"

    A compatible checkpoint must match Quark's exact scale rounding, NaN and
    zero rules, nibble order, padding, and scale swizzle. Use Quark's exported
    `qdq_mxfp4_triton` or `dq_mxfp4_triton` for interoperability. The example
    isolates the numerical idea so it can be tested without a full model.

Quark also has a Triton E5M3 conversion path with explicit subnormal and
round-to-nearest-even handling. These kernels belong to optimization and
fake-quantization. Quark does not own KDA, MoE routing, or the production
expert GEMMs.

## 5. Stage 2: Move Cache and Recurrent State

Offline examples keep state in batch order. A serving runtime cannot. Requests
arrive and finish at different times, so a scheduler maps each active request
to a persistent cache slot.

[`state_cache.py`](examples/15-triton-k3/state_cache.py) shows the core pattern:

```python
row = tl.program_id(0)
slot = tl.load(slots_ptr + row)
cols = tl.arange(0, BLOCK)
values = tl.load(source_ptr + row * width + cols, mask=cols < width)
tl.store(cache_ptr + slot * width + cols, values, mask=cols < width)
```

The same pattern appears in KV caches, recurrent KDA states, Mamba states, and
speculative-decoding metadata. Production kernels add page offsets, layer and
head strides, quantized storage, and bounds from request metadata.

Two writes to the same slot race. The wrapper therefore rejects duplicate
slots. A production scheduler either guarantees uniqueness or defines an
atomic/ordered update.

KV cache and recurrent state are not interchangeable. MLA stores token-indexed
keys and values, usually in pages. KDA carries a fixed-shape matrix state from
one token to the next. Both need request-to-slot indirection, but their
allocation, lifetime, and rollback rules differ.

## 6. Stage 3A: One Recurrent KDA Decode Step

For one token, a simplified KDA state update is

$$
D_t = \operatorname{Diag}(\alpha_t)S_{t-1},
$$

$$
r_t = v_t - D_t^\mathsf{T}k_t,\qquad
S_t = D_t + \beta_t k_t r_t^\mathsf{T},\qquad
o_t = S_t^\mathsf{T}q_t.
$$

| Symbol | Shape | Meaning |
|---|---:|---|
| \(q_t,k_t,\alpha_t\) | \(K\) | query, key, and channel decay |
| \(v_t,r_t,o_t\) | \(V\) | value, prediction residual, and output |
| \(S_t,D_t\) | \(K\times V\) | recurrent state and its decayed value |
| \(\beta_t\) | scalar in this teaching form | update strength |

[`kda_step.py`](examples/15-triton-k3/kda_step.py) launches a two-dimensional
grid. The first axis selects batch × head. The second selects a \(V\)-tile.
Each program loads the full \(K\) dimension and its own columns of state:

```python
decayed = state * alpha[:, None]
residual = v - tl.sum(decayed * k[:, None], axis=0)
updated = decayed + k[:, None] * (beta * residual)[None, :]
out = tl.sum(updated * q[:, None], axis=0)
```

The state stays FP32 because rounding error is recurrent. Q, K, and V may use a
lower precision in a tuned kernel. Real K3 paths also fuse input extraction,
Q/K L2 normalization, bounded decay, beta activation, and output gating.

This teaching form makes the recurrence visible. It is not a claim that every
production K3 decode uses Triton. SGLang can dispatch to its Triton KDA path,
FlashKDA, CuTe DSL, FlashInfer, Helion, or another supported backend according
to the device, mode, dtype, and installed extensions.

## 7. Stage 3B: Chunkwise Prefill and Speculative State

### 7.1 Decode is recurrent

Decode receives one new token. The recurrent kernel above is appropriate:
read one state, update it, write one state. Continuous batching supplies the
cache slot for each request.

### 7.2 Prefill is chunkwise

Prefill receives many prompt tokens. Applying decode one token at a time leaves
the GPU underused. Chunkwise KDA instead:

1. computes gate prefix sums;
2. forms intra-chunk QK and KK products;
3. builds a WY representation and solves a triangular system;
4. propagates state between chunks;
5. reconstructs outputs.

The public implementations under
[FLA `ops/kda`](https://github.com/fla-org/flash-linear-attention/tree/fa06b39153e09b2347387fff990d7d2d07ddd04b/fla/ops/kda)
show the complete algorithm, including variable-length packed input. Rewriting
that code as a short tutorial kernel would hide the important invariants, so
this chapter uses the production API as the advanced exercise.

The chunk size is a tuning choice, not a change to the model. Small chunks have
more state-passing overhead. Large chunks increase temporary storage and can
reduce occupancy. Packed variable-length prompts also need offsets and masks
so one request never reads another request's tokens.

### 7.3 Speculative decoding and ReplaySSM

Draft tokens may be rejected. A kernel must not overwrite the committed state
until verification finishes. Common designs keep intermediate snapshots,
select a parent state for each verification-tree node, and fold only accepted
tokens into the main state. SGLang's ReplaySSM path adds a ring buffer to avoid
recomputing every state. The scheduler supplies parent and accepted-token
metadata; kernels gather, replay, and commit the matching states.

The safety rule is simple: speculative work may update temporary state, but
only accepted tokens may update the persistent KDA state or MLA KV cache.

## 8. Stage 3C: MLA Attention

Twenty-four K3 layers use gated MLA rather than KDA. This path has a different
state model:

1. project and normalize Q and K;
2. apply RoPE to the positional components;
3. write compressed KV data into the paged cache;
4. read the pages selected for each request;
5. run prefill or split-KV decode attention;
6. merge partial outputs and apply the output gate.

Triton is a good fit for cache transforms, RoPE, normalization, online
softmax, split reduction, and output fusion. A tuned MLA attention core may
instead use FlashInfer, FlashMLA, CuTe DSL, AITER, or another architecture-
specific implementation. KDA state is a recurrent matrix; MLA state is a
token-indexed paged cache. SGLang's hybrid attention layer must manage both.

## 9. Stage 4: MoE Routing and Data Movement

After attention, K3 routes each token to 16 of 896 routed experts. The full
path is larger than one grouped matrix multiplication:

```text
router logits
  → top-k expert selection and weights
  → count and align tokens by expert
  → permute or dispatch tokens
  → expert GEMM 1
  → SiTU and optional activation quantization
  → expert GEMM 2
  → weighted combine and restore token order
```

Triton works well for top-k helpers, expert alignment, permutation,
quantization, grouped GEMM, and weighted combine. In expert parallel mode,
dispatch and combine also cross devices. The metadata is irregular, but each
expert's matrix work is regular after tokens are grouped.

Production MoE GEMMs are often CUDA/CUTLASS, CuTe DSL, AITER, CK, or vendor
library kernels. Those paths can use architecture-specific matrix
instructions, layouts, persistent scheduling, and tuned low-precision
epilogues. Triton remains useful for portable paths and the data movement
around those GEMMs.

## 10. Stage 5: Quantization Through the Serving Path

Quantization appears at several boundaries:

- checkpoint conversion packs weights and scales;
- weight loaders preserve the exact nibble and scale layout;
- activation kernels compute per-token or per-group scales;
- GEMMs consume MXFP4, MXFP8, FP8, INT8, AWQ, or another supported format;
- cache kernels may convert values to a quantized storage format.

Do not treat every format with “FP4” in its name as equivalent. Scale type,
group size, rounding, packing, padding, and swizzle are part of the interface.
The Quark example in section 4 teaches MXFP4 numerics, while the production
consumer decides the required physical layout.

## 11. Stage 6: Sampling and Communication

After the final model layer, SGLang still has GPU work. Sampling can apply
temperature and penalties, renormalize top-p or min-p distributions, draw or
reject candidates, and reconstruct a speculative tree. These operations use
short reductions, masks, scans, and gathers, so Triton can replace a chain of
small framework launches.

Tensor, data, expert, and sequence parallelism add communication. Triton can
prepare symmetric-memory buffers, fuse residual or scaling work with local
copies, and handle sequence-parallel metadata. It does not replace the network
transport. NCCL, RCCL, or another communication backend still moves data
between devices.

The same Triton patterns now repeat across the request: masked loads, row and
online reductions, pointer indirection, stable permutation, tile GEMM, scans,
and fusion.

## 12. Backend Dispatch: Why a Path May Not Be Triton

SGLang selects an operator implementation at runtime or during model setup.
The decision can depend on:

- CUDA or ROCm, GPU architecture, and installed extensions;
- prefill, decode, or speculative verification mode;
- dtype, quantization scheme, head size, page size, and batch shape;
- graph-capture and distributed-execution requirements;
- whether a tuned backend supports the exact model feature.

CUDA/CUTLASS is often chosen for NVIDIA-specific tensor-core layouts and
hand-tuned pipelines. CuTe DSL exposes explicit layout and MMA control for
similar specialization. AITER packages AMD-tuned CK, HIP, assembly, Triton,
and other implementations. FlashInfer and vendor libraries provide maintained
specialized operators. Triton is strongest when portability, fast iteration,
fusion, and custom data movement matter more than the last device-specific
optimization.

Therefore:

- **FlashKDA is CUDA/CUTLASS, not Triton.**
- AITER is a dispatch and operator library; an AITER call is not proof that
  the selected kernel is Triton.
- A `.py` wrapper may launch CUDA, CuTe DSL, AITER, or a library kernel.
- Performance results must name the selected backend, dtype, shape, and
  hardware.

Trace the backend from the model layer to its registry, mode-specific
implementation, and final kernel launch. Do not infer it from a Python
filename or from the presence of Triton elsewhere in SGLang.

## Key Takeaways

1. Quark, K3, and SGLang occupy different layers of the stack.
2. The official K3 repository has no Triton source, and FlashKDA is a
   CUDA/CUTLASS project.
3. KDA decode is a recurrent state update; KDA prefill is a chunkwise matrix
   algorithm.
4. MLA uses a paged token cache, while KDA carries a fixed-shape recurrent
   state.
5. Triton is useful throughout serving, but SGLang dispatch may select a more
   specialized backend.

## 13. Production Integration

A teaching kernel becomes an SGLang backend only after its contracts are
clear:

1. define accepted devices, dtypes, shapes, modes, and quantization formats;
2. make the wrapper validate strides, alignment, cache slots, and metadata;
3. register the implementation behind an explicit dispatch condition;
4. keep a known-correct fallback for unsupported inputs;
5. connect prefill, decode, and speculative-state interfaces separately;
6. preserve transactional state: commit only accepted tokens;
7. expose the selected backend in logs or metrics;
8. pin SGLang, Triton, PyTorch, and ROCm/CUDA versions together.

Keep committed recurrent state in FP32 unless a validated format allows
another choice. Treat Quark packing and scale swizzling as part of the data
format. Test NVIDIA and AMD separately: portable source does not imply the
same tile shapes, compiler output, or tuning. Use K3's license name;
“open-weight” is more precise than unqualified “open source.”

## 14. Testing

Run the existing examples from the repository root:

```bash
cd tutorials/examples/15-triton-k3
python3 test_model_kernels.py
```

Without a GPU, the script enables `TRITON_INTERPRET=1`. It checks a
non-power-of-two SiTU input, mutated KDA state and output, all-zero and partial
MX blocks, and an indexed-cache round trip.

Production tests should add:

- odd heads and hidden sizes, partial pages, empty masks, and zero-token work;
- packed variable-length prompts and mixed prefill/decode batches;
- duplicate or invalid cache slots and request cancellation;
- speculative branches with zero, partial, and full acceptance;
- quantization boundary values, packing order, padding, and scale layouts;
- parity for every enabled backend and an unsupported-shape fallback;
- multi-rank expert dispatch, combine, and communication failures.

Check numerical results before timing. Reassociated reductions need explicit
tolerances, and recurrent tests should compare every intermediate state, not
only the final token.

## 15. Profiling

Warm the JIT and caches before measuring. On NVIDIA, the small KDA example can
be inspected with:

```bash
ncu -k regex:kda_step_kernel python3 test_model_kernels.py
```

Use the matching ROCm profiler on AMD. Profile the complete SGLang request as
well as individual kernels. Separate prefill throughput, decode latency, and
speculative verification; their shapes and bottlenecks differ.

Record at least:

- the selected backend and exact software revisions;
- device, dtype, quantization, batch, sequence, and cache shapes;
- kernel time, launch count, memory traffic, occupancy, and spills;
- communication time and overlap for distributed runs;
- end-to-end latency and throughput after warm-up.

A faster isolated kernel can lose end-to-end if it needs extra conversions,
cache movement, synchronization, or dispatch overhead.

## Exercises

1. Fuse Q/K L2 normalization into `kda_step_kernel`. Keep the sums in FP32.
2. Extend the state cache with separate request, layer, and head strides.
3. Pack two E2M1 codes per byte, then compare the byte order with Quark 0.12.
4. Add a sequence loop around the KDA reference and verify every intermediate
   state, not only the final output.
5. Call the pinned FLA chunkwise KDA API on packed variable-length prompts and
   compare it with recurrent decode.
6. Design a ReplaySSM test where the accepted prefix has lengths 0, 1, and the
   full draft length.
7. Draw the cache layouts for one KDA layer and one MLA layer under continuous
   batching.
8. Trace SGLang's K3 backend selection on your machine. List which operators
   use Triton, CUDA/CUTLASS, CuTe DSL, AITER, FlashInfer, or a vendor library,
   then explain each fallback.
