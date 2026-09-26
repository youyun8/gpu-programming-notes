# 15 – Triton in Quark, Kimi K3, and SGLang

> **Part IV · Portable Model Kernels** · Prerequisites: [14 – Triton](14-triton.md)
> and [13 – Softmax, LayerNorm, and FlashAttention](13-softmax-attention.md) ·
> Programs: [`examples/15-triton-k3/`](examples/15-triton-k3/test_model_kernels.py) ·
> Next: [05 – CDNA3 and MFMA](05-amd-cdna3-mfma.md)

This chapter moves from small Triton kernels to real model systems. It follows
three projects with different jobs:

- **AMD Quark** converts and quantizes models. Its Triton kernels implement
  numerical formats such as MXFP4 and FP8.
- **Kimi K3** is an open-weight mixture-of-experts model. Its main new operator
  is Kimi Delta Attention (KDA).
- **SGLang** serves models. It combines Triton with CUDA, CuTe DSL, AITER,
  FlashInfer, and vendor libraries. A file in SGLang is not necessarily Triton.

The examples here are small enough to run in Triton's CPU interpreter. They
teach the same data flow as the production kernels without copying
version-specific implementation details.

**You will learn**

- where Triton is used in Quark, K3 serving, and SGLang;
- how SiTU-GLU, MXFP4-style quantization, indexed caches, and recurrent KDA work;
- how prefill, decode, continuous batching, and speculative decoding change a
  kernel;
- how to decide whether an example is a faithful format implementation, a
  teaching model, or a production backend.

## 1. Keep the Project Boundaries Clear

| Layer | Project | Responsibility |
|---|---|---|
| Model definition | [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) | Architecture, weights, and technical report |
| Model optimization | [AMD Quark](https://github.com/amd/Quark/tree/release/0.12) | Quantization, calibration, checkpoint conversion |
| Serving runtime | [SGLang](https://github.com/sgl-project/sglang) | Batching, cache ownership, scheduling, backend dispatch |
| Portable kernels | Triton / FLA | Quantization, attention, state updates, routing, and data movement |
| Hardware-specific kernels | AITER, CuTe DSL, FlashKDA, FlashInfer | Faster paths for supported devices and shapes |

The official K3 repository does not contain Triton source. Public Triton
implementations used to serve K3 live mainly in
[SGLang's KDA path](https://github.com/sgl-project/sglang/tree/fc9e1c8d296216ff1e216dfbe7286ef392448d28/python/sglang/srt/layers/attention/linear)
and [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention/tree/fa06b39153e09b2347387fff990d7d2d07ddd04b/fla/ops/kda).
[FlashKDA](https://github.com/MoonshotAI/FlashKDA) is a CUDA/CUTLASS kernel,
not a Triton kernel; Triton is its portable fallback.

!!! note "Versions used in this chapter"

    Source links are pinned to Quark 0.12-era code, SGLang commit
    `fc9e1c8`, and FLA commit `fa06b39`. These projects move quickly. Check
    signatures against the version installed in your environment.

## 2. The K3 Operator Map

K3 has 93 layers: 69 KDA layers and 24 gated Multi-head Latent Attention
(MLA) layers. It is also a sparse MoE with 896 routed experts and 16 selected
experts per token. This creates more work than one attention kernel.

| Area | Typical kernels | Why Triton helps |
|---|---|---|
| Input preparation | short causal convolution, Q/K normalization, gate and beta transforms | several small operations can become one launch |
| KDA decode | recurrent state update and output projection | fixed-size state tiles map well to block tensors |
| KDA prefill | local QK/KK products, triangular solve, chunk state propagation | regular dense tiles and scans |
| MLA | RoPE, QK norm, paged cache, split-KV attention, output merge | custom cache layouts and fusion |
| MoE | top-k, token permutation, grouped expert GEMM, weighted combine | irregular metadata around regular GEMMs |
| Quantization | MXFP4/MXFP8/FP8 conversion and scaling | reductions plus bit/data conversion |
| Activation and norm | SiTU-GLU, RMSNorm, gated output norm | bandwidth-bound chains become one kernel |
| Serving metadata | indexed state reads/writes, page tables, accepted-token maps | masks and pointer arithmetic are concise |

SGLang may choose a different backend for each row. Its kernel namespace also
contains CUDA JIT, CuTe DSL, AITER, Helion, FlashInfer, and library calls.
Always inspect the selected backend and dispatch condition before attributing
performance to Triton.

## 3. Case 1: Fuse SiTU-GLU

K3 replaces the usual SwiGLU with a bounded activation. For gate input \(g\)
and up input \(u\), the teaching kernel computes

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

## 4. Case 2: Understand Quark's MX Kernels

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

## 5. Case 3: Move State Through a Serving Cache

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

## 6. Case 4: One Recurrent KDA Decode Step

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

## 7. Decode, Prefill, and Speculation Are Different Algorithms

### 7.1 Decode

Decode receives one new token. The recurrent kernel above is appropriate:
read one state, update it, write one state. Continuous batching supplies the
cache slot for each request.

### 7.2 Prefill

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

### 7.3 Speculative decoding

Draft tokens may be rejected. A kernel must not overwrite the committed state
until verification finishes. Common designs keep intermediate snapshots,
select a parent state for each verification-tree node, and fold only accepted
tokens into the main state. SGLang's ReplaySSM path adds a ring buffer to avoid
recomputing every state.

## 8. The Wider SGLang Triton Map

K3 uses only part of SGLang. The same Triton skills apply to the runtime's
other operator groups:

| Group | Representative use cases |
|---|---|
| Attention | paged and split-KV decode, prefill, GQA/MQA, sliding windows, sinks, softcaps, MLA |
| KV cache | indexed writes, gather/scatter, page movement, quantized-cache conversion |
| MoE | top-k, expert alignment, token permutation, grouped GEMM, weighted combine |
| Quantization | per-token/group FP8, INT8, AWQ dequantization, MXFP8 and NVFP4 helpers |
| Norm and activation | RMSNorm, residual + norm, gated norm, SiLU/GELU-and-mul |
| Mamba/SSM | causal convolution, chunk BMM, scan, state passing and cache updates |
| Sampling | top-p/min-p renormalization, rejection sampling and tree reconstruction |
| Communication | fused all-reduce/residual helpers and sequence-parallel metadata |

The repeated patterns are more important than the file count: masked loads,
row reductions, online reductions, pointer indirection, stable permutation,
tile GEMM, and kernel fusion.

## 9. Run and Validate

```bash
cd tutorials/examples/15-triton-k3
python3 test_model_kernels.py
```

Without a GPU, the script enables `TRITON_INTERPRET=1`. It checks
non-power-of-two sizes, the mutated KDA state, an all-zero MX block, a partial
MX block, and an indexed-cache round trip.

On a GPU, profile both the kernel and the surrounding runtime:

```bash
ncu -k regex:kda_step_kernel python3 test_model_kernels.py
```

Check numerical results before timing. Reassociated reductions and recurrent
state updates need explicit tolerances. Never compare latency before warming
the JIT cache.

## 10. Production Checklist

1. Pin SGLang, Triton, PyTorch, and ROCm/CUDA versions together.
2. Record which backend dispatch selected; “SGLang performance” is not a
   Triton-only result.
3. Keep committed recurrent state in FP32 unless a validated format says
   otherwise.
4. Test odd shapes, partial pages, empty masks, and variable-length batches.
5. Treat Quark's packing and swizzling as part of the data format.
6. Test CUDA and AMD separately. Portable source does not imply identical tile
   shapes or tuning.
7. Use the K3 license name. “Open-weight” is more precise than unqualified
   “open source.”

## Key Takeaways

1. Quark, K3, and SGLang occupy different layers of the stack.
2. Triton is strongest where reduction, data conversion, and fusion meet
   regular tiles.
3. KDA decode is a recurrent state update; KDA prefill is a chunkwise matrix
   algorithm.
4. Serving adds cache indirection, variable lengths, and transactional state.
5. A small teaching kernel must state what it omits from the production format
   or backend.

## Exercises

1. Fuse Q/K L2 normalization into `kda_step_kernel`. Keep the sums in FP32.
2. Extend the state cache with separate request, layer, and head strides.
3. Pack two E2M1 codes per byte, then compare the byte order with Quark 0.12.
4. Add a sequence loop around the KDA reference and verify every intermediate
   state, not only the final output.
5. Trace SGLang's K3 backend selection on your machine and list which
   operators use Triton, AITER, CUDA, or a library.
