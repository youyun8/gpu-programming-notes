# Tutorials

Work through these in order; each one points at the practice problems that exercise it.

| # | Topic | Practice |
|---|-------|----------|
| 00 | [Getting started: toolchain, submissions, timing, roofline](00-getting-started.md) | Vector Addition, ReLU |
| 01 | [CUDA execution model: indexing, warps, latency hiding, occupancy](01-execution-model.md) | Vector Addition, Matrix Addition |
| 02 | [Memory hierarchy: coalescing, bank conflicts, transpose](02-memory-hierarchy.md) | Matrix Transpose, Matrix Copy |
| 03 | [Parallel reduction: work/depth, shuffles, accuracy, monoids](03-parallel-reduction.md) | Reduction, Softmax, Dot Product |
| 04 | [Tiled matrix multiplication: intensity, shared and register tiling](04-tiled-matmul.md) | Matrix Multiplication, GEMM |
| 05 | [AMD CDNA3 & MFMA: from CUDA to wave64 matrix cores](05-amd-cdna3-mfma.md) | [`amd/mfma_gemm.hip`](amd/mfma_gemm.hip) |
| 06 | [Inside a hand-written AMD GEMM: aiter's bf16 asm kernels](06-aiter-asm-gemm.md) | Disassemble aiter `.co` files |
| 07 | [hipBLASLt & TensileLite: GEMM kernels written by a program](07-hipblaslt-tensilelite.md) | `hipblaslt-bench`, offline tuning |
| 08 | [Deploying this site](08-deploying-this-site.md) | MkDocs, GitHub Pages |

Chapters 05–07 form an AMD track. They assume 04, and they read best in
order: first the hardware and instruction (05), then a hand-written kernel
(06), then the generator that produces thousands of such kernels (07).

## The One Formula to Keep in Mind

Every chapter and every problem page comes back to the roofline bound
(chapter 00):

$$
T_{\min} = \max\left(\frac{W}{F},\ \frac{Q}{\beta}\right), \qquad I = \frac{W}{Q}, \qquad I^{\star} = \frac{F}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $W$ | Useful flops of the kernel |
| $Q$ | Bytes it must move to and from DRAM |
| $F$ | Peak compute throughput |
| $\beta$ | Peak DRAM bandwidth |
| $T_{\min}$ | The best achievable time |
| $I, I^{\star}$ | Arithmetic intensity of the kernel, and the GPU's ridge point |

Chapters 01–03 are about reaching $Q/\beta$ for kernels with $I < I^{\star}$
(almost all elementwise, reduction and normalization problems). Chapters
04–07 are about raising the *effective* $I$ of matrix multiplication at
every level of the memory hierarchy until $W/F$ is the bound.

## How the Pages Are Organised

- Each tutorial states its goals, derives the key formulas (each followed
  by a table of symbols), shows complete kernels, and ends with practice
  problems.
- Each problem page has the same sections: *Problem*, *Formulation*,
  *Approach*, *Cost analysis*, *Pitfalls*, *Verification*, *Related*, and
  then the full solution source.

## Planned

- Profiling with Nsight Compute / rocprofv3 & the roofline model
- Warp-level primitives (`__shfl_sync`, cooperative groups)
- Scan / prefix sum (see the Cumsum and Prefix Sum problem pages meanwhile)
- Convolution (1D / 2D) and stencils
- Softmax, LayerNorm & online algorithms (FlashAttention-style)
- Tensor cores on NVIDIA (WMMA / MMA / CuTe)
- Triton for the same problems

## Reading List

- *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), 4th ed.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Mark Harris, *Optimizing Parallel Reduction in CUDA*
- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*
- AMD, *AMD Instinct MI300 ISA Reference Guide* (CDNA3)
- AMD, [Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- aiter, `docs/isa_kernel_optimization.md`
- Osama et al., *Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU* (2023)
