# Tutorials

Work through these in order; each one points at the practice problems that exercise it.

| # | Topic | Practice |
|---|-------|----------|
| 00 | [Getting started (no local GPU required)](00-getting-started.md) | – |
| 01 | [CUDA execution model: threads, blocks, grids](01-execution-model.md) | Vector Addition |
| 02 | [Memory hierarchy & coalescing](02-memory-hierarchy.md) | Matrix Transpose, Matrix Copy |
| 03 | [Parallel reduction](03-parallel-reduction.md) | Reduction, Softmax, Dot Product |
| 04 | [Tiled matrix multiplication](04-tiled-matmul.md) | Matrix Multiplication, GEMM |
| 05 | [AMD CDNA3 & MFMA: from CUDA to wave64 matrix cores](05-amd-cdna3-mfma.md) | [`amd/mfma_gemm.hip`](amd/mfma_gemm.hip) |
| 06 | [Inside a hand-written AMD GEMM: aiter's bf16 asm kernels](06-aiter-asm-gemm.md) | Disassemble aiter `.co` files |
| 07 | [hipBLASLt & TensileLite: GEMM kernels written by a program](07-hipblaslt-tensilelite.md) | `hipblaslt-bench`, offline tuning |
| 08 | [Deploying this site](08-deploying-this-site.md) | MkDocs, GitHub Pages |

Chapters 05–07 form an AMD track. They assume 04, and they read best in
order: first the hardware and instruction (05), then a hand-written kernel
(06), then the generator that produces thousands of such kernels (07).

## Planned

- Profiling with Nsight Compute / rocprofv3 & the roofline model
- Warp-level primitives (`__shfl_sync`, cooperative groups)
- Scan / prefix sum
- Convolution (1D / 2D) and stencils
- Softmax, LayerNorm & online algorithms (FlashAttention-style)
- Tensor cores on NVIDIA (WMMA / MMA / CuTe)
- Triton for the same problems

## Reading list

- *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), 4th ed.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Mark Harris, *Optimizing Parallel Reduction in CUDA*
- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*
- AMD, *AMD Instinct MI300 ISA Reference Guide* (CDNA3)
- AMD, [Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- aiter, `docs/isa_kernel_optimization.md`
- Osama et al., *Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU* (2023)
