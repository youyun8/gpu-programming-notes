# Tutorials

Work through these in order; each one points at the practice problems that exercise it.

| # | Topic | Practice |
|---|-------|----------|
| 00 | [Getting started (no local GPU required)](00-getting-started.md) | – |
| 01 | [CUDA execution model: threads, blocks, grids](01-execution-model.md) | Vector Addition |
| 02 | [Memory hierarchy & coalescing](02-memory-hierarchy.md) | Matrix Transpose, Matrix Copy |
| 03 | [Parallel reduction](03-parallel-reduction.md) | Reduction, Softmax, Dot Product |
| 04 | [Tiled matrix multiplication](04-tiled-matmul.md) | Matrix Multiplication, GEMM |

## Planned

- 05 – Profiling with Nsight Compute & the roofline model
- 06 – Warp-level primitives (`__shfl_sync`, cooperative groups)
- 07 – Scan / prefix sum
- 08 – Convolution (1D / 2D) and stencils
- 09 – Softmax, LayerNorm & online algorithms (FlashAttention-style)
- 10 – Tensor cores (WMMA / MMA / CuTe)
- 11 – Triton for the same problems

## Reading list

- *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), 4th ed.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Mark Harris, *Optimizing Parallel Reduction in CUDA*
- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*
