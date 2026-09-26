# 00 – Getting Started

## Where to run code

You do **not** need a local NVIDIA GPU:

| Option | Notes |
|--------|-------|
| [LeetGPU](https://leetgpu.com) playground | Free, runs CUDA/Triton/PyTorch in the browser, has a GPU emulator mode. |
| [Tensara](https://tensara.org) | Submissions are benchmarked on real GPUs (T4, A100, H100, …). |
| Google Colab / Kaggle | Free T4 GPU; use `!nvcc` in a notebook cell. |
| Cloud VM (Lambda, RunPod, …) | Needed for Nsight profiling at full fidelity. |

## Minimal local toolchain

```bash
nvcc --version                         # CUDA toolkit
nvidia-smi                             # driver + GPU
nvcc -O3 -arch=native -o hello hello.cu && ./hello
```

## Anatomy of a submission

Both platforms give you **device pointers** and call a C-linkage entry point;
you write the kernel and launch it.

```cpp
#include <cuda_runtime.h>

__global__ void myKernel(const float* in, float* out, int n) { /* ... */ }

// LeetGPU entry point
extern "C" void solve(const float* in, float* out, int n) {
    myKernel<<<(n + 255) / 256, 256>>>(in, out, n);
    cudaDeviceSynchronize();
}

// Tensara entry point
extern "C" void solution(const float* in, float* out, size_t n) { /* ... */ }
```

Always copy the exact signature from the starter code; parameter order and
integer types (`int` vs `size_t`) differ per problem.

## Error checking while developing

```cpp
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

myKernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());        // launch configuration errors
CUDA_CHECK(cudaDeviceSynchronize());   // errors raised while running
```

`compute-sanitizer ./app` catches out-of-bounds accesses and races.
