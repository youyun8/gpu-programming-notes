# 00 – Getting Started

This chapter sets up everything the later chapters and the practice pages
assume:

- where to run CUDA code (you do not need a local GPU);
- what `nvcc` does with a `.cu` file;
- the shape of a LeetGPU or Tensara submission;
- how to check errors, time a kernel, and judge whether the time is good.

## 1. Where to run code

| Option | Notes |
|--------|-------|
| [LeetGPU](https://leetgpu.com) playground | Free. Runs CUDA, Triton and PyTorch in the browser, and has a GPU emulator mode. |
| [Tensara](https://tensara.org) | Submissions are benchmarked on real GPUs (T4, A100, H100, …). |
| Google Colab / Kaggle | Free T4 GPU; use `!nvcc` in a notebook cell. |
| Cloud VM (Lambda, RunPod, …) | Needed for Nsight profiling at full fidelity. |
| This repo's [cuemu](../tools/cuemu/README.md) | Runs any solution on your **CPU** against the official reference tests. It checks correctness only, not speed. |

A sensible loop is:

1. write the kernel locally;
2. check it with cuemu (seconds, no GPU);
3. submit, or run it on Colab for timing.

## 2. The toolchain

```bash
nvcc --version                         # CUDA toolkit
nvidia-smi                             # driver and GPU
nvcc -O3 -arch=native -o hello hello.cu && ./hello
```

`nvcc` splits a `.cu` file into host code and device code:

```
hello.cu ──► host part  ──► g++/clang ──────────────────────────► host object
         └─► device part ──► PTX (virtual ISA, compute_XX) ──► ptxas ──► SASS (sm_XX)
                                                   └────── both embedded in a "fatbin"
```

- **PTX** is a portable virtual instruction set. The driver can compile it
  just in time for a newer GPU.
- **SASS** is the real machine code of one architecture.
- `-arch=sm_80` produces SASS for A100 plus PTX for `compute_80`.
  `-arch=native` targets the GPU in your machine.
- `-gencode arch=compute_90,code=sm_90` spells out both parts explicitly.
- `cuobjdump -sass app` shows the SASS; reading it is how you check that
  a loop was unrolled or that a load became `LDG.E.128`.

Useful flags:

| Flag | Effect |
|------|--------|
| `-O3` | Host optimisation. Device code is optimised by default. |
| `-lineinfo` | Source lines in profilers, with no slowdown. |
| `-G` | Device debug build. Very slow; use only with `cuda-gdb`. |
| `--use_fast_math` | Approximate `expf`, `sinf`, division, flush denormals. Often fails tight tolerances. |
| `-Xptxas -v` | Prints registers, shared memory and spills per kernel. |
| `-std=c++17` | Modern C++ in device code (templates, `constexpr`, lambdas). |

## 3. Anatomy of a submission

Both platforms give you **device pointers** and call a C-linkage entry
point. You write the kernel and launch it.

```cpp
#include <cuda_runtime.h>

__global__ void scaleKernel(const float* in, float* out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 2.0f * in[idx];
}

// LeetGPU entry point
extern "C" void solve(const float* in, float* out, int n) {
    constexpr int kBlockSize = 256;
    scaleKernel<<<(n + kBlockSize - 1) / kBlockSize, kBlockSize>>>(in, out, n);
    cudaDeviceSynchronize();
}

// Tensara entry point
extern "C" void solution(const float* in, float* out, size_t n) { /* ... */ }
```

| | LeetGPU | Tensara |
|---|---|---|
| Entry point | `solve(...)` | `solution(...)` |
| Sizes | usually `int` | usually `size_t` |
| Timing | wall time of `solve` | GPU time of `solution`, averaged over runs |
| Tolerance | per problem | per problem (`rtol`, `atol`), often looser for big reductions |
| Hardware | selectable GPU, or emulator | T4, A100, H100, L40S, … |

Always copy the exact signature from the starter code: the parameter order
and the integer types differ from problem to problem.

The pointers are device pointers. Dereferencing one on the host crashes.
Some Tensara problems pass a `shape` array that may live on either side;
`cudaMemcpy(..., cudaMemcpyDefault)` copies it correctly in both cases
(see [Tensara – Argmax](../tensara/argmax/)).

## 4. Error checking while developing

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

Kernel launches are **asynchronous**. An out-of-bounds write shows up as an
error on the *next* synchronizing call, which may be far from the bug.
`compute-sanitizer ./app` (memcheck, racecheck, synccheck) pinpoints
out-of-bounds accesses and shared-memory races.

## 5. Timing a kernel

Use CUDA events. They are recorded on the GPU's stream, so they measure
GPU time, not launch overhead:

```cpp
cudaEvent_t start_event, stop_event;
cudaEventCreate(&start_event);
cudaEventCreate(&stop_event);

scaleKernel<<<grid, block>>>(d_in, d_out, n);          // warm-up
cudaEventRecord(start_event);
for (int rep = 0; rep < kReps; ++rep) scaleKernel<<<grid, block>>>(d_in, d_out, n);
cudaEventRecord(stop_event);
cudaEventSynchronize(stop_event);

float elapsed_ms = 0.0f;
cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
const double seconds_per_call = elapsed_ms * 1e-3 / kReps;
```

A time alone says little. Turn it into rates and compare them with the
hardware limits:

$$
\beta_{\text{eff}} = \frac{Q}{t}, \qquad F_{\text{eff}} = \frac{W}{t}, \qquad
I = \frac{W}{Q}
$$

| Symbol | Meaning |
|---|---|
| $t$ | measured time per call (seconds) |
| $Q$ | bytes the kernel *must* move to and from DRAM (inputs read once, outputs written once) |
| $W$ | useful floating-point operations (an FMA counts as 2) |
| $\beta_{\text{eff}}$ | effective bandwidth, bytes/s |
| $F_{\text{eff}}$ | achieved throughput, flop/s |
| $I$ | arithmetic intensity, flops per byte |

The **roofline model** bounds the best possible time:

$$
t \ \ge\ T_{\min} = \max\left(\frac{W}{F},\ \frac{Q}{\beta}\right), \qquad
I^{\star} = \frac{F}{\beta}
$$

| Symbol | Meaning |
|---|---|
| $F$ | peak compute throughput of the GPU for the data type used |
| $\beta$ | peak DRAM bandwidth |
| $T_{\min}$ | lower bound on the kernel time |
| $I^{\star}$ | ridge point: kernels with $I < I^{\star}$ are memory-bound, those with $I > I^{\star}$ compute-bound |

Some rough numbers:

| GPU | FP32 $F$ | $\beta$ | $I^{\star}$ (FP32) |
|---|---|---|---|
| T4 | ~8 TFLOP/s | ~320 GB/s | ~25 flop/B |
| A100 40 GB | ~19.5 TFLOP/s | ~1.55 TB/s | ~13 flop/B |
| H100 SXM | ~67 TFLOP/s | ~3.35 TB/s | ~20 flop/B |

Worked example: $y = 2x$ on $n = 2^{26}$ floats has $W = n$ and
$Q = 8n = 537$ MB, so $I = 1/8$. That is far below any ridge point, so the
kernel is memory-bound, with $T_{\min} = 0.35$ ms on an A100. Measuring
0.40 ms means 87 % of peak bandwidth: done. Every problem page on this site
has a *Cost analysis* section that does this calculation.

## 6. Testing on the CPU with cuemu

[cuemu](../tools/cuemu/README.md) translates a solution into ordinary C++,
builds it with `clang++`, and runs every CUDA thread as a user-space fiber,
one thread block at a time. `__syncthreads()` is a real barrier and warp
shuffles really exchange values between the 32 lanes. The output is then
compared with the platform's own PyTorch reference, with the platform's
tolerances, on small versions of the official test cases:

```bash
scripts/fetch_upstream.sh                              # problem definitions -> .upstream/
python3 tools/cuemu/run_tests.py leetgpu/001-vector-add
python3 tools/cuemu/run_tests.py tensara/softmax
python3 tools/cuemu/run_tests.py --reverse leetgpu/004-reduction   # reversed thread order
```

It catches indexing bugs, missing bounds checks, out-of-bounds writes
(guard pages after every buffer), wrong layouts and numerical mistakes.
`--reverse` schedules threads in reverse order, which exposes many missing
barriers. It says nothing about speed, and since blocks run one at a time
it cannot find races *between* blocks; use `compute-sanitizer` on a real
GPU for those.

## 7. Checklist before submitting

- [ ] Signature copied exactly (order, `int` vs `size_t`, `const`).
- [ ] Every thread guards its index (`if (idx < n)`), including the last
      partial block.
- [ ] 64-bit indices when $n$ or byte offsets can exceed $2^{31}$.
- [ ] Buffers that are accumulated into (atomics) are zeroed first.
- [ ] Temporary `cudaMalloc`s are freed; `cudaDeviceSynchronize()` before
      `cudaFree` if kernels may still be using them.
- [ ] The cost analysis says what the bound is, and the measured time is
      close to it.

## Practice

- [LeetGPU – Vector Addition](../leetgpu/001-vector-add/),
  [Tensara – Vector Addition](../tensara/vector-addition/)
- [LeetGPU – Color Inversion](../leetgpu/007-color-inversion/),
  [Tensara – ReLU](../tensara/relu/)
