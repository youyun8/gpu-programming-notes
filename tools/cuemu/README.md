# cuemu – Run CUDA Solutions on a CPU

`cuemu` lets every solution in this repository be tested without an NVIDIA
GPU. It is how CI checks all 185 solutions against the platforms' own
reference implementations.

## How It Works

1. **Translate** (`cuemu.py`). A few source-level rewrites turn a `.cu` file
   into C++:
   - CUDA headers are dropped; local headers (`#include "x.cuh"`) are pasted
     in, so they are translated too.
   - `extern __shared__ T x[]` becomes `T* x = cuemu::dynamicSmem<T>()`.
   - `kernel<<<grid, block, smem>>>(args)` becomes
     `cuemu::Launcher(grid, block, smem)(kernel, args)`.

   Everything else, including `__global__`, `threadIdx`, `__shared__`,
   atomics, `half`/`bfloat16`, vector types and WMMA, is provided by
   `cuemu.h`.
2. **Build.** The result is built with `clang++ -std=c++20 -O2 -shared` into a
   shared library and cached by content hash (`cuemu.py run` builds an
   executable instead, for programs with their own `main()`).
3. **Execute.** Each CUDA thread runs as a user-space fiber, and blocks run
   one at a time. This gives real semantics:
   - `__syncthreads()` is a real barrier. Divergent barriers are reported as
     deadlocks.
   - Warp operations (`__shfl*_sync`, `__ballot_sync`, `__match_any_sync`,
     `__reduce_*_sync`) rendezvous the 32 lanes of a warp.
   - WMMA fragments check the alignment rules (32-byte pointers,
     `ldm·sizeof(T) % 16 == 0`) that real hardware silently punishes.
   - `cp.async` through `<cuda_pipeline.h>` (`__pipeline_memcpy_async`,
     `__pipeline_commit`, `__pipeline_wait_prior`) is emulated with
     **deferred** copies: data lands only when a wait covers its group, so
     reading a pipeline stage too early fails as it would on a GPU.
   - Kernels that use inline PTX `ldmatrix` / `mma.sync.m16n8k16` can call
     `cuemuLdmatrix` and `cuemuMmaM16N8K16` under `#ifdef __CUEMU__`; they
     implement the PTX-documented fragment layouts (see
     [tutorials/gemm/09-mma-sync.cu](../../tutorials/gemm/09-mma-sync.cu)).
4. **Check** (`run_tests.py`). The shared library is loaded with `ctypes` and
   run on the upstream test cases. The output is compared with the upstream
   PyTorch reference, using the upstream tolerances.
   - **Guard pages** after every buffer catch out-of-bounds writes.
   - Inputs are **checked for modification**.
   - Every problem runs in its own process with a timeout.
   - `--reverse` schedules threads in reverse order, which exposes missing
     barriers.
   - Tensara's benchmark-sized cases are too large to emulate, so
     `tensara_small_cases.py` derives scaled-down and odd-sized variants from
     the same generators.
   - `lowp_reference.py` provides CPU stand-ins for the GPU-only references
     used by the FP4/FP8 problems: flashinfer NVFP4 and `scaled_mm` with
     swizzled scales.

## Usage

```bash
scripts/fetch_upstream.sh                                # clone problem definitions into .upstream/
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-test.txt

python3 tools/cuemu/run_tests.py leetgpu/001-vector-add  # one problem
python3 tools/cuemu/run_tests.py --all -j 8              # everything
python3 tools/cuemu/run_tests.py --platform tensara --all --json build/tensara.json
python3 tools/cuemu/run_tests.py --reverse leetgpu/004-reduction   # race detection
python3 tools/cuemu/cuemu.py translate leetgpu/022-gemm/solution.cu  # see the translation
python3 tools/cuemu/cuemu.py run tutorials/gemm/03-cp-async.cu -- --test  # a program with its own main()
```

| Environment variable | Default | Meaning |
|----------------------|---------|---------|
| `CUEMU_MAX_ELEMENTS` | 2²² | Skip test cases with larger tensors (a README can raise it with `cuemu_max_elements:`) |
| `CUEMU_TIMEOUT` | 300 | Per-problem timeout, seconds |
| `CUEMU_CXX` | `clang++` | Compiler |

## Limits

- **Correctness only.** Timings mean nothing on a CPU.
- **Unsupported:** cooperative groups, CUB, Thrust, cuBLAS/cuDNN and inline
  PTX (other than through the `__CUEMU__` hooks above). Solutions here avoid
  them on purpose.
- **Scheduling:** threads interleave only at barriers and warp collectives. A
  data race without a barrier may pass here and fail on a GPU; `--reverse`
  catches the most common kind.
- **Top-p sampling** cannot reproduce `torch.multinomial`'s RNG. For it the
  runner checks that every sampled token lies in the nucleus.
