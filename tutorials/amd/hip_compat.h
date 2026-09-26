// Minimal stand-in for <hip/hip_runtime.h> so device code can be compiled
// (and its ISA inspected) with a stock clang that has no ROCm installation:
//
//   clang++ -x hip -nogpuinc -nogpulib --cuda-device-only --offload-arch=gfx942 \
//           -O3 -S -o - mfma_gemm.hip
//
// With a real ROCm toolchain (hipcc / amdclang++) the genuine header is used.
#pragma once

#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define threadIdx_x (threadIdx.x)
#define blockIdx_x (blockIdx.x)
#define blockIdx_y (blockIdx.y)
#else
#define HIP_COMPAT_DEVICE_ONLY 1
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))
#define threadIdx_x (__builtin_amdgcn_workitem_id_x())
#define blockIdx_x (__builtin_amdgcn_workgroup_id_x())
#define blockIdx_y (__builtin_amdgcn_workgroup_id_y())
// Same sequence HIP uses: release fence, s_barrier, acquire fence (workgroup scope).
#define __syncthreads()                                   \
  do {                                                    \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); \
    __builtin_amdgcn_s_barrier();                         \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); \
  } while (0)
#endif
