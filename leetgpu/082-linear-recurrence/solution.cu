// Linear Recurrence (LeetGPU)
// https://leetgpu.com/challenges/linear-recurrence
//
// h[t] = a[t] * h[t-1] + x[t] is a composition of affine maps
// f_t(h) = a_t h + x_t, and composition is associative:
//     (A1, X1) then (A2, X2) = (A1 * A2, A2 * X1 + X2).
// So the recurrence is a parallel scan. One block per sequence:
//   1. each of the 1024 threads folds its contiguous chunk into one map;
//   2. a block-wide scan over the maps (fp64) gives each thread its h_in;
//   3. each thread replays its chunk sequentially from h_in, writing h.
#include <cuda_runtime.h>

constexpr int kThreads = 1024;

struct Affine {
    double a;
    double x;
};

__device__ __forceinline__ Affine compose(Affine first, Affine second) {
    return Affine{first.a * second.a, second.a * first.x + second.x};
}

__global__ void linearRecurrence(const float* a, const float* x, float* h, int len) {
    __shared__ double s_a[32];
    __shared__ double s_x[32];
    const float* ar = a + static_cast<size_t>(blockIdx.x) * len;
    const float* xr = x + static_cast<size_t>(blockIdx.x) * len;
    float* hr = h + static_cast<size_t>(blockIdx.x) * len;

    const int per_thread = (len + kThreads - 1) / kThreads;
    const int begin = threadIdx.x * per_thread;
    const int end = min(begin + per_thread, len);

    // Identity map, then fold the chunk. Position 0 has no predecessor (a is ignored).
    Affine agg{1.0, 0.0};
    for (int t = begin; t < end; ++t) agg = compose(agg, Affine{t == 0 ? 0.0 : static_cast<double>(ar[t]), xr[t]});

    // Inclusive block scan of maps (warp shuffles, then across warps).
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    Affine incl = agg;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const Affine prev{__shfl_up_sync(0xffffffffu, incl.a, offset), __shfl_up_sync(0xffffffffu, incl.x, offset)};
        if (lane >= offset) incl = compose(prev, incl);
    }
    if (lane == 31) {
        s_a[warp] = incl.a;
        s_x[warp] = incl.x;
    }
    __syncthreads();
    if (warp == 0) {
        Affine w{s_a[lane], s_x[lane]};
        for (int offset = 1; offset < 32; offset <<= 1) {
            const Affine prev{__shfl_up_sync(0xffffffffu, w.a, offset), __shfl_up_sync(0xffffffffu, w.x, offset)};
            if (lane >= offset) w = compose(prev, w);
        }
        s_a[lane] = w.a;
        s_x[lane] = w.x;
    }
    __syncthreads();
    // Exclusive map of all earlier threads applied to h = 0 gives h_in.
    Affine excl_in_warp{__shfl_up_sync(0xffffffffu, incl.a, 1), __shfl_up_sync(0xffffffffu, incl.x, 1)};
    if (lane == 0) excl_in_warp = Affine{1.0, 0.0};
    const Affine before_warp = warp > 0 ? Affine{s_a[warp - 1], s_x[warp - 1]} : Affine{1.0, 0.0};
    double h_prev = compose(before_warp, excl_in_warp).x;

    for (int t = begin; t < end; ++t) {
        const double cur = (t == 0 ? 0.0 : static_cast<double>(ar[t]) * h_prev) + xr[t];
        hr[t] = static_cast<float>(cur);
        h_prev = cur;
    }
}

// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {
    linearRecurrence<<<B, kThreads>>>(a, x, h, L);
    cudaDeviceSynchronize();
}
