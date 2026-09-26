// GAE Reverse Scan (LeetGPU)
// https://leetgpu.com/challenges/parallel-reverse-scan-gae
//
// delta_t = r_t + gamma V_{t+1} - V_t (V_S = 0);  A_t = delta_t + c A_{t+1}, c = gamma lambda.
// The reverse recurrence is a scan of affine maps x -> c x + delta run from the
// end of the sequence. One block per sequence: each thread folds a contiguous
// chunk (walking backwards), a block scan (fp64) hands every thread the
// advantage flowing in from its right neighbour chunk, then the chunk is
// replayed right to left.
#include <cuda_runtime.h>

constexpr int kThreads = 1024;

__global__ void gae(const float* rewards, const float* values, float* adv, int s, float gamma, float lam) {
    __shared__ double s_mul[32];
    __shared__ double s_add[32];
    const float* r = rewards + static_cast<size_t>(blockIdx.x) * s;
    const float* v = values + static_cast<size_t>(blockIdx.x) * s;
    float* a = adv + static_cast<size_t>(blockIdx.x) * s;
    // GAE: A_t = delta_t + (gamma lambda) A_{t+1}, with delta_t = r_t + gamma V_{t+1} - V_t
    // (V after the last step is 0). One block per trajectory.
    const double c = static_cast<double>(gamma) * lam;
    auto delta = [&](int t) {
        const float next_v = t + 1 < s ? v[t + 1] : 0.0f;
        return static_cast<double>(r[t] + gamma * next_v - v[t]);
    };

    // Thread k owns the k-th chunk counted from the END of the sequence, so the
    // block scan runs in the direction of the recurrence.
    const int per_thread = (s + kThreads - 1) / kThreads;
    const int hi = s - threadIdx.x * per_thread;   // exclusive upper bound
    const int lo = max(hi - per_thread, 0);
    // Map of this chunk: A_lo = mul * A_hi + add.
    double mul = 1.0, add = 0.0;
    for (int t = hi - 1; t >= lo; --t) {
        add = delta(t) + c * add;
        mul *= c;
    }
    if (hi <= 0) {
        mul = 1.0;
        add = 0.0;
    }
    // Inclusive scan of maps: composing chunk k after chunks 0..k-1 (to its right).
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    double im = mul, ia = add;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const double pm = __shfl_up_sync(0xffffffffu, im, offset);
        const double pa = __shfl_up_sync(0xffffffffu, ia, offset);
        if (lane >= offset) {
            ia = ia + im * pa;  // apply earlier map first, then this one
            im = im * pm;
        }
    }
    if (lane == 31) {
        s_mul[warp] = im;
        s_add[warp] = ia;
    }
    __syncthreads();
    if (warp == 0) {
        double wm = s_mul[lane], wa = s_add[lane];
        for (int offset = 1; offset < 32; offset <<= 1) {
            const double pm = __shfl_up_sync(0xffffffffu, wm, offset);
            const double pa = __shfl_up_sync(0xffffffffu, wa, offset);
            if (lane >= offset) {
                wa = wa + wm * pa;
                wm = wm * pm;
            }
        }
        s_mul[lane] = wm;
        s_add[lane] = wa;
    }
    __syncthreads();
    // Advantage entering this chunk from the right = (exclusive prefix)(A_S = 0).add
    double em = __shfl_up_sync(0xffffffffu, im, 1);
    double ea = __shfl_up_sync(0xffffffffu, ia, 1);
    if (lane == 0) {
        em = 1.0;
        ea = 0.0;
    }
    const double carry = (warp > 0 ? ea + em * s_add[warp - 1] : ea);
    // Replay the chunk backwards from the incoming advantage, writing A_t (in double).
    double running = carry;
    for (int t = hi - 1; t >= lo; --t) {
        running = delta(t) + c * running;
        a[t] = static_cast<float>(running);
    }
}

// rewards, values, advantages are device pointers
extern "C" void solve(const float* rewards, const float* values, float* advantages, float gamma, float lam, int B, int S) {
    // One block per batch row.
    gae<<<B, kThreads>>>(rewards, values, advantages, S, gamma, lam);
    cudaDeviceSynchronize();
}
