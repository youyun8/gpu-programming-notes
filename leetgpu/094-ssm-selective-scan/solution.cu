// SSM Selective Scan (LeetGPU)
// https://leetgpu.com/challenges/ssm-selective-scan
//
// h[t, n] = exp(delta_t A[d, n]) h[t-1, n] + delta_t B[t, n] u_t
// y[t]    = sum_n C[t, n] h[t, n] + skip[d] u_t          (per batch b, channel d)
// Channels are independent, so each thread owns one (b, d) and keeps its whole
// state vector (d_state <= 64) and its row of A in registers, stepping through
// time. All threads of a block share b, so B[b, t, :] and C[b, t, :] are staged
// in shared memory for 32 timesteps at a time and read as broadcasts; u and
// delta reads are coalesced across d (channels-last layout).
#include <cuda_runtime.h>

constexpr int kThreads = 128;
constexpr int kMaxState = 64;
constexpr int kTimeChunk = 32;

__global__ void __launch_bounds__(kThreads)
selectiveScan(const float* u, const float* delta, const float* a, const float* b_proj, const float* c_proj,
              const float* skip, float* y, int seq, int d_model, int d_state) {
    // B_t and C_t of 32 time steps, shared by all channels of the block.
    __shared__ float s_b[kTimeChunk][kMaxState];
    __shared__ float s_c[kTimeChunk][kMaxState];
    // One thread per (batch, channel d): the whole state vector h (d_state <= 64) and the
    // channel's row of A stay in registers for the entire sequence.
    const int batch = blockIdx.y;
    const int d = blockIdx.x * kThreads + threadIdx.x;
    const bool active = d < d_model;

    float h[kMaxState];
    float a_row[kMaxState];
#pragma unroll
    for (int n = 0; n < kMaxState; ++n) {
        h[n] = 0.0f;
        a_row[n] = (active && n < d_state) ? a[static_cast<size_t>(d) * d_state + n] : 0.0f;
    }
    const float skip_d = active ? skip[d] : 0.0f;

    // Walk the sequence in chunks of 32 steps: stage B and C (every thread must reach
    // the barriers, hence the late `continue` for inactive threads).
    for (int t0 = 0; t0 < seq; t0 += kTimeChunk) {
        const int steps = min(kTimeChunk, seq - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < steps * d_state; i += kThreads) {
            const int tt = i / d_state;
            const int n = i % d_state;
            const size_t g = (static_cast<size_t>(batch) * seq + t0 + tt) * d_state + n;
            s_b[tt][n] = b_proj[g];
            s_c[tt][n] = c_proj[g];
        }
        __syncthreads();
        if (!active) continue;
        // Sequential recurrence per channel: h = exp(dt A) h + dt B u, y = C . h + D u.
        for (int tt = 0; tt < steps; ++tt) {
            const size_t idx = (static_cast<size_t>(batch) * seq + t0 + tt) * d_model + d;
            const float dt = delta[idx];
            const float ut = u[idx];
            float acc = 0.0f;
#pragma unroll
            for (int n = 0; n < kMaxState; ++n) {
                if (n < d_state) {
                    h[n] = expf(dt * a_row[n]) * h[n] + (dt * s_b[tt][n]) * ut;
                    acc = fmaf(s_c[tt][n], h[n], acc);
                }
            }
            y[idx] = acc + skip_d * ut;
        }
    }
}

// u, delta, A, B, C, skip, y are device pointers
extern "C" void solve(const float* u, const float* delta, const float* A, const float* B, const float* C,
                      const float* skip, float* y, int batch, int seq_len, int d_model, int d_state) {
    const dim3 grid((d_model + kThreads - 1) / kThreads, batch);
    selectiveScan<<<grid, kThreads>>>(u, delta, A, B, C, skip, y, seq_len, d_model, d_state);
    cudaDeviceSynchronize();
}
