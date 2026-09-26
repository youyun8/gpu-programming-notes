// Multi-Agent Simulation (boids alignment step) (LeetGPU)
// https://leetgpu.com/challenges/multi-agent-simulation
//
// For every agent: average velocity of all others within r = 5, then
// v' = v + 0.05 (v_avg - v), p' = p + v'.
// Brute force O(N^2) with shared-memory tiles of 256 agents (float4 loads of
// [x, y, vx, vy]). The neighbour test must match the reference bit for bit at
// the boundary, so dist^2 = dx*dx + dy*dy is evaluated without FMA contraction.
#include <cuda_runtime.h>

constexpr int kBlock = 256;
constexpr float kRadiusSq = 25.0f;
constexpr float kAlpha = 0.05f;

__global__ void flock(const float4* agents, float4* next, int n) {
    __shared__ float4 tile[kBlock];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float4 me = i < n ? agents[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float sum_vx = 0.0f, sum_vy = 0.0f;
    int count = 0;
    for (int t0 = 0; t0 < n; t0 += kBlock) {
        if (t0 + threadIdx.x < n) tile[threadIdx.x] = agents[t0 + threadIdx.x];
        __syncthreads();
        const int m = min(kBlock, n - t0);
        for (int t = 0; t < m; ++t) {
            const float4 o = tile[t];
            const float dx = __fsub_rn(me.x, o.x);
            const float dy = __fsub_rn(me.y, o.y);
            const float d2 = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
            if (t0 + t != i && d2 < kRadiusSq) {
                sum_vx += o.z;
                sum_vy += o.w;
                ++count;
            }
        }
        __syncthreads();
    }
    if (i < n) {
        const float avg_x = count > 0 ? sum_vx / count : me.z;
        const float avg_y = count > 0 ? sum_vy / count : me.w;
        const float vx = me.z + kAlpha * (avg_x - me.z);
        const float vy = me.w + kAlpha * (avg_y - me.w);
        next[i] = make_float4(me.x + vx, me.y + vy, vx, vy);
    }
}

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {
    flock<<<(N + kBlock - 1) / kBlock, kBlock>>>(reinterpret_cast<const float4*>(agents), reinterpret_cast<float4*>(agents_next), N);
    cudaDeviceSynchronize();
}
