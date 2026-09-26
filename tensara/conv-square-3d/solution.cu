// 3D Convolution with a cubic kernel (Tensara)
// https://tensara.org/problems/conv-square-3d
//
// "Same" 3D convolution of a size^3 volume with a K^3 kernel (K <= 11), zero
// padding K/2. One thread per output voxel, threadIdx.x along the innermost
// axis (coalesced; neighbouring threads share cached input lines); the kernel
// (<= 1331 taps) is staged in shared memory and read as broadcasts.
#include <cuda_runtime.h>

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kMaxTaps = 11 * 11 * 11;

__global__ void conv3dSame(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ out, int n, int k) {
    __shared__ float s_w[kMaxTaps];
    const int tid = threadIdx.y * kBlockX + threadIdx.x;
    for (int i = tid; i < k * k * k; i += kBlockX * kBlockY) s_w[i] = w[i];
    __syncthreads();
    const int x = blockIdx.x * kBlockX + threadIdx.x;
    const int y = blockIdx.y * kBlockY + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= n || y >= n) return;
    const int p = k / 2;
    float acc = 0.0f;
    for (int dz = 0; dz < k; ++dz) {
        const int zz = z + dz - p;
        if (zz < 0 || zz >= n) continue;
        for (int dy = 0; dy < k; ++dy) {
            const int yy = y + dy - p;
            if (yy < 0 || yy >= n) continue;
            const float* row = in + (static_cast<size_t>(zz) * n + yy) * n;
            const float* wr = s_w + (dz * k + dy) * k;
            for (int dx = 0; dx < k; ++dx) {
                const int xx = x + dx - p;
                if (xx >= 0 && xx < n) acc = fmaf(row[xx], wr[dx], acc);
            }
        }
    }
    out[(static_cast<size_t>(z) * n + y) * n + x] = acc;
}

// A, B, C are device pointers
extern "C" void solution(const float* A, const float* B, float* C, size_t size, size_t K) {
    const int n = static_cast<int>(size);
    const dim3 grid((n + kBlockX - 1) / kBlockX, (n + kBlockY - 1) / kBlockY, n);
    conv3dSame<<<grid, dim3(kBlockX, kBlockY)>>>(A, B, C, n, static_cast<int>(K));
}
