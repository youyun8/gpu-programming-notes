// Matrix Power (LeetGPU)
// https://leetgpu.com/challenges/matrix-power
//
// A^P by binary exponentiation (O(log P) GEMMs instead of P - 1), multiplying
// in the same order as torch.linalg.matrix_power so rounding matches closely:
//   P = 1: A;  P = 2: A A;  P = 3: (A A) A;
//   else: walk the bits of P, z = A, A^2, A^4, ...; result = result @ z per set bit.
// Each product is the 64 x 64 register-blocked SGEMM.
#include <cuda_runtime.h>

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads) sgemm(const float* a, const float* b, float* c, int n) {
    __shared__ float a_tile[kTileK][kTileM + 4];
    __shared__ float b_tile[kTileK][kTileN + 4];
    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < n; k0 += kTileK) {
        for (int i = tid; i < kTileM * kTileK; i += kThreads) {
            const int r = i / kTileK;
            const int kk = i % kTileK;
            a_tile[kk][r] = (row0 + r < n && k0 + kk < n) ? a[static_cast<size_t>(row0 + r) * n + k0 + kk] : 0.0f;
        }
        for (int i = tid; i < kTileK * kTileN; i += kThreads) {
            const int kk = i / kTileN;
            const int cc = i % kTileN;
            b_tile[kk][cc] = (k0 + kk < n && col0 + cc < n) ? b[static_cast<size_t>(k0 + kk) * n + col0 + cc] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float a_frag[4];
            float b_frag[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a_frag[i] = a_tile[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b_frag[j] = b_tile[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const int r = row0 + ty + 16 * i;
        if (r >= n) continue;
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + tx + 16 * j;
            if (col < n) c[static_cast<size_t>(r) * n + col] = acc[i][j];
        }
    }
}

static void matmul(const float* a, const float* b, float* c, int n) {
    const dim3 grid((n + kTileN - 1) / kTileN, (n + kTileM - 1) / kTileM);
    sgemm<<<grid, kThreads>>>(a, b, c, n);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    if (P == 1) {
        cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        return;
    }
    if (P == 2 || P == 3) {
        float* tmp = nullptr;
        cudaMalloc(&tmp, bytes);
        if (P == 2) {
            matmul(input, input, output, N);
        } else {
            matmul(input, input, tmp, N);
            matmul(tmp, input, output, N);
        }
        cudaDeviceSynchronize();
        cudaFree(tmp);
        return;
    }
    // Four scratch buffers: z, next z, result, next result.
    float* buf = nullptr;
    cudaMalloc(&buf, 4 * bytes);
    float* z = buf;
    float* z_next = buf + static_cast<size_t>(N) * N;
    float* result = buf + 2 * static_cast<size_t>(N) * N;
    float* result_next = buf + 3 * static_cast<size_t>(N) * N;
    const float* z_cur = input;
    const float* res_cur = nullptr;
    bool first_z = true;
    int p = P;
    while (p > 0) {
        const int bit = p % 2;
        p /= 2;
        if (!first_z) {
            matmul(z_cur, z_cur, z_next, N);
            float* t = z;
            z = z_next;
            z_next = t;
            z_cur = z;
        }
        first_z = false;
        if (bit == 1) {
            if (res_cur == nullptr) {
                cudaMemcpy(result, z_cur, bytes, cudaMemcpyDeviceToDevice);
            } else {
                matmul(res_cur, z_cur, result_next, N);
                float* t = result;
                result = result_next;
                result_next = t;
            }
            res_cur = result;
        }
    }
    cudaMemcpy(output, res_cur, bytes, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaFree(buf);
}
