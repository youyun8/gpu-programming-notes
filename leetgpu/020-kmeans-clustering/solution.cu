// K-Means Clustering (LeetGPU)
// https://leetgpu.com/challenges/k-means-clustering
//
// max_iterations rounds of Lloyd's algorithm on 2-D points:
//   assign: one thread per point, centroids staged in shared memory, squared
//           distance dx^2 + dy^2 (no FMA contraction, like the reference),
//           ties -> lowest index; per-block sums/counts accumulate in shared
//           memory (fp64) and are flushed with one global atomic per cluster;
//   update: one thread per cluster, mean = sum / count (unchanged if empty).
// Labels are those of the last assignment, as in the reference.
#include <cuda_runtime.h>
#include <cfloat>

constexpr int kThreads = 256;
constexpr int kMaxK = 1000;

__global__ void assignPoints(const float* x, const float* y, int* labels, const float* cx, const float* cy, double* sum_x,
                             double* sum_y, unsigned int* counts, int n, int k) {
    __shared__ float s_cx[kMaxK];
    __shared__ float s_cy[kMaxK];
    __shared__ double s_sx[kMaxK];
    __shared__ double s_sy[kMaxK];
    __shared__ unsigned int s_cnt[kMaxK];
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        s_cx[c] = cx[c];
        s_cy[c] = cy[c];
        s_sx[c] = 0.0;
        s_sy[c] = 0.0;
        s_cnt[c] = 0;
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float px = x[i], py = y[i];
        float best = FLT_MAX;
        int best_c = 0;
        for (int c = 0; c < k; ++c) {
            const float dx = __fsub_rn(px, s_cx[c]);
            const float dy = __fsub_rn(py, s_cy[c]);
            const float d = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
            if (d < best) {
                best = d;
                best_c = c;
            }
        }
        labels[i] = best_c;
        atomicAdd(&s_sx[best_c], static_cast<double>(px));
        atomicAdd(&s_sy[best_c], static_cast<double>(py));
        atomicAdd(&s_cnt[best_c], 1u);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        if (s_cnt[c]) {
            atomicAdd(&sum_x[c], s_sx[c]);
            atomicAdd(&sum_y[c], s_sy[c]);
            atomicAdd(&counts[c], s_cnt[c]);
        }
    }
}

__global__ void updateCentroids(float* cx, float* cy, double* sum_x, double* sum_y, unsigned int* counts, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    if (counts[c]) {
        cx[c] = static_cast<float>(sum_x[c] / counts[c]);
        cy[c] = static_cast<float>(sum_y[c] / counts[c]);
    }
    sum_x[c] = 0.0;
    sum_y[c] = 0.0;
    counts[c] = 0;
}

// all pointers are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels, float* initial_centroid_x,
                      float* initial_centroid_y, float* final_centroid_x, float* final_centroid_y, int sample_size, int k,
                      int max_iterations) {
    cudaMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);
    void* buf = nullptr;
    cudaMalloc(&buf, k * (2 * sizeof(double) + sizeof(unsigned int)));
    cudaMemset(buf, 0, k * (2 * sizeof(double) + sizeof(unsigned int)));
    double* sum_x = static_cast<double*>(buf);
    double* sum_y = sum_x + k;
    unsigned int* counts = reinterpret_cast<unsigned int*>(sum_y + k);
    int blocks = (sample_size + kThreads - 1) / kThreads;
    blocks = blocks > 1024 ? 1024 : blocks;
    for (int it = 0; it < max_iterations; ++it) {
        assignPoints<<<blocks, kThreads>>>(data_x, data_y, labels, final_centroid_x, final_centroid_y, sum_x, sum_y, counts,
                                           sample_size, k);
        updateCentroids<<<(k + 255) / 256, 256>>>(final_centroid_x, final_centroid_y, sum_x, sum_y, counts, k);
    }
    cudaDeviceSynchronize();
    cudaFree(buf);
}
