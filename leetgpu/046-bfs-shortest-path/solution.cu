// BFS Shortest Path (LeetGPU)
// https://leetgpu.com/challenges/bfs-shortest-path
//
// Level-synchronous BFS on a 4-connected grid, run by ONE persistent block so
// the level barrier is a cheap __syncthreads() instead of a kernel launch per
// level (paths in a maze can be hundreds of thousands of levels long).
// The frontier lives in two global queues; threads expand frontier cells in
// parallel and claim unvisited neighbours with atomicCAS on the visited map,
// so every cell is enqueued exactly once. Total work is O(rows * cols).
#include <cuda_runtime.h>

constexpr int kThreads = 1024;

__global__ void bfsGrid(const int* grid, int* result, int* visited, int* q0, int* q1, int rows, int cols, int start, int goal) {
    __shared__ int s_cur_size;
    __shared__ int s_next_size;
    __shared__ int s_found;
    if (threadIdx.x == 0) {
        s_cur_size = 1;
        s_next_size = 0;
        s_found = (start == goal);
        q0[0] = start;
        visited[start] = 1;
    }
    __syncthreads();
    int* cur = q0;
    int* next = q1;
    int level = 0;
    while (!s_found && s_cur_size > 0) {
        const int cur_size = s_cur_size;
        for (int i = threadIdx.x; i < cur_size; i += kThreads) {
            const int cell = cur[i];
            const int r = cell / cols, c = cell % cols;
            const int nbr[4] = {r > 0 ? cell - cols : -1, r + 1 < rows ? cell + cols : -1, c > 0 ? cell - 1 : -1,
                                c + 1 < cols ? cell + 1 : -1};
            for (int t = 0; t < 4; ++t) {
                const int nb = nbr[t];
                if (nb < 0 || grid[nb] != 0) continue;
                if (atomicCAS(&visited[nb], 0, 1) == 0) {
                    if (nb == goal) s_found = 1;
                    next[atomicAdd(&s_next_size, 1)] = nb;
                }
            }
        }
        __syncthreads();
        ++level;
        if (threadIdx.x == 0) {
            s_cur_size = s_next_size;
            s_next_size = 0;
        }
        int* t = cur;
        cur = next;
        next = t;
        __syncthreads();
    }
    if (threadIdx.x == 0) result[0] = s_found ? level : -1;
}

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row, int start_col, int end_row,
                      int end_col) {
    const size_t cells = static_cast<size_t>(rows) * cols;
    int* buf = nullptr;
    cudaMalloc(&buf, 3 * cells * sizeof(int));
    cudaMemset(buf, 0, cells * sizeof(int));
    bfsGrid<<<1, kThreads>>>(grid, result, buf, buf + cells, buf + 2 * cells, rows, cols, start_row * cols + start_col,
                            end_row * cols + end_col);
    cudaDeviceSynchronize();
    cudaFree(buf);
}
