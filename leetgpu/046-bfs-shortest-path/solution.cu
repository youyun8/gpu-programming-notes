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
    // Level-synchronous BFS in a single block: frontier sizes and the "goal reached" flag in shared memory.
    __shared__ int s_cur_size;
    __shared__ int s_next_size;
    __shared__ int s_found;
    // The start cell forms the first frontier.
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
    // One iteration per BFS level, until the goal is found or the frontier is empty.
    while (!s_found && s_cur_size > 0) {
        const int cur_size = s_cur_size;
        for (int i = threadIdx.x; i < cur_size; i += kThreads) {
            // Expand a frontier cell: up, down, left and right neighbours (-1 = off the grid).
            const int cell = cur[i];
            const int r = cell / cols, c = cell % cols;
            const int nbr[4] = {r > 0 ? cell - cols : -1, r + 1 < rows ? cell + cols : -1, c > 0 ? cell - 1 : -1,
                                c + 1 < cols ? cell + 1 : -1};
            for (int t = 0; t < 4; ++t) {
                const int nb = nbr[t];
                if (nb < 0 || grid[nb] != 0) continue;
                // atomicCAS claims each free cell exactly once; the claiming thread appends it to the next frontier.
                if (atomicCAS(&visited[nb], 0, 1) == 0) {
                    if (nb == goal) s_found = 1;
                    next[atomicAdd(&s_next_size, 1)] = nb;
                }
            }
        }
        // Level complete: swap the frontiers (the barrier also publishes s_found and the new size).
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
    // Distance in steps, or -1 if the goal is unreachable.
    if (threadIdx.x == 0) result[0] = s_found ? level : -1;
}

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row, int start_col, int end_row,
                      int end_col) {
    const size_t cells = static_cast<size_t>(rows) * cols;
    // Scratch: visited flags (zeroed) and two frontier queues, each one int per cell.
    int* buf = nullptr;
    cudaMalloc(&buf, 3 * cells * sizeof(int));
    cudaMemset(buf, 0, cells * sizeof(int));
    bfsGrid<<<1, kThreads>>>(grid, result, buf, buf + cells, buf + 2 * cells, rows, cols, start_row * cols + start_col,
                            end_row * cols + end_col);
    cudaDeviceSynchronize();
    cudaFree(buf);
}
