// FP16 Batched Matrix Multiplication (LeetGPU)
// https://leetgpu.com/challenges/fp16-batched-matrix-multiplication
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
}
