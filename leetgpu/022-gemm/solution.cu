// General Matrix Multiplication (GEMM) (LeetGPU)
// https://leetgpu.com/challenges/general-matrix-multiplication-gemm
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
}
