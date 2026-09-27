"""Chapter 14, kernel 3: tiled matrix multiplication with grouped ordering and autotuning.

C (M x N) = A (M x K) @ B (K x N). Each program computes one BLOCK_M x BLOCK_N
tile of C, looping over K in steps of BLOCK_K; `tl.dot` compiles to tensor-core
instructions, and the compiler stages the loads through shared memory with
`num_stages` buffers (Matrix Multiplication 3 and 4 do the same by hand).
Programs are ordered in groups of GROUP_M tile rows so that consecutive
programs reuse the same tiles of B from L2 (Matrix Multiplication 6).
"""
import torch
import triton
import triton.language as tl

CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
]


@triton.jit
def tile_coordinates(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    """Maps a linear program id to a (tile row, tile column), GROUP_M tile rows at a time."""
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    per_group = GROUP_M * tiles_n                  # programs in one group of tile rows
    group = pid // per_group
    first_m = group * GROUP_M
    group_m = min(tiles_m - first_m, GROUP_M)      # the last group may be shorter
    tile_m = first_m + (pid % per_group) % group_m
    tile_n = (pid % per_group) // group_m
    return tile_m, tile_n


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    tile_m, tile_n = tile_coordinates(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    # Pointer tiles: a 2-D block of addresses built by broadcasting a column and a row vector.
    a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
    b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rows[:, None] < M) & (ks[None, :] + k0 < K), other=0.0)
        b = tl.load(b_ptrs, mask=(ks[:, None] + k0 < K) & (cols[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)                    # acc += a @ b on tensor cores
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(rows[:, None] < M) & (cols[None, :] < N))


# The same kernel, benchmarked over CONFIGS the first time each (M, N, K) is seen.
matmul_kernel_tuned = triton.autotune(configs=CONFIGS, key=["M", "N", "K"])(matmul_kernel)


def matmul(a: torch.Tensor, b: torch.Tensor, tuned: bool = None, **config) -> torch.Tensor:
    """tuned=None autotunes on a GPU; otherwise pass BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M."""
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0]
    assert a.device == b.device and a.dtype == b.dtype
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c
    args = (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    tuned = a.is_cuda and not config if tuned is None else tuned
    if tuned:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
        matmul_kernel_tuned[grid](*args)
    else:
        config = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 4, **config}
        grid = (triton.cdiv(M, config["BLOCK_M"]) * triton.cdiv(N, config["BLOCK_N"]),)
        matmul_kernel[grid](*args, **config)
    return c
