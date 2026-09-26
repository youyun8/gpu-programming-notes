# 01 – CUDA Execution Model

## The hierarchy

```
Grid  ── many Blocks  (scheduled independently onto SMs, any order)
Block ── up to 1024 Threads (share shared memory, can __syncthreads())
Warp  ── 32 consecutive threads of a block, execute in lock-step (SIMT)
```

- A **kernel** is a function run by every thread of a grid.
- `blockIdx`, `blockDim`, `threadIdx`, `gridDim` are built-in `dim3` variables.
- Blocks cannot synchronize with each other inside a kernel (short of cooperative
  launch); if you need a global barrier, end the kernel and launch another.

## Global index

```cpp
const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 1D
const int row = blockIdx.y * blockDim.y + threadIdx.y;   // 2D
const int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Launch enough blocks to cover the problem: `num_blocks = (n + block_size - 1) / block_size`,
and always guard `if (idx < n)`.

## Grid-stride loops

Decouple grid size from problem size — useful when `n` is huge or you want a
fixed number of blocks (e.g. a multiple of the SM count):

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = f(in[i]);
}
```

## Choosing a block size

- A multiple of 32 (warp size). 128–512 is typical; 256 is a safe default.
- Bigger blocks are not automatically faster: occupancy is limited by registers
  and shared memory per SM. `cudaOccupancyMaxPotentialBlockSize` can suggest one.

## Warp divergence

Threads of one warp taking different branches are serialized. Branch on
something uniform across the warp (e.g. `warp_id`) when possible.

## Practice

- [LeetGPU – Vector Addition](../leetgpu/001-vector-addition)
- [Tensara – Vector Addition](../tensara/vector-addition)
