# GPU Programming Notes

Personal notes on GPU programming (CUDA first, Triton later) plus solutions to
[LeetGPU challenges](https://leetgpu.com/challenges) and
[Tensara problems](https://tensara.org/problems).

## Layout

```
tutorials/          concept notes, read in order (00, 01, ...)
leetgpu/NNN-slug/   README.md (write-up) + solution.cu
tensara/slug/       README.md (write-up) + solution.cu
templates/problem/  template used by scripts/new_problem.py
scripts/            scaffolding + index generation
```

## Workflow

```bash
# 1. Scaffold a problem
python3 scripts/new_problem.py leetgpu "Matrix Transpose" --difficulty easy

# 2. Paste the starter code into solution.cu, solve, submit, record runtime in README.md
#    and set `status: solved` in its front matter.

# 3. Refresh the tables below
python3 scripts/build_index.py

# 4. (optional, needs nvcc) compile-check every solution
make check
```

CI compiles every `solution.cu` with `nvcc` inside an NVIDIA CUDA container
(no GPU needed) and verifies that the tables below are up to date.

## Tutorials

See [tutorials/](tutorials/README.md).

## LeetGPU

<!-- BEGIN LEETGPU INDEX -->
| Problem | Difficulty | Tags | Status | Source |
|---|---|---|---|---|
| [Vector Addition](leetgpu/001-vector-addition) | easy | elementwise | ✅ | [link](https://leetgpu.com/challenges/vector-addition) |
| [Matrix Multiplication](leetgpu/002-matrix-multiplication) | easy | gemm, shared-memory, tiling | ✅ | [link](https://leetgpu.com/challenges/matrix-multiplication) |
<!-- END LEETGPU INDEX -->

## Tensara

<!-- BEGIN TENSARA INDEX -->
| Problem | Difficulty | Tags | Status | Source |
|---|---|---|---|---|
| [Vector Addition](tensara/vector-addition) | easy | elementwise, vectorized | ✅ | [link](https://tensara.org/problems/vector-addition) |
<!-- END TENSARA INDEX -->

## Code conventions

- Classes / structs: `PascalCase`
- Functions, kernels, lambdas: `camelCase`
- `constexpr` / `const` globals: `kCamelCase`
- Other variables, parameters, namespaces: `snake_case`
- Platform entry points (`solve`, `solution`) keep the exact name from the starter code.
- Comments in English.
