---
title: Simple Inference
platform: LeetGPU
upstream: easy/41_simple_inference
url: https://leetgpu.com/challenges/simple-inference
difficulty: easy
tags: [pytorch, linear-layer]
status: solved
---

# Simple Inference

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/simple-inference)

## Problem
Run the forward pass of a given `nn.Linear` model (PyTorch-only challenge).

## Approach
`y = x Wᵀ + b` is one GEMM with a fused bias: `torch.addmm(bias, x, W.t(),
out=output)` writes straight into the output tensor without an intermediate,
inside `torch.inference_mode()` to skip autograd bookkeeping.

## Pitfalls
- The model may be created with `bias=False` — then `model.bias is None`.
