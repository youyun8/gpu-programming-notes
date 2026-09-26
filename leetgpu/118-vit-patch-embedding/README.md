---
title: Vision Transformer Patch Embedding
platform: LeetGPU
upstream: medium/118_vit_patch_embedding
url: https://leetgpu.com/challenges/vision-transformer-patch-embedding
difficulty: medium
tags: [gemm, im2col, vision]
status: solved
---

# Vision Transformer Patch Embedding

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/vision-transformer-patch-embedding)

## Problem
ViT stem: `P×P` patches → linear projection + bias, prepend CLS, add positional embeddings.

## Approach
An NT-GEMM with an **implicit im2col** A-tile loader: patch pixels are
gathered directly from the NCHW image, so the patch matrix is never
materialized. Bias and positional embeddings are fused into the epilogue, and
a tiny kernel writes the CLS rows.
