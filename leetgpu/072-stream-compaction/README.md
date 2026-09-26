---
title: Stream Compaction
platform: LeetGPU
upstream: medium/72_stream_compaction
url: https://leetgpu.com/challenges/stream-compaction
difficulty: medium
tags: [scan, compaction]
status: solved
---

# Stream Compaction

**Platform:** LeetGPU · **Difficulty:** medium · [Problem statement](https://leetgpu.com/challenges/stream-compaction)

## Problem
Stable compaction of the positive elements to the front; zero-fill the rest.

## Approach
Compaction = exclusive scan of the 0/1 predicate + scatter, in the reduce-then-
scan structure with 2048-element chunks: per-chunk counts, a single-block scan
of the counts (which also gives the total), then per-chunk block scans that
assign output slots and scatter. A last kernel zero-fills `[total, N)`.
