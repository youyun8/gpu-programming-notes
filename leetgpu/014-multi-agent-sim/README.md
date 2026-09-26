---
title: Multi-Agent Simulation
platform: LeetGPU
upstream: hard/14_multi_agent_sim
url: https://leetgpu.com/challenges/multi-agent-simulation
difficulty: hard
tags: [n-body, simulation, shared-memory]
status: solved
---

# Multi-Agent Simulation

**Platform:** LeetGPU · **Difficulty:** hard · [Problem statement](https://leetgpu.com/challenges/multi-agent-simulation)

## Problem
One boids alignment step: each agent moves toward the average velocity of the
neighbours within radius 5.

## Approach
Brute-force `O(N²)` with shared-memory tiles of 256 agents loaded as `float4`
`[x, y, vx, vy]`. The neighbour test (`dx² + dy² < 25`) uses non-fused
`__fmul_rn`/`__fadd_rn`, so agents exactly on the radius are classified the
same way as by the reference.

## Pitfalls
- For much larger `N`, a uniform grid (cell size = radius) with a
  sort-by-cell pass makes the neighbour search `O(N)`.
