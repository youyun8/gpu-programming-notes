---
title: RGB to Grayscale
platform: LeetGPU
upstream: easy/66_rgb_to_grayscale
url: https://leetgpu.com/challenges/rgb-to-grayscale
difficulty: easy
tags: [image, elementwise]
status: solved
---

# RGB to Grayscale

**Platform:** LeetGPU · **Difficulty:** easy · [Problem statement](https://leetgpu.com/challenges/rgb-to-grayscale)

## Problem
Convert interleaved RGB float32 to grayscale with `0.299 R + 0.587 G + 0.114 B`.

## Approach
One thread per pixel. The three loads per thread are 12-byte strided, but a
warp's loads cover one contiguous 384-byte range, so each cache line is
fetched once.
