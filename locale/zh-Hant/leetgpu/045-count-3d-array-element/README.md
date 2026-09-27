---
title: 計算 3D 陣列元素數量
platform: LeetGPU
upstream: medium/45_count_3d_array_element
url: https://leetgpu.com/challenges/count-3d-array-element
difficulty: medium
tags: [reduction, counting, warp-intrinsics]
status: solved
---

# 計算 3D 陣列元素數量

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/count-3d-array-element)

## 題意

計算一個 $N \times M \times K$ int32 張量中 $P$ 的出現次數
（$1 \le N, M, K \le 1000$；效能評測使用 $500^3$）。如同 2D 版本，
張量在記憶體中連續儲存，因此問題可以化為對 $NMK \le 10^9$ 個元素
進行平坦計數。只有換算成**位元組數**時才可能超過 $2^{31}$，
因此偏移使用 64 位元。

## 圖解

![三維陣列計數：仍然是一次攤平、連續的計數](figure.svg)

圖中畫出張量的兩個切片；它們在記憶體中前後相接，所以整個張量仍是一個攤平的陣列。唯一要注意的是位元組偏移量需要 64 位元運算。

## 數學表述

$$
\text{count} = \sum_{a=0}^{N-1}\sum_{b=0}^{M-1}\sum_{c=0}^{K-1}\bigl[\,x_{abc} = P\,\bigr], \qquad \text{offset}(a, b, c) = (aM + b)K + c
$$

| 符號 | 意義 |
|---|---|
| $N,\ M,\ K$ | 張量維度 |
| $x_{abc}$ | 元素（int32） |
| $P$ | 要計數的值 |
| 偏移 | 平坦的列優先索引 |
| 計數 | `output[0]` 中的精確結果 |

## 解題思路

採用與[計算 2D 陣列元素數量](../044-count-2d-array-element/)相同的
核心函式：64 位元元素計數、以網格跨步方式載入 `int4`、
`__reduce_add_sync`，以及每個 warp 執行一次 `atomicAdd`。

## 成本分析

$$
Q = 4NMK \ \text{bytes}, \qquad T_{\min} = \frac{4NMK}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| $\beta$ | DRAM 頻寬 |

效能評測資料為 500 MB，也就是在 2 TB/s 下約需 250 µs。在最大大小
（$10^9$ 個元素 = 4 GB）下，必須使用 64 位元索引。

## 常見陷阱

- **32 位元迴圈計數器**會在 $NMK > 2^{31}/4$ 個向量時溢位，
  也就是索引位元組或使用大型步幅時。
- 累加前應先**將輸出歸零**。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 中完全相等。

## 延伸閱讀

- [計算陣列元素數量](../043-count-array-element/)、
  [計算 2D 陣列元素數量](../044-count-2d-array-element/)。
