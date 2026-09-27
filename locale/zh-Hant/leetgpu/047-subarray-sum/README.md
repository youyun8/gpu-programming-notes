---
title: 子陣列總和
platform: LeetGPU
upstream: medium/47_subarray_sum
url: https://leetgpu.com/challenges/subarray-sum
difficulty: medium
tags: [reduction, integer, warp-intrinsics]
status: solved
---

# 子陣列總和

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/subarray-sum)

## 題意

對長度為 $N$ 的陣列，加總 int32 值 `input[S..E]`（含兩端）
（$N \le 10^8$，值域為 $[1, 10]$；效能評測使用 $N = 10^8$）。
結果是精確的 int32。整數加法精確且具有結合律，因此歸約設計可以比浮點數
更簡單。

## 圖解

![子陣列和：以精確的整數歸約加總 x[S] … x[E]](figure.svg)

只會讀取標示的範圍。整數加法是精確的，因此執行緒可以用任何順序（包括原子操作）合併部分和。

## 數學表述

$$
\text{out} = \sum_{i=S}^{E} x_i
$$

| 符號 | 意義 |
|---|---|
| $N$ | 陣列長度 |
| $x_i$ | 輸入值（int32） |
| $S,\ E$ | 以 0 為起點且包含兩端的起始與結束索引，$0 \le S \le E < N$ |
| out | 精確總和，寫入 `output[0]` |

最大可能值為 $10 \cdot 10^8 = 10^9 < 2^{31}$，因此 int32 不會溢位。

## 解題思路

- 執行 `cudaMemset(output, 0)`。
- 啟動 $\min(\lceil (E-S+1)/256\rceil, 2048)$ 個區塊。執行緒 $g$
  會加總 $x_{S+g}, x_{S+g+P}, \dots$（網格跨步，其中 $P$ 是執行緒
  總數）；連續的執行緒會讀取連續位址。
- `__reduce_add_sync` 以一個指令完成 warp 歸約，再由第 0 個執行緒
  執行一次 `atomicAdd`。

因為整數加法完全精確，原子操作的不確定順序不會影響結果。與浮點歸約不同，
本題不需要第二輪。

## 成本分析

$$
Q = 4(E - S + 1)\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（只讀取要求的範圍） |
| $\beta$ | DRAM 頻寬 |

## 常見陷阱

- **未對齊的起點。** `input + S` 不一定以 16 位元組對齊，因此載入
  `int4` 需要純量前序處理。純量網格跨步迴圈更簡單，而且仍可合併存取。
- **空網格**不可能出現，因為 $E \ge S$ 保證至少有一個元素。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 中完全相等，
包括 $S = E$。

## 延伸閱讀

- [2D 子陣列總和](../048-2d-subarray-sum/)、
  [3D 子陣列總和](../049-3d-subarray-sum/)、
  [計算陣列元素數量](../043-count-array-element/)。
