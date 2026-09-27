---
title: 交錯合併陣列
platform: LeetGPU
upstream: easy/63_interleave
url: https://leetgpu.com/challenges/interleave-arrays
difficulty: easy
tags: [memory-bound, vectorized, data-movement]
status: solved
---

# 交錯合併陣列

**平台：** LeetGPU · **難度：** 簡單 · [題目說明](https://leetgpu.com/challenges/interleave-arrays)

## 題意

將兩個長度為 $N$ 的 float32 陣列交錯合併成長度為 $2N$ 的陣列：
$[a_0, b_0, a_1, b_1, \dots]$（$N \le 5\times10^7$；基準測試為
$N = 2.5\times10^7$）。這是單純的資料配置轉換：將陣列結構
（structure-of-arrays，SoA）轉為結構陣列（array-of-structures，AoS），
適用於複數、(x, y) 點，或許多 API 所需的 `float2` 配置。

## 圖解

![交錯合併（SoA → AoS）：o[2i] = a[i]，o[2i + 1] = b[i]](figure.svg)

藍色值來自 A，橘色值來自 B。執行緒 i 以一個 float2 寫出 (aᵢ, bᵢ)，因此讀取與寫入都保持連續。

## 數學表述

$$
o_{2i} = a_i, \qquad o_{2i+1} = b_i, \qquad 0 \le i < N
$$

| 符號 | 意義 |
|---|---|
| $N$ | 每個輸入的長度 |
| $a_i,\ b_i$ | 輸入 `A[i]`、`B[i]`（float32） |
| $o_k$ | 輸出，$0 \le k < 2N$ |

## 解題思路

執行緒 $i$ 讀取 $a_i$ 與 $b_i$，並以**單一 `float2`** 一起寫入
`output[2i]`：

- 對整個 warp 而言，$a$ 與 $b$ 的讀取皆為連續存取（兩個 128 位元組區段）；
- 寫入為 32 個連續 `float2` = 256 個連續位元組，每個執行緒進行一次
  完全合併的 8 位元組寫入。

若改用兩次獨立的 4 位元組寫入至 `output[2i]` 與 `output[2i+1]`，
每條寫入指令都會碰觸 256 個位元組，卻只使用其中一半。每條指令會浪費一半寫入頻寬
（L2 最終會合併兩半，但要求數量會加倍）。

## 成本分析

$$
Q = 4N + 4N + 8N = 16N\ \text{bytes}, \qquad T_{\min} = \frac{16N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取 $a$ 與 $b$，寫入 $2N$ 個 float |
| $\beta$ | DRAM 頻寬 |

基準測試為 400 MB，在 2 TB/s 下約需 200 µs。

## 常見陷阱

- **對齊。** 將 `output` 重新解讀為 `float2*` 需要 8 位元組對齊，
  `cudaMalloc` 會提供此對齊。
- **索引寬度。** $2N \le 10^8$ 可放入 `int`。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆完全相等。

## 延伸閱讀

- [矩陣轉置](../003-matrix-transpose/)（配置變更的二維推廣）、[矩陣複製](../031-matrix-copy/)。
