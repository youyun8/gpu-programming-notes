---
title: 2D 子陣列總和
platform: LeetGPU
upstream: medium/48_2d_subarray_sum
url: https://leetgpu.com/challenges/2d-subarray-sum
difficulty: medium
tags: [reduction, integer, warp-intrinsics]
status: solved
---

# 2D 子陣列總和

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/2d-subarray-sum)

## 問題

加總 $N \times M$ int32 矩陣中的矩形
`input[S_ROW..E_ROW][S_COL..E_COL]`（含邊界）
（$N, M \le 10^4$，值域為 $[1, 10]$；效能評測使用
$N = M = 10^4$）。結果是精確的 int32。

## 公式

$$
\text{out} = \sum_{r = r_0}^{r_1}\ \sum_{c = c_0}^{c_1} x_{r c}, \qquad
\text{flattened: } i \mapsto (r, c) = \Bigl(r_0 + \bigl\lfloor i / w \bigr\rfloor,\ c_0 + (i \bmod w)\Bigr),\ 0 \le i < h w
$$

| 符號 | 意義 |
|---|---|
| $N,\ M$ | 矩陣列數與欄數 |
| $x_{rc}$ | 偏移 $rM + c$ 處的元素 |
| $r_0, r_1$ | `S_ROW`、`E_ROW` |
| $c_0, c_1$ | `S_COL`、`E_COL` |
| $h,\ w$ | 矩形高度 $r_1 - r_0 + 1$ 與寬度 $c_1 - c_0 + 1$ |
| $i$ | 矩形內的平坦索引，欄變化最快 |

## 方法

將矩形攤平成包含 $h w$ 個元素的一維索引空間，且欄索引變化最快。
除了每 $w$ 個元素會跨到下一列外，連續執行緒會讀取同一列中的連續元素，
因此可合併存取。其餘部分採用[子陣列總和](../047-subarray-sum/)的核心函式：
以網格跨步方式累加、使用 `__reduce_add_sync`，並由每個 warp 執行一次
`atomicAdd`。平坦索引使用 64 位元（$h w \le 10^8$ 可放入 32 位元，
但乘積仍以安全的方式計算）。

## 成本分析

$$
Q = 4hw\ \text{bytes (useful)}, \qquad \text{sectors touched} \approx h\left\lceil\frac{4w}{32}\right\rceil + h
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 實際需要的位元組數 |
| 區段 | 讀取的 32 位元組 DRAM 區段；每個列片段的兩端可能各跨越一個額外區段 |

對窄矩形（$w$ 很小）而言，每列的額外成本會占主要部分。對全寬效能評測而言，
效率幾乎是 100%。

## 常見陷阱

- **2D 執行緒映射**（每個執行緒負責一列並走訪各欄）會讓連續執行緒
  讀取相隔 $M$ 的位址，無法合併存取。
- **每個元素都進行整數除法**（`i / w`、`i % w`）並非免費，
  但成本會隱藏在記憶體延遲之後。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 中完全相等，
包括 1 × 1 與涵蓋完整矩陣的矩形。

## 相關內容

- [子陣列總和](../047-subarray-sum/)、
  [3D 子陣列總和](../049-3d-subarray-sum/)。
