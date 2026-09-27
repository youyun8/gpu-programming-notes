---
title: 3D 子陣列總和
platform: LeetGPU
upstream: medium/49_3d_subarray_sum
url: https://leetgpu.com/challenges/3d-subarray-sum
difficulty: medium
tags: [reduction, integer, warp-intrinsics]
status: solved
---

# 3D 子陣列總和

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/3d-subarray-sum)

## 題意

加總 $N \times M \times K$ int32 體積資料中的立方體
`input[S_DEP..E_DEP][S_ROW..E_ROW][S_COL..E_COL]`
（$N, M, K \le 500$，值域為 $[1, 10]$；效能評測使用 $500^3$）。
結果是精確的 int32。

## 圖解

![三維子陣列和：加總跨多個深度切片的方塊內所有值](figure.svg)

方塊涵蓋深度 1 與 2（藍色格），深度 0 在範圍外。與二維相同，以攤平索引走訪方塊即可保持合併存取。

## 數學表述

$$
\text{out} = \sum_{a = a_0}^{a_1}\ \sum_{b = b_0}^{b_1}\ \sum_{c = c_0}^{c_1} x_{abc}, \qquad \text{offset}(a, b, c) = (aM + b)K + c
$$

$$
i \mapsto \Bigl(a_0 + \bigl\lfloor i / (h w)\bigr\rfloor,\ \ b_0 + \bigl\lfloor i / w\bigr\rfloor \bmod h,\ \ c_0 + (i \bmod w)\Bigr), \qquad 0 \le i < d h w
$$

| 符號 | 意義 |
|---|---|
| $N,\ M,\ K$ | 體積資料的維度（深度、列、欄） |
| $x_{abc}$ | 位於上述列優先偏移的元素 |
| $a_0..a_1,\ b_0..b_1,\ c_0..c_1$ | 包含兩端的深度、列與欄範圍 |
| $d,\ h,\ w$ | 立方體各維長度 $a_1-a_0+1$、$b_1-b_0+1$、$c_1-c_0+1$ |
| $i$ | 立方體內的平坦索引（欄變化最快） |

最大可能總和為 $10 \cdot 500^3 = 1.25\times10^9 < 2^{31}$。

## 解題思路

延伸 [2D 子陣列總和](../048-2d-subarray-sum/)的平坦歸約來處理三個座標。
平坦索引使用 64 位元，欄變化最快（可合併存取），其餘則和前面一樣使用
warp 歸約，並由每個 warp 執行一次原子操作。

## 成本分析

$$
Q = 4dhw \ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 讀取的位元組數 |
| $\beta$ | DRAM 頻寬 |

完整的 $500^3$ 立方體為 500 MB，也就是在 2 TB/s 下約需 250 µs。

## 常見陷阱

- **索引拆解順序**必須符合記憶體配置（深度、列、欄），否則讀取會產生步幅。
- 以 `long long` 計算 $d h w$ 等中間乘積，可避免 **32 位元溢位**。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 中完全相等。

## 延伸閱讀

- [子陣列總和](../047-subarray-sum/)、
  [2D 子陣列總和](../048-2d-subarray-sum/)。
