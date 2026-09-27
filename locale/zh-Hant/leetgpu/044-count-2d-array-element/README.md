---
title: 計算 2D 陣列元素數量
platform: LeetGPU
upstream: medium/44_count_2d_array_element
url: https://leetgpu.com/challenges/count-2d-array-element
difficulty: medium
tags: [reduction, counting, warp-intrinsics]
status: solved
---

# 計算 2D 陣列元素數量

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/count-2d-array-element)

## 題意

計算一個 $N \times M$ int32 矩陣中 $K$ 的出現次數
（$1 \le N, M \le 10^4$，值域為 $[1, 100]$；
效能評測使用 $N = M = 10^4$、$K = 1$）。矩陣在記憶體中連續儲存，
因此 2D 形狀只會決定元素總數。

## 圖解

![二維陣列計數：矩陣是連續存放的，所以直接在攤平的陣列上計數](figure.svg)

綠色格等於 K = 1。逐列讀取時，矩陣就是右側攤平的陣列，因此一維計數的 kernel 可以原封不動沿用。

## 數學表述

$$
\text{count} = \sum_{r=0}^{N-1}\sum_{c=0}^{M-1} \bigl[\,x_{rc} = K\,\bigr] = \sum_{i=0}^{NM-1} \bigl[\,x_i = K\,\bigr], \qquad i = rM + c
$$

| 符號 | 意義 |
|---|---|
| $N,\ M$ | 列數與欄數 |
| $x_{rc}$ | 第 $r$ 列、第 $c$ 欄的元素（int32） |
| $i$ | 攤平後的列優先索引 |
| $K$ | 要計數的值 |
| 計數 | `output[0]` 中的精確結果 |

## 解題思路

本題使用[計算陣列元素數量](../043-count-array-element/)的平坦計數核心函式
（載入 `int4`、使用 `__reduce_add_sync`，每個 warp 執行一次原子操作），
並以 **64 位元** `long long` 計算元素數量。雖然此處 $N M$ 最大只有
$10^8$，以 64 位元計算仍能讓核心函式安全支援更大的形狀。迴圈索引也使用
64 位元。

## 成本分析

$$
Q = 4NM \ \text{bytes}, \qquad T_{\min} = \frac{4NM}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| $\beta$ | DRAM 頻寬 |

效能評測資料為 400 MB，也就是在 2 TB/s 下約需 200 µs。當值均勻分布於
$[1, 100]$ 時，約有 1% 的元素符合條件。每個 warp 的原子操作很稀疏，
成本幾乎為零。

## 常見陷阱

- **啟動 2D 網格**並以 `x[r][c]` 索引雖然可行，卻會增加索引運算，
  也讓向量化更複雜。
- 對更大的變形題而言，**`int` 計算 $N \cdot M$ 可能溢位**：
  請使用 64 位元。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 中完全相等。

## 延伸閱讀

- [計算陣列元素數量](../043-count-array-element/)、
  [計算 3D 陣列元素數量](../045-count-3d-array-element/)。
