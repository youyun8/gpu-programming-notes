---
title: 計算陣列元素數量
platform: LeetGPU
upstream: medium/43_count_array_element
url: https://leetgpu.com/challenges/count-array-element
difficulty: medium
tags: [reduction, counting, warp-intrinsics, atomics]
status: solved
---

# 計算陣列元素數量

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/count-array-element)

## 問題

計算 $N$ 個 int32 值中有多少個等於 $K$
（$1 \le N \le 10^8$；效能評測使用 $N = 10^8$）。結果是精確的 int32。
這是一項映射步驟為比較的歸約，也展示了**單指令 warp 歸約**
`__reduce_add_sync`（sm_80+）。

## 公式

$$
\text{count} = \sum_{i=0}^{N-1} \bigl[\,x_i = K\,\bigr]
$$

| 符號 | 意義 |
|---|---|
| $N$ | 元素數量 |
| $x_i$ | 輸入值（int32） |
| $K$ | 要計數的值 |
| $[\cdot]$ | Iverson 括號（條件為真時為 1，否則為 0） |
| 計數 | 結果，寫入 `output[0]` |

整數加法具有結合律，而且**完全精確**，因此任何歸約順序都會得到相同答案。
所以此處可以安全使用原子操作，不像浮點總和。

## 方法

1. 執行 `cudaMemset(output, 0)`。
2. 執行 `countEqual`（最多 2048 個區塊 × 256 個執行緒）：
   - 以網格跨步迴圈載入 `int4` 向量，並加總四次比較
     `(v.x == K) + … `。`bool` 不需分支即可轉為 0/1。純量迴圈則處理尾端。
   - `__reduce_add_sync(0xffffffff, count)` 以**一個指令**
     （`REDUX.SUM`）加總 32 個執行緒的計數，不需 5 個 shuffle 加法步驟。
   - 若計數不為零，每個 warp 的第 0 個執行緒執行一次
     `atomicAdd(output, count)`。

最多只有 $2048 \cdot 8 = 16\,384$ 次全域原子操作會存取同一個位址。
相較於讀取 400 MB，這項成本可忽略不計。

## 成本分析

$$
Q = 4N \ \text{bytes}, \qquad T_{\min} = \frac{4N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（每個元素讀取一次） |
| $\beta$ | DRAM 頻寬 |

效能評測資料為 400 MB，也就是在 2 TB/s 下約需 200 µs。

## 常見陷阱

- **忘記將 `output` 歸零。** 如此會從垃圾值開始累加。
- **`__reduce_add_sync` 需要 sm_80。** 在較舊的 GPU 上，請改用 shuffle
  迴圈。（[cuemu](../../tools/cuemu/README.md) 模擬器已實作此功能。）
- **效能評測的 $K$（501 010）超出值域**，因此正確答案為 0。
  核心函式會自然處理此情況。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 中完全相等。

## 相關內容

- [計算 2D 陣列元素數量](../044-count-2d-array-element/)、
  [計算 3D 陣列元素數量](../045-count-3d-array-element/)、
  [直方圖](../013-histogramming/)、[歸約](../004-reduction/)。
