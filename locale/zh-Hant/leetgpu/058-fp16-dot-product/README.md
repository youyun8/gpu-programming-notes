---
title: FP16 點積
platform: LeetGPU
upstream: medium/58_fp16_dot_product
url: https://leetgpu.com/challenges/fp16-dot-product
difficulty: medium
tags: [reduction, fp16, mixed-precision]
status: solved
---

# FP16 點積

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/fp16-dot-product)

## 題意

計算兩個長度為 $N$ 的 fp16 向量之點積，以 fp32 累加並回傳 fp16
（$N \le 10^8$；基準測試為 $N = 10^8$；容許誤差 `0.05`）。
只要維持 fp32 累加，相較於 float32 將每個元素的位元組數減半，
就能讓受頻寬限制的歸約執行時間減半。

## 圖解

![FP16 內積：位元組數只有 fp32 的一半，但以 fp32 累加](figure.svg)

輸入是 fp16（第一排），但每個乘積與所有部分和都以 fp32 保存，只有最終結果才捨入為 fp16。

## 數學表述

$$
s = \operatorname{fp16}\!\Bigl(\sum_{i=0}^{N-1}\operatorname{fp32}(a_i)\,\operatorname{fp32}(b_i)\Bigr)
$$

| 符號 | 意義 |
|---|---|
| $N$ | 向量長度 |
| $a_i,\ b_i$ | fp16 輸入（10 位元尾數，最大值 65 504） |
| $s$ | 最後一次才四捨五入為 fp16 的結果 |

**為什麼不用 fp16 累加？** fp16 只有 11 個有效位元。執行中的總和一旦達到約
2048，加上小於 1 的值就完全不會產生影響（被吞沒），總和也不再增加。
因此參考實作會先擴寬至 fp32，正確的核心也必須如此。

## 解題思路

1. **`partialDots`**（≤ 1024 個區塊）：以網格步幅迴圈走訪 `half2` 配對
   （一次 32 位元載入可取得兩個 fp16 值）。`__half22float2` 會擴寬兩者，
   並以兩次 `fmaf` 累加至 fp32。執行緒 0 處理奇數長度時最後一個元素。
   區塊歸約以 float64 執行，結果寫入 `g_partials`。
2. **`finalSum`**：一個區塊以 float64 加總部分結果，並轉為 fp16
   （`__float2half`，四捨五入至最近值）。

## 成本分析

$$
Q = 2 \cdot 2N = 4N\ \text{bytes}, \qquad T_{\min} = \frac{4N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（兩個 fp16 向量） |
| $\beta$ | DRAM 頻寬 |

基準測試為 400 MB，在 2 TB/s 下約需 200 µs。相同 $N$ 時，
這是 float32 [點積](../017-dot-product/)執行時間的一半。

## 常見陷阱

- **`half2` 對齊**：需要 4 位元組對齊。`cudaMalloc` 回傳的基底指標符合要求。
- **奇數 $N$**：必須恰好由一個執行緒加上最後一個元素。
- **結果的 fp16 溢位。** 總和超過 65 504 時會成為 $\infty$，與參考實作相同。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆以
`0.05` 通過，包括奇數 $N$。

## 延伸閱讀

- [點積](../017-dot-product/)、[GEMM（fp16）](../022-gemm/)。
