---
title: 均方誤差
platform: LeetGPU
upstream: medium/27_mean_squared_error
url: https://leetgpu.com/challenges/mean-squared-error
difficulty: medium
tags: [reduction, two-pass, loss]
status: solved
---

# 均方誤差

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/mean-squared-error)

## 題意

計算兩個長度為 $N$ 的 float32 陣列之均方誤差
（$1 \le N \le 10^8$，值位於 $[-1000, 1000]$；
基準測試為 $N = 5\times10^7$；容許誤差為 `1e-5`），
並寫入 `mse[0]`。這是一種會在輸入時進行轉換的歸約：
先將差值平方。

## 圖解

![均方誤差：先平方每個差值，再歸約並除以 N](figure.svg)

每個執行緒先計算差值的平方（第二排），再由樹狀歸約加總，最後一步除以 N。圖中數字是八組範例的實際值。

## 數學表述

$$
\operatorname{MSE} = \frac{1}{N}\sum_{i=0}^{N-1} \bigl(p_i - t_i\bigr)^2
$$

| 符號 | 意義 |
|---|---|
| $N$ | 元素數 |
| $p_i$ | 預測值（float32） |
| $t_i$ | 目標值（float32） |
| $\operatorname{MSE}$ | 以 float32 儲存的結果 |

每個平方項最大可達 $(2000)^2 = 4\times10^6$。$10^8$ 個這類項目的總和
可達 $4\times10^{14}$，遠超出 float32 的 24 位元尾數所能精確累加的
範圍。只有每個執行緒的部分結果使用 float32 累加
（每個執行緒只加總數百項）；跨執行緒的層級則使用 float64。

## 解題思路

使用與[歸約](../004-reduction/)相同的兩階段結構：

1. **`partialSquares`**：以網格跨步方式走訪成對的 `float4`，
   每次迭代為 4 個 lane 累加 $(a-b)^2$（成對相加以縮短相依鏈），
   再處理純量尾端。區塊內以 float64 歸約，每個區塊產生一個部分結果。
2. **`finalMean`**：一個區塊以 float64 加總各部分結果，除以 $N$，
   再捨入為 float32。

## 成本分析

$$
Q = 8N\ \text{bytes}, \qquad W = 3N, \qquad I = \frac{3}{8}\ \text{FLOP/byte}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（兩個陣列各讀取一次） |
| $W$ | FLOP 數：每個元素一次減法、乘法與加法 |
| $I$ | 算術強度 |
| $\beta$ | DRAM 頻寬 |

基準測試為 400 MB，因此在 2 TB/s 下約需 200 µs。
此核心函式受限於記憶體。

## 常見陷阱

- **最後以 float32 執行除法。** 這沒有問題，但在上層使用 float32
  加總則不行（原因如上）。
- **當 $N < 4$ 時網格有 0 個區塊**：必須限制下限為 1。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
以 `1e-5` 容許誤差通過。

## 延伸閱讀

- [歸約](../004-reduction/)、[點積](../017-dot-product/)。
- Tensara [MSE 損失](../../tensara/mse-loss/)、
  [Huber 損失](../../tensara/huber-loss/)。
