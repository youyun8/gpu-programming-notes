---
title: Leaky ReLU
platform: LeetGPU
upstream: easy/23_leaky_relu
url: https://leetgpu.com/challenges/leaky-relu
difficulty: easy
tags: [elementwise, activation, vectorized, memory-bound]
status: solved
---

# Leaky ReLU

**平台：** LeetGPU · **難度：** easy · [題目敘述](https://leetgpu.com/challenges/leaky-relu)

## 題意

對 $N$ 個 float32 值套用斜率 $\alpha = 0.01$ 的 Leaky ReLU
（$1 \le N \le 10^8$、$\lvert x_i\rvert \le 1000$；
基準測試為 $N = 5\times10^7$；容許誤差為 `1e-6`）。
它與 ReLU 不同，會為負輸入保留一個小梯度，以避免訓練時出現
「死亡」單元。

## 圖解

![Leaky ReLU：負輸入保留很小的斜率 α = 0.01](figure.svg)

x 軸延伸到 −40，才看得出平緩的負斜率：x = −40 時輸出為 −0.4（紅點）。虛線是一般 ReLU，供比較。

## 數學表述

$$
y_i = \operatorname{LeakyReLU}(x_i) =
\begin{cases} x_i, & x_i > 0 \\ \alpha\, x_i, & x_i \le 0 \end{cases}
\qquad \alpha = 0.01
$$

| 符號 | 意義 |
|---|---|
| $N$ | 元素數 |
| $x_i,\ y_i$ | 輸入與輸出值（float32） |
| $\alpha$ | 負值側的斜率 |

當 $0 < \alpha < 1$ 時，也可寫成 $y = \max(x, \alpha x)$。

## 解題思路

使用與 [ReLU](../021-relu/) 相同的向量化範本：每個執行緒處理一個
`float4`，再加上純量尾端。三元運算式 `x > 0 ? x : 0.01f * x`
會編譯為一次乘法與一次選擇（`FSEL`），因此沒有分歧分支。

乘積 `0.01f * x` 是單一 float32 乘法。它的捨入方式與 PyTorch
以 float32 計算 `alpha * input` 完全相同；在嚴格的 `1e-6`
容許誤差下，這點很重要。

## 成本分析

$$
Q = 8N \ \text{bytes}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 傳輸量（讀取輸入、寫入輸出） |
| $\beta$ | DRAM 頻寬 |

基準測試：$Q = 400$ MB，因此在 2 TB/s 下
$T_{\min} \approx 200\ \mu s$。

## 常見陷阱

- **雙精度常數。** 寫成 `0.01 * x`（double 常值）會提升為 float64，
  在消費級 GPU 上速度很慢，而且捨入結果可能不同。應使用 `0.01f`。
- **邊界。** $x = 0$ 屬於 $\alpha x$ 分支，但兩個分支都會得到 0。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
以 `1e-6` 容許誤差通過。

## 延伸閱讀

- [ReLU](../021-relu/)、Tensara [Leaky ReLU](../../tensara/leaky-relu/)、
  [ELU](../../tensara/elu/)。
