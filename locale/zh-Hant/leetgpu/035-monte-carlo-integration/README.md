---
title: 蒙地卡羅積分
platform: LeetGPU
upstream: medium/35_monte_carlo_integration
url: https://leetgpu.com/challenges/monte-carlo-integration
difficulty: medium
tags: [reduction, statistics, two-pass]
status: solved
---

# 蒙地卡羅積分

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/monte-carlo-integration)

## 題意

從 $n$ 個預先計算的樣本 $y_i = f(x_i)$ 估算
$\int_a^b f(x)\,dx$，其中 $x_i$ 均勻分布於 $[a, b]$
（$1 \le n \le 10^8$、$-1000 \le a < b \le 1000$、
$\lvert y_i \rvert \le 10^4$；效能評測使用 $n = 10^7$；容許誤差為 `1e-2`）。
隨機取樣已經完成，因此 GPU 的工作就是計算平均值。本頁也會說明這個估計量
為何有效，以及它的準確度。

## 圖解

![蒙地卡羅積分：(b − a) 乘上取樣函數值的平均](figure.svg)

陰影面積就是積分值。橘色點是題目給定的樣本 yᵢ = f(xᵢ)，其平均為綠線；估計值就是以該高度、寬為 [a, b] 的長方形面積。

## 數學表述

$$
I = \int_a^b f(x)\,dx \;\approx\; \hat I_n = (b - a)\cdot\frac{1}{n}\sum_{i=0}^{n-1} y_i, \qquad y_i = f(x_i),\ x_i \sim \mathcal U[a, b]
$$

| 符號 | 意義 |
|---|---|
| $a,\ b$ | 積分上下限 |
| $f$ | 被積函數（只提供其取樣值） |
| $x_i$ | $[a, b]$ 上獨立的均勻取樣點 |
| $y_i$ | 函數值（float32 輸入 `y_samples`） |
| $n$ | 樣本數 |
| $I$ | 精確積分值 |
| $\hat I_n$ | 蒙地卡羅估計值，寫入 `result[0]` |

因為均勻分布的 $x$ 滿足 $\mathbb E[f(x)] = I/(b-a)$，所以這個估計量
沒有偏差；依據中央極限定理，其誤差會以 $1/\sqrt n$ 的速率縮小：

$$
\mathbb E[\hat I_n] = I, \qquad \operatorname{Std}[\hat I_n] = \frac{(b-a)\,\sigma_f}{\sqrt n}, \qquad \sigma_f^2 = \operatorname{Var}_{x\sim\mathcal U[a,b]}[f(x)]
$$

| 符號 | 意義 |
|---|---|
| $\mathbb E,\ \operatorname{Std},\ \operatorname{Var}$ | 隨機樣本的期望值、標準差與變異數 |
| $\sigma_f$ | 均勻分布下 $f$ 的標準差 |

GPU 只需準確計算**樣本平均值**。它本身的捨入誤差必須遠低於統計誤差；
使用 float64 部分總和即可輕鬆達成。

## 解題思路

採用[歸約](../004-reduction/)中的兩階段歸約：

1. **`partialSums`**：以網格跨步方式載入 `float4`，每個執行緒以
   float32 累加，再以 float64 區塊歸約，讓每個區塊產生一個部分總和
   （最多 1024 個區塊）。
2. **`finalize`**：以一個區塊用 float64 加總所有部分總和，並寫入
   $(b - a)\cdot\text{sum}/n$；$b - a$ 以 double 計算。

## 成本分析

$$
Q = 4n \ \text{bytes}, \qquad W = n, \qquad T_{\min} = \frac{4n}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（每個樣本讀取一次） |
| $W$ | 加法次數 |
| $\beta$ | DRAM 頻寬 |

效能評測資料為 40 MB，因此在 2 TB/s 下約需 20 µs。

## 常見陷阱

- **以 float32 計算 $(b-a)$。** 當 $a, b$ 最大為 $\pm1000$ 時已足夠精確，
  但改用 double 並不會增加成本。
- **過早除法。** 在加總前先將每個樣本除以 $n$，會在極小數值上浪費精度。
  應在最後只除一次。

## 驗證

所有 LeetGPU 測試案例均以 `1e-2` 的容許誤差在
[cuemu](../../tools/cuemu/README.md) 通過。

## 延伸閱讀

- [歸約](../004-reduction/)、[均方誤差](../027-mean-squared-error/)。
