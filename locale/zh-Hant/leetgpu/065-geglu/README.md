---
title: 高斯誤差閘控線性單元
platform: LeetGPU
upstream: easy/65_geglu
url: https://leetgpu.com/challenges/gaussian-error-gated-linear-unit
difficulty: easy
tags: [elementwise, activation, gated, gelu]
status: solved
---

# 高斯誤差閘控線性單元

**平台：** LeetGPU · **難度：** 簡單 · [題目說明](https://leetgpu.com/challenges/gaussian-error-gated-linear-unit)

## 題意

對一維向量套用 GEGLU：將長度為 $N$ 的輸入分成 $\mathbf x_1$ 與
$\mathbf x_2$ 兩半，並輸出
$\mathbf x_1 \odot \operatorname{GELU}(\mathbf x_2)$
（$N \le 10^6$ 且為偶數；值域為 $[-100, 100]$；容許誤差 `1e-4`）。
GEGLU 是 T5 v1.1 與數種擴散 Transformer 的閘控 MLP 活化函數。
請注意，此處活化的是**後半部**，與 [SwiGLU](../054-swiglu/) 不同。

## 圖解

![GEGLU：前半段乘上後半段的 GELU](figure.svg)

這裡後半段（橘）是通過 GELU 的閘門，前半段（藍）是被閘控的值。輸出 i 結合兩半各自的第 i 個元素。

## 數學表述

$$
y_i = x_i \cdot \operatorname{GELU}(x_{i + N/2}), \qquad
\operatorname{GELU}(u) = u\,\Phi(u) = \frac{u}{2}\left(1 + \operatorname{erf}\!\left(\frac{u}{\sqrt2}\right)\right)
$$

| 符號 | 意義 |
|---|---|
| $N$ | 輸入長度（偶數）；輸出長度為 $N/2$ |
| $x_i$ | 前半部（被閘控的值），$0 \le i < N/2$ |
| $x_{i+N/2}$ | 後半部（通過 GELU 的閘控） |
| $\Phi$ | 標準常態累積分布函數 |
| $\operatorname{erf}$ | 誤差函數，$\operatorname{erf}(z) = \frac{2}{\sqrt\pi}\int_0^z e^{-t^2}dt$ |
| $y_i$ | 輸出 |

這是**精確**的 GELU。常見的 tanh 近似
$\tfrac u2\bigl(1 + \tanh(\sqrt{2/\pi}(u + 0.044715u^3))\bigr)$
差異最高約為 $10^{-3}$，超過容許誤差。

## 解題思路

每個輸出使用一個執行緒：載入兩半（兩道合併存取資料流），再計算
`x1 * (0.5f * x2 * (1.0f + erff(x2 * 0.70710678f)))`。CUDA 的 `erff`
誤差最多為 2 ulp。乘以 $1/\sqrt2$ 而非除以 $\sqrt2$ 可省下一次除法，
且差異遠低於容許誤差。

## 成本分析

$$
Q = 6N\ \text{bytes}, \qquad W \approx \tfrac N2\,(c_{\operatorname{erf}} + 5)
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 位元組數：讀取 $N$ 個 float，寫入 $N/2$ 個 |
| $W$ | 每個輸出一次 `erff`（有理函數／多項式近似，約 20 條指令）加上數次乘加 |

當 $N = 10^6$（6 MB）時，核心只需數微秒，且在多數 GPU 上受記憶體頻寬限制。

## 常見陷阱

- **活化哪一半。** 此處的 GEGLU 對*後半部*套用 GELU。交換兩半會得到錯誤答案，
  而且在隨機資料中不易察覺。
- **Tanh 近似**（如上所述）。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆以
`1e-4` 通過，包括 $\pm100$。

## 延伸閱讀

- [SwiGLU](../054-swiglu/)、Tensara [GELU](../../tensara/gelu/)。
