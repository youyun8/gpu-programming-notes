---
title: Sigmoid 活化函數
platform: LeetGPU
upstream: easy/68_sigmoid
url: https://leetgpu.com/challenges/sigmoid-activation
difficulty: easy
tags: [elementwise, activation, vectorized, transcendental]
status: solved
---

# Sigmoid 活化函數

**平台：** LeetGPU · **難度：** 簡單 · [題目說明](https://leetgpu.com/challenges/sigmoid-activation)

## 題意

對 $N$ 個 float32 值套用 logistic sigmoid（$N \le 10^8$，輸入皆為有限值；
基準測試 $N = 5\times10^7$；容許誤差 `1e-5`）。

## 圖解

![邏輯斯 sigmoid 把任何輸入壓縮到 (0, 1)](figure.svg)

曲線通過 (0, 0.5)（紅點），兩端分別趨近 0 與 1。直接套用公式，在 float32 下兩個極端都能得到正確結果。

## 數學表述

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \qquad \sigma(-x) = 1 - \sigma(x), \qquad \sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)
$$

| 符號 | 意義 |
|---|---|
| $x$ | 輸入值 |
| $\sigma(x)$ | 位於 $(0, 1)$ 的輸出 |
| $\sigma'$ | 導數（用於反向傳播及 [Logistic 迴歸](../034-logistic-regression/)） |

**極端值。** 當 $x \to -\infty$ 時，$e^{-x}$ 會溢位成 $+\infty$，
而 $1/\infty = 0$，這正是正確的極限。當 $x \to +\infty$ 時，
$e^{-x} \to 0$，結果恰好為 1。因此，在 float32 中直接使用此公式很安全，
不需要分支。

## 解題思路

使用 `float4` 的逐元素範本（參見 [ReLU](../021-relu/)）：每個執行緒計算
4 個 sigmoid，尾端不足的部分則以純量處理。每個元素需要一次精確的 `expf`
和一次 IEEE 除法。

## 成本分析

$$
Q = 8N\ \text{bytes}, \qquad W \approx N\,(c_{\exp} + c_{\div} + 1)
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| $W$ | 指令數：`expf`（約 10–20）、除法（約 10）及一次加法 |
| $c_{\exp},\ c_{\div}$ | 每次呼叫的指令成本 |

每 8 位元組約需 25 條指令，因此此核心接近平衡點。在記憶體頻寬充裕的
GPU（HBM3）上，它可能轉為計算受限。以 `__frcp_rn` 或 `__fdividef`
取代除法，可在僅犧牲極少精度的情況下減少指令數。

## 常見陷阱

- **`__expf`** 對較大的 $\lvert x\rvert$ 精度較低，不過結果在此仍能通過。
- **計算 $e^{x}/(1+e^{x})$** 在 $x > 88$ 時會溢位成 NaN。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-5`
通過。

## 延伸閱讀

- [SiLU](../052-silu/)、[Logistic 迴歸](../034-logistic-regression/)、Tensara [Sigmoid](../../tensara/sigmoid/)、
  [Hard Sigmoid](../../tensara/hard-sigmoid/)。
