---
title: 數值裁剪
platform: LeetGPU
upstream: easy/62_value_clipping
url: https://leetgpu.com/challenges/value-clipping
difficulty: easy
tags: [elementwise, clamp]
status: solved
---

# 數值裁剪

**平台：** LeetGPU · **難度：** 簡單 · [題目說明](https://leetgpu.com/challenges/value-clipping)

## 問題

將 $N$ 個 float32 值逐一裁剪至 $[\ell, h]$ 範圍內（$N \le 10^5$、
$\ell \le h$；容許誤差 `1e-5`）。裁剪（clamping）常見於梯度／活化值穩定處理、
PPO 比率裁剪，以及量化之前。

## 公式

$$
y_i = \operatorname{clamp}(x_i, \ell, h) = \min\bigl(\max(x_i, \ell),\ h\bigr) =
\begin{cases} \ell, & x_i < \ell \\ x_i, & \ell \le x_i \le h \\ h, & x_i > h \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $N$ | 元素數量 |
| $x_i,\ y_i$ | 輸入值與輸出值 |
| $\ell,\ h$ | 下界與上界（`lo`、`hi`） |

## 方法

每個元素使用一個執行緒：`fminf(fmaxf(x, lo), hi)`。這是兩條
`FMNMX` 指令，沒有分支，也不會發生分歧。

## 成本分析

$$
Q = 8N\ \text{bytes}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（讀取並寫入每個元素） |
| $\beta$ | DRAM 頻寬 |

當 $N = 10^5$（800 KB）時，核心受啟動成本限制（數 µs）。

## 注意事項

- **NaN 輸入。** `fmaxf(NaN, lo) = lo`，但 `torch.clamp(NaN)` 會回傳
  NaN。測試不包含 NaN。
- **min/max 的順序**只會在 $\ell > h$ 時造成影響，但限制條件已排除此情形。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆通過，
包括 $\ell = h$。

## 相關內容

- [ReLU](../021-relu/)（從下方以 0 裁剪）、[PPO 裁剪損失](../107-ppo-clipped-surrogate-loss/)。
- Tensara [Hard Sigmoid](../../tensara/hard-sigmoid/)、[Threshold](../../tensara/threshold/)。
