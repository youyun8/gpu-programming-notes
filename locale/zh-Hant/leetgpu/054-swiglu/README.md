---
title: Swish 閘控線性單元
platform: LeetGPU
upstream: easy/54_swiglu
url: https://leetgpu.com/challenges/swish-gated-linear-unit
difficulty: easy
tags: [elementwise, activation, gated]
status: solved
---

# Swish 閘控線性單元

**平台：** LeetGPU · **難度：** 簡單 · [題目說明](https://leetgpu.com/challenges/swish-gated-linear-unit)

## 題意

對一維向量套用 SwiGLU 閘控：將長度為 $N$ 的 float32 輸入分成
$\mathbf x_1$（前 $N/2$）與 $\mathbf x_2$（後 $N/2$）兩半，
並輸出 $\operatorname{SiLU}(\mathbf x_1)\odot\mathbf x_2$，其長度為 $N/2$
（$N \le 10^5$ 且為偶數；值域為 $[-100, 100]$；`atol = 1e-4`、`rtol = 1e-5`）。
在 LLM 的 MLP 中，$\mathbf x_1$ 與 $\mathbf x_2$ 分別是「gate」與「up」投影。

## 圖解

![SwiGLU 閘控：前半段取 SiLU 後乘上後半段](figure.svg)

前半段（橘）是閘門，後半段（藍）是被閘控的值。輸出 i 由兩半各自的第 i 個元素配對而得，因此輸出長度是輸入的一半。

## 數學表述

$$
y_i = \operatorname{SiLU}(x_i)\cdot x_{i + N/2} = \frac{x_i}{1 + e^{-x_i}}\cdot x_{i + N/2}, \qquad 0 \le i < N/2
$$

| 符號 | 意義 |
|---|---|
| $N$ | 輸入長度（偶數） |
| $x_i$ | 前半部：閘控輸入，$0 \le i < N/2$ |
| $x_{i+N/2}$ | 後半部：被閘控的值 |
| SiLU | $x\,\sigma(x)$；請參閱 [SiLU](../052-silu/) |
| $y_i$ | 輸出，長度為 $N/2$ |

SiLU 平滑且非單調，因此這個「Swish 閘控」比 ReLU 閘控更容易訓練。
它是 LLaMA、Mistral 與 PaLM 使用的活化函數。

## 解題思路

每個輸出 $i$ 使用一個執行緒：
- 載入 $x_i$ 與 $x_{i+N/2}$。對一個 warp 而言，兩者都是連續的
  128 位元組區段，也就是兩道合併存取資料流；
- 計算 `x1 / (1 + expf(-x1)) * x2`。

若 $N = 0$，則略過核心啟動（0 個區塊是無效設定）。

## 成本分析

$$
Q = 4N + 2N = 6N \ \text{bytes}, \qquad W \approx \tfrac{N}{2}\,(c_{\exp} + c_{\div} + 2)
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 位元組數：讀取 $N$ 個 float，寫入 $N/2$ 個 float |
| $W$ | 每個輸出：一次 `expf`、一次除法、一次加法與一次乘法 |
| $c_{\exp},\ c_{\div}$ | 精確指數與除法的指令成本（各約 10–20） |

當 $N = 10^5$ 時，核心受啟動成本限制。在實際 MLP 中，這個閘控會融合至
gate/up GEMM 的結尾階段（請參閱 [SwiGLU MLP 區塊](../084-swiglu-mlp-block/)）。

## 常見陷阱

- **分半與交錯不同。** `chunk(2)` 會切成連續的兩半，並*不是*配對偶數與奇數元素。
- **輸出長度**為 $N/2$。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆通過，
包括 $N = 2$。

## 延伸閱讀

- [SiLU](../052-silu/)、[GEGLU](../065-geglu/)、[SwiGLU MLP 區塊](../084-swiglu-mlp-block/)。
