---
title: Log Softmax
platform: Tensara
upstream: log-softmax
url: https://tensara.org/problems/log-softmax
difficulty: easy
tags: [softmax, online-softmax, warp-per-row]
status: solved
---

# Log Softmax

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/log-softmax)

## 題意

對 $M\times N$ 的 float32 矩陣（$4096^2$ … $8192^2$）逐列計算
log-softmax，結果須與 `F.log_softmax(x, dim=1)` 一致。檢查條件為
`rtol = 1e-4`、`atol = 2e-5`。

## 圖解

![逐列 log-softmax：每個元素減去該列的 log-sum-exp](figure.svg)

掃過一列一次即可求出 log-sum-exp（紫色）；輸出列就是輸入減去這個數。

## 數學表述

$$
y_{ij} = \ln\frac{e^{x_{ij}}}{\sum_{k} e^{x_{ik}}} = x_{ij} - \operatorname{LSE}_i, \qquad
\operatorname{LSE}_i = m_i + \ln\sum_{k=0}^{N-1} e^{x_{ik} - m_i}, \qquad m_i = \max_k x_{ik}
$$

| 符號 | 意義 |
|---|---|
| $x_{ij}, y_{ij}$ | 第 $i$ 列、第 $j$ 欄的輸入與輸出 |
| $\operatorname{LSE}_i$ | 第 $i$ 列的 log-sum-exp |
| $m_i$ | 該列最大值；減去它可讓每個指數都 $\le 0$（不會溢位） |

最大值與總和會在一次走訪中，以數對 $(m, s)$ 的**線上**更新同時計算，
其中 $s = \sum e^{x - m}$：

$$
(m_1, s_1) \oplus (m_2, s_2) = \Bigl(M,\ s_1 e^{m_1 - M} + s_2 e^{m_2 - M}\Bigr), \qquad M = \max(m_1, m_2)
$$

| 符號 | 意義 |
|---|---|
| $(m, s)$ | 累計最大值，以及 $e^{x - m}$ 的累計總和 |
| $\oplus$ | 具結合律的合併；新元素 $x$ 以 $(x, 1)$ 合併 |
| $M$ | 兩個最大值中較大者 |

## 解題思路

1. **每列使用一個 warp。** 每個 lane 以 $\oplus$ 將跨步取得的元素合併到私有的
   $(m, s)$ 數對；再用 5 步 `__shfl_xor_sync` 蝶形運算合併 32 個數對，
   讓每個 lane 最後都得到該列的 $(m_i, s_i)$。
2. $\operatorname{LSE}_i = m_i + \ln s_i$。
3. **寫入階段**：$y_{ij} = x_{ij} - \operatorname{LSE}_i$（不需要指數運算）。

## 成本分析

$$
Q = 4MN\ (\text{read}) + 4MN\ (\text{re-read}) + 4MN\ (\text{write}), \qquad \#\exp = MN\ (\text{plus merges})
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 位元組數；若一列（最多 32 KB）仍在快取中，重新讀取會命中 L2 |
| #exp | 線上走訪中的指數運算數量 |

在 $8192^2$ 時：輸入與輸出共 268 MB，以 2 TB/s 計約 0.27 ms。
與 softmax 不同，寫入階段不需要第二次指數運算。

## 常見陷阱

- **穩定性**：當 $x > 88$ 時，$\ln\sum e^{x}$ 會溢位；務必先減去最大值。
- **初始值 $m = -\text{FLT\_MAX}$**（而非 $-\infty$），如此
  $e^{m_1 - M}$ 才不會計算出 $e^{-\infty + \infty}$。
- **不要計算 $\ln(\operatorname{softmax})$**：很小的機率會下溢為 0，產生 $-\infty$。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [Softmax](../softmax/)、[KL 損失](../kl-loss/)、
  LeetGPU [Softmax](../../leetgpu/005-softmax/)、
  LeetGPU [類別交叉熵](../../leetgpu/025-categorical-cross-entropy-loss/)。
