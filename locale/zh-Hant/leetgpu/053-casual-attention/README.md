---
title: 因果自注意力
platform: LeetGPU
upstream: hard/53_casual_attention
url: https://leetgpu.com/challenges/causal-self-attention
difficulty: hard
tags: [attention, causal-mask, flash-attention, online-softmax]
status: solved
---

# 因果自注意力

**平台：** LeetGPU · **難度：** 困難 · [題目說明](https://leetgpu.com/challenges/causal-self-attention)

## 問題

因果（遮罩）單頭自注意力：$Q, K, V \in \mathbb R^{M\times d}$
（float32，$M \le 10^4$、$d \le 128$；容許誤差 `1e-4`）。查詢 $i$
只能注意到 $j \le i$ 的鍵，就像所有純解碼器語言模型在訓練或預填充時一樣。
遮罩會移除約一半工作，而良好的核心會直接略過，不會先計算再丟棄。

## 公式

$$
s_{ij} = \frac{\mathbf q_i\cdot\mathbf k_j}{\sqrt d}, \qquad
\tilde s_{ij} = \begin{cases} s_{ij}, & j \le i \\ -\infty, & j > i\end{cases}, \qquad
O_{i,:} = \sum_{j=0}^{i} \frac{e^{\tilde s_{ij} - m_i}}{\sum_{j' \le i} e^{\tilde s_{ij'} - m_i}}\ \mathbf v_j
$$

| 符號 | 意義 |
|---|---|
| $M$ | 序列長度（查詢數 = 鍵數 = 值數） |
| $d$ | 頭維度 |
| $\mathbf q_i,\ \mathbf k_j,\ \mathbf v_j$ | $Q$、$K$、$V$ 的列 |
| $s_{ij}$ | 經縮放的分數 |
| $\tilde s_{ij}$ | 遮罩後的分數（$e^{-\infty} = 0$，所以未來的鍵權重為 0） |
| $m_i$ | *可見*鍵 $j \le i$ 的列最大值 |
| $O_{i,:}$ | 輸出列 $i$ |

第 $i$ 列有 $i + 1$ 個可見鍵，因此（查詢、鍵）配對總數為

$$
\sum_{i=0}^{M-1} (i + 1) = \frac{M(M+1)}{2} \approx \frac{M^2}{2}
$$

## 方法

採用 FlashAttention 風格的核心（同[柔性最大值注意力](../006-softmax-attention/)），
並做兩項因果調整：

- **區塊層級略過圖塊。** 一個區塊負責連續 8 個查詢列
  $[r_0, r_0 + 7]$，只反覆處理 $j_0 \le \text{last\_row}$ 的鍵圖塊。
  對區塊中每一列而言都完全位於未來的圖塊不會載入，因此可省下約一半圖塊。
- **對角圖塊內逐 lane 遮罩。** Lane $\ell$ 為鍵
  $j = j_0 + \ell$ 評分，並設定 `allowed = active && j <= row`。
  被遮罩的 lane 令 $p = 0$，且在圖塊最大值中使用分數 $-\infty$。

其餘都是標準作法：8 個 warp（每個負責一個查詢列）、共享記憶體中每塊 32 個鍵的
K/V 圖塊（pitch 129 以避免 bank 衝突）、每個 lane 為一個鍵評分、透過
$\alpha = e^{m - m'}$ 重新縮放的線上 softmax、以 `__shfl_sync` 廣播
$p_j$ 來更新 $PV$，以及每個 lane 使用 4 個暫存器累加器（$d \le 128$）。

每一列的第一個鍵（$j = 0$）永遠可見，因此執行中的分母為正，絕不會除以零。

## 成本分析

$$
W \approx 4d\cdot\frac{M(M+1)}{2} = 2dM(M+1), \qquad \text{tiles loaded per block} = \left\lceil\frac{r_0 + 8}{32}\right\rceil
$$

| 符號 | 意義 |
|---|---|
| $W$ | 僅計可見配對的分數與 $PV$ FLOP 數 |
| $r_0$ | 區塊的第一個查詢列 |

這是稠密注意力成本的一半。當 $M = 10^4$、$d = 128$ 時：
$W \approx 2.6\times10^{10}$ FLOP，受限於 fp32 FMA 與共享記憶體載入的運算能力。

## 注意事項

- **在每個 warp 中略過遮罩迴圈**（而不是以區塊為單位）會使 warp
  在 `__syncthreads()` 發生分歧。圖塊迴圈上限必須對整個區塊一致
  （`last_row`），細部遮罩則逐 lane 套用。
- **直接使用 $-\infty$** 來計算 `expf(-inf - m)` 沒有問題（結果為 0），
  但 `-inf - (-inf)` 是 NaN。第一個圖塊一定包含鍵 0，因此從第一個圖塊起，
  執行中的最大值就是有限值。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆以
`1e-4` 通過，包括 $M = 1$，以及 $M$ 不是 8 或 32 的倍數。

## 相關內容

- [柔性最大值注意力](../006-softmax-attention/)、[滑動視窗注意力](../059-sliding-window-attn/)、
  [衰減因果注意力](../092-decaying-causal-attention/)、[含匯聚點的注意力](../112-attention-with-sinks/)。
