---
title: 含線性偏置的注意力
platform: LeetGPU
upstream: medium/55_attn_w_linear_bias
url: https://leetgpu.com/challenges/attention-with-linear-biases
difficulty: medium
tags: [attention, alibi, gemm, softmax, fused-epilogue]
status: solved
---

# 含線性偏置的注意力

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/attention-with-linear-biases)

## 問題

含線性偏置的注意力（**ALiBi**，Press 等人，2022）。不使用位置嵌入，
而是將與查詢和鍵之間距離成正比的偏置加到分數上：
$Q \in \mathbb R^{M \times d}$、$K, V \in \mathbb R^{N\times d}$，
斜率 $\alpha \in [-1, 1]$（$M, N \le 2048$、$d \le 1024$；
基準測試為 $M = N = 2048$；容許誤差 `1e-4`）。

## 公式

$$
S_{ij} = \frac{\mathbf q_i\cdot\mathbf k_j}{\sqrt d} + \alpha\,(i - j), \qquad
P_{ij} = \frac{e^{S_{ij} - m_i}}{\sum_{j'} e^{S_{ij'} - m_i}}, \qquad
O = PV
$$

| 符號 | 意義 |
|---|---|
| $M,\ N$ | 查詢數與鍵數 |
| $d$ | 頭維度（最高 1024） |
| $\mathbf q_i,\ \mathbf k_j$ | $Q$ 與 $K$ 的列 |
| $\alpha$ | ALiBi 斜率（單頭，因此只有一個斜率） |
| $i - j$ | 帶正負號的相對位置（查詢索引減去鍵索引） |
| $S_{ij}$ | 加入偏置並縮放後的分數 |
| $m_i$ | $S$ 的列最大值 |
| $P_{ij}$ | 注意力權重（逐列 softmax） |
| $O$ | 輸出，$M \times d$ |

當 $\alpha < 0$（因果語言模型常用的設定）時，距離較遠的鍵會受到線性懲罰。
這讓模型能外推至比訓練時更長的序列。

## 方法

當 $d$ 最高達 1024 時，其他地方採用的逐 warp flash 設計會需要很大的暫存器累加器。
由於 $M N \le 4.2$M，分數矩陣（≤ 16 MB）可輕鬆容納，因此使用清楚的
**三核心**管線：

1. **$S = QK^{\mathsf T}\cdot\tfrac{1}{\sqrt d} + \alpha(i - j)$**：
   使用「NT」形式、64 × 64 且以暫存器分塊的 SGEMM（$B$ 運算元以轉置方式讀取）。
   載入 $B$ 圖塊時以 $k$ 為最快變動維度，因此對 $K$ 各列的讀取可合併。
   縮放與 ALiBi 項在**結尾階段**套用，此時每個執行緒已知道自己的 $(i, j)$。
   偏置不增加額外成本。
2. **原地逐列 softmax**，每列一個 warp：求最大值、取指數、求和，再正規化。
3. **$O = PV$**：使用同一個 SGEMM 範本的「NN」形式。

範本 `sgemm<kTransB, kAlibi>` 透過編譯期分支，從同一份原始碼產生兩個 GEMM。

## 成本分析

$$
W = 2MNd + 2MNd + O(MN), \qquad Q \approx 4\,(MN\cdot 3) + 4\left(Md + Nd\right)\cdot\frac{\max(M,N)}{64} + 4Md
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP 數：兩個 GEMM 加上 softmax |
| $Q$ | 位元組數：寫入 $S$、在 softmax 中讀寫、在第二個 GEMM 中讀取，加上分塊 GEMM 的運算元流量 |

當 $M = N = 2048$、$d = 1024$ 時：$W \approx 17$ GFLOP（由 GEMM 主導），
分數流量為 48 MB。具現化 $S$ 比融合的 flash 核心多出約 3 × 16 MB 的流量，
相較於此處的 GEMM 執行時間並不大。

## 注意事項

- **偏置的正負號。** 公式是 $\alpha\,(i - j)$，其中 $i$ 是查詢列、
  $j$ 是鍵欄，與參考實作的 `arange(M)[:,None] - arange(N)[None,:]` 完全一致。
- **先加偏置再求最大值。** 偏置是分數的一部分，必須在 softmax 的最大值減法前加入。
- **此題沒有因果遮罩。** ALiBi 本身不是遮罩。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆以
`1e-4` 通過，包括 $\alpha = \pm1$、$d = 1024$ 與 $M \ne N$。

## 相關內容

- [Softmax 注意力](../006-softmax-attention/)、[因果注意力](../053-casual-attention/)、
  [衰減因果注意力](../092-decaying-causal-attention/)。
