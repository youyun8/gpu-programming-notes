---
title: 衰減因果注意力
platform: LeetGPU
upstream: medium/92_decaying_causal_attention
url: https://leetgpu.com/challenges/decaying-causal-attention
difficulty: medium
tags: [attention, retention, retnet, causal-mask]
status: solved
---

# 衰減因果注意力

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/decaying-causal-attention)

## 問題

**保留機制**（Retention，RetNet）的平行形式：不使用 softmax 的因果注意力，其權重會隨距離呈幾何衰減（$Q, K, V \in \mathbb R^{S\times d}$、$S \le 8192$、$d \le 256$、$0 < \gamma \le 1$；效能測試為 $S = 4096$、$d = 64$；容許誤差 `1e-3`）。Retention 也有等價的*遞迴*形式，每一步只需 $O(1)$ 狀態；這正是它用於推論時的主要優點。

## 公式

$$
O_n = \sum_{m=0}^{n} \gamma^{\,n-m}\; \frac{Q_n\cdot K_m}{\sqrt d}\; V_m, \qquad
\text{i.e.}\quad O = \Bigl(\tfrac{1}{\sqrt d}QK^{\mathsf T}\odot D\Bigr)V, \quad D_{nm} = \begin{cases}\gamma^{n-m}, & m \le n\\ 0, & m > n\end{cases}
$$

| 符號 | 意義 |
|---|---|
| $S$ | 序列長度 |
| $d$ | 模型／注意力頭維度 |
| $Q_n,\ K_m,\ V_m$ | 查詢、鍵和值矩陣的資料列 |
| $\gamma$ | 位於 $(0, 1]$ 的衰減因子 |
| $D$ | 因果衰減遮罩（下三角） |
| $O_n$ | 輸出資料列 $n$ |

**等價遞迴形式**（此處未使用，但可用來說明模型）：

$$
\mathbf S_n = \gamma\,\mathbf S_{n-1} + K_n^{\mathsf T}V_n, \qquad O_n = \tfrac{1}{\sqrt d}\,Q_n\,\mathbf S_n
$$

| 符號 | 意義 |
|---|---|
| $\mathbf S_n$ | 位置 $n$ 之後的 $d\times d$ 遞迴狀態 |

由於沒有 softmax，因此不需要計算每列最大值或正規化因子，每一項都只是單純的加權總和。

## 方法

採用 FlashAttention 的骨架，但不包含線上 softmax 的簿記：

- 8 個 warp = 每區塊 8 個查詢資料列。由 32 個鍵組成的 K 與 V 分塊會暫存在**動態**共享記憶體中（K 的間距為 $d+1$）。當 $d = 256$ 時最多約 74 KB，因此啟動端會透過 `cudaFuncSetAttribute` 選用較大的共享記憶體。
- lane $\ell$ 計算鍵 $j = j_0 + \ell$ 的完整內積，再乘上 $\gamma^{n - j}$（`powf`）；若 $j > n$，則使用 0。
- $PV$ 更新會以 `__shfl_sync` 廣播每個 lane 的權重，且每個 lane 在暫存器中累加其 $\le 8$ 個輸出欄。
- **略過因果分塊。** 鍵分塊只處理到該區塊的最後一列，因此約有一半的分塊完全不會被存取。

## 成本分析

$$
W \approx 4d\cdot\frac{S(S+1)}{2} + \frac{S(S+1)}{2}\,c_{\text{pow}}, \qquad Q_{\min} = 16Sd\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 因果三角範圍內分數與 $PV$ 的浮點運算次數，另加每個可見配對一次 `powf` |
| $c_{\text{pow}}$ | `powf` 的成本（約 20–40 條指令） |
| $Q_{\min}$ | 各讀取 $Q, K, V$ 一次，並寫入 $O$ 一次 |

效能測試：約 $2.1$ GFLOP，加上 $8.4$M 次 `powf`。較省成本的替代方式是從兩個小型查找表計算 $\gamma^{n-j} = \gamma^{n-j_0}\cdot\gamma^{-\ell}$，但在 $\gamma$ 很小且距離很長時會溢位；`powf` 較穩健。

## 常見問題

- **縮放只套用於分數。** $\gamma^{n-m}$ 會在縮放後的內積上相乘，兩者共同構成同一個權重。
- **下溢。** 當 $\gamma < 1$ 且距離很大時，$\gamma^{n-m}$ 會下溢為 0，這是正確結果。
- **不做正規化。** 若習慣性地像其他注意力核心一樣加入 softmax，結果會出錯。

## 驗證

所有 LeetGPU 測試案例都以 `1e-3` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $\gamma = 1$（因果線性注意力）與 $d = 256$。

## 相關內容

- [因果注意力](../053-casual-attention/)、[線性注意力](../056-linear-attention/)、[線性遞迴](../082-linear-recurrence/)（遞迴觀點）。
