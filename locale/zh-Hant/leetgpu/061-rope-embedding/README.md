---
title: 旋轉位置嵌入
platform: LeetGPU
upstream: medium/61_rope_embedding
url: https://leetgpu.com/challenges/rotary-positional-embedding
difficulty: medium
tags: [elementwise, rope, llm, positional-encoding]
status: solved
---

# 旋轉位置嵌入

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/rotary-positional-embedding)

## 問題

對 $M$ 個維度為 $D$ 的查詢向量套用旋轉位置嵌入（RoPE），並提供形狀相同的
預先計算 $\cos$ 與 $\sin$ 表（$M, D \le 10^4$，$D$ 為偶數；
基準測試為 $M = 2^{20}$、$D = 128$；容許誤差 `1e-4`）。
這些表使用 LLaMA/GPT-NeoX 的**對半分割**配置。RoPE 將絕對位置編碼為旋轉，
因此點積 $\mathbf q_m \cdot \mathbf k_n$ 只取決於相對位移 $m - n$。

## 公式

$$
\operatorname{RoPE}(\mathbf x) = \mathbf x \odot \mathbf c + \operatorname{rotate\_half}(\mathbf x)\odot\mathbf s, \qquad
\operatorname{rotate\_half}([\mathbf x_1;\ \mathbf x_2]) = [-\mathbf x_2;\ \mathbf x_1]
$$

對單一列及 $0 \le j < D/2$（令 $h = D/2$）：

$$
y_j = x_j\,c_j - x_{j+h}\,s_j, \qquad y_{j+h} = x_{j+h}\,c_{j+h} + x_j\,s_{j+h}
$$

| 符號 | 意義 |
|---|---|
| $M$ | Token 數（列數） |
| $D$ | 頭維度（偶數） |
| $h$ | 維度的一半 $D/2$ |
| $\mathbf x$ | $Q$ 的一列 |
| $\mathbf x_1,\ \mathbf x_2$ | $\mathbf x$ 的前半與後半 |
| $\mathbf c,\ \mathbf s$ | `cos` 與 `sin` 表的列；對半分割表示 $c_j = c_{j+h}$ 且 $s_j = s_{j+h}$ |
| $\odot$ | 逐元素乘積 |
| $\mathbf y$ | 輸出列 |

當 $c_j = \cos(m\theta_j)$、$s_j = \sin(m\theta_j)$，且 token 位置為
$m$ 時，每一對 $(x_j, x_{j+h})$ 都會在自己的二維平面中
旋轉 $m\theta_j$：

$$
\begin{pmatrix} y_j \\ y_{j+h}\end{pmatrix} = \begin{pmatrix}\cos m\theta_j & -\sin m\theta_j\\ \sin m\theta_j & \cos m\theta_j\end{pmatrix}\begin{pmatrix} x_j \\ x_{j+h}\end{pmatrix}
$$

| 符號 | 意義 |
|---|---|
| $m$ | Token 位置 |
| $\theta_j$ | 第 $j$ 對的頻率（通常為 $10000^{-2j/D}$） |

## 方法

一列中的每一**對** $(j, j+h)$ 使用一個執行緒：共 $M \cdot D/2$ 個執行緒，
以 64 位元索引進行網格步幅迴圈。

- 載入 $x_j$、$x_{j+h}$、兩個 $c$ 與兩個 $s$，再計算兩個輸出。
- 每個輸入元素恰好讀取一次。若逐元素映射，每個 $x$ 都會讀取兩次。
- 合併存取：連續執行緒會存取一列內連續的 $j$，所以 warp
  會從前半部讀取 32 個連續 float，再從後半部讀取 32 個：每個陣列各有兩個完整區段。

核心同時使用 $c_j$ 與 $c_{j+h}$（而非只用一個），因此即使呼叫端傳入
未重複的表，也能符合參考實作。

## 成本分析

$$
Q = 4\cdot 4MD \ \text{bytes} \quad(\text{read } Q, \cos, \sin;\ \text{write output}), \qquad W = 3MD
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| $W$ | FLOP 數：每個輸出元素 2 次乘法與 1 次加法 |

基準測試 $M = 2^{20}$、$D = 128$ 時，$Q = 2.1$ GB，在 2 TB/s 下約需 1 ms。
完全受記憶體頻寬限制。正式環境的核心會即時計算 $\cos/\sin$，其輸入為 $m\theta_j$
（使流量減半），並將 RoPE 融合至 QKV 投影的結尾階段或注意力核心中。

## 注意事項

- **交錯與對半分割不同。** GPT-J 風格的 RoPE 旋轉相鄰配對
  $(2j, 2j+1)$。此題使用兩半中的配對 $(j, j+h)$。
- **正負號慣例。** `rotate_half` 會將*後半部*取負，再移到前方。
- **網格大小。** 最大尺寸時 $MD/2$ 可能超過 $2^{31}$，所以索引使用
  `long long`，並限制網格大小，再以網格步幅迴圈處理。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆以
`1e-4` 通過，包括 $D = 2$。

## 相關內容

- [多頭潛在注意力](../114-multi-head-latent-attention/)（解耦的 RoPE）、
  [LLaMA Transformer 區塊](../093-llama-transformer-block/)。
