---
title: 含匯聚權杖的注意力
platform: LeetGPU
upstream: medium/112_attention_with_sinks
url: https://leetgpu.com/challenges/attention-with-sinks
difficulty: medium
tags: [attention, sliding-window, streaming-llm, causal-mask]
status: solved
---

# 含匯聚權杖的注意力

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/attention-with-sinks)

## 題意

**StreamingLLM** 的注意力模式：每個查詢都能看到前 `num_sinks` 個權杖（「注意力匯聚權杖」），以及由最近 `window_size` 個權杖組成的滑動視窗（$Q, K, V \in \mathbb R^{M\times d}$、$M \le 10^4$、$d \le 128$、匯聚權杖數 $\le 16$）。這可讓任意長度串流的 KV 快取維持固定大小，同時避免單純視窗法在逐出起始權杖時造成品質崩潰；這些起始權杖通常會吸收大量注意力權重。

## 圖解

![Attention sink：前 nₛ 個 token 加上最近 w 個 token 的滑動視窗](figure.svg)

每個查詢都看得到兩個 sink 欄（橘）與最近三個鍵組成的視窗（藍）。因此第 9 列（綠）注意的是鍵 0、1、7、8、9。

## 數學表述

$$
\mathcal A_i = \bigl\{\, j \le i \ :\ j < n_s\ \ \lor\ \ j \ge i - w + 1 \,\bigr\}, \qquad
O_i = \sum_{j\in\mathcal A_i}\frac{e^{s_{ij} - m_i}}{\sum_{j'\in\mathcal A_i} e^{s_{ij'} - m_i}}\,V_j, \qquad s_{ij} = \frac{Q_i\cdot K_j}{\sqrt d}
$$

| 符號 | 意義 |
|---|---|
| $M,\ d$ | 序列長度與頭維度 |
| $n_s$ | 匯聚權杖數（`num_sinks`） |
| $w$ | 視窗大小（`window_size`），包含目前權杖 |
| $\mathcal A_i$ | 查詢 $i$ 可用的鍵：匯聚權杖（不超過 $i$）加上最近 $w$ 個位置 |
| $s_{ij}$ | 縮放後的分數 |
| $m_i$ | 可用集合中的最大值 |
| $O_i$ | 輸出資料列 |

$\lvert\mathcal A_i\rvert \le n_s + w$，因此每個查詢的成本固定，而非 $O(i)$。

## 解題思路

Flash 風格核心（8 個 warp = 每區塊 8 個查詢資料列 $[r_0, r_1]$、每次 32 個 K/V 的分塊、每個 lane 負責一個鍵的評分、線上 softmax）只會走訪**兩個鍵範圍**：

1. 匯聚權杖 $[0,\ \min(n_s, r_1 + 1))$；
2. 視窗聯集 $[\max(r_0 - w + 1, \text{end of range 1}),\ r_1]$，也就是 8 個資料列中任一列可能需要的所有內容。

兩個範圍都會逐分塊迭代。在分塊內，僅當 $j \le i \land (j < n_s \lor j \ge i - w + 1)$ 時，lane $\ell$ 的鍵 $j$ 才能供該 warp 的資料列 $i$ 使用；否則其分數為 $-\infty$、權重為 0。迴圈邊界在整個區塊內一致，因此所有 warp 都會參與每個屏障。

## 成本分析

$$
W \approx 4d\sum_i \lvert\mathcal A_i\rvert \le 4dM(n_s + w), \qquad \text{tiles per block} \approx \left\lceil\frac{n_s}{32}\right\rceil + \left\lceil\frac{w + 7}{32}\right\rceil
$$

| 符號 | 意義 |
|---|---|
| $W$ | 可用配對的浮點運算次數 |
| 每區塊分塊數 | 每個含 8 列的區塊載入的鍵分塊數 |

成本與 $M$ 呈線性關係。當 $w$ 很小時，一個分塊內的大多數 lane 都會被遮罩，這與[滑動視窗注意力](../059-sliding-window-attn/)有相同的效率問題。

## 常見陷阱

- 早期資料列的**匯聚權杖與視窗會重疊**：第二個範圍從第一個範圍的末端開始，因此不會重複走訪任何鍵，避免在 softmax 中重複計算。
- **因果性也適用於匯聚權杖**：當 $n_s = 4$ 時，資料列 0 仍只能看到鍵 0。
- **視窗慣例**：$j \ge i - w + 1$ 包含目前權杖，亦即總共有 $w$ 個鍵。

## 驗證

所有 LeetGPU 測試案例都在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $n_s \ge M$（完整因果注意力）與 $w = 1$。

## 延伸閱讀

- [滑動視窗注意力](../059-sliding-window-attn/)、[因果注意力](../053-casual-attention/)、[INT8 KV 快取注意力](../096-int8-kv-cache-attention/)。
