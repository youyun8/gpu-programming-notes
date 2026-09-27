---
title: 滑動視窗自注意力
platform: LeetGPU
upstream: hard/59_sliding_window_attn
url: https://leetgpu.com/challenges/sliding-window-self-attention
difficulty: hard
tags: [attention, sliding-window, flash-attention, local-attention]
status: solved
---

# 滑動視窗自注意力

**平台：** LeetGPU · **難度：** 困難 · [題目說明](https://leetgpu.com/challenges/sliding-window-self-attention)

## 題意

滑動視窗自注意力：查詢 $i$ 只注意鍵 $j$，且須符合
$\lvert i - j\rvert \le w$（對稱、非因果的帶狀範圍），作用於
$Q, K, V \in \mathbb R^{M\times d}$（容許誤差 `1e-5`）。
這是 Longformer 與 Mistral 的局部注意力模式（Mistral 使用因果變體）。
**只要核心完全不存取帶狀範圍外的鍵**，成本便可從 $O(M^2 d)$
降至 $O(M w d)$。

## 圖解

![滑動視窗注意力：查詢 i 只看距離 w 以內的鍵](figure.svg)

可見的分數在對角線附近形成寬度 2w + 1 的帶狀區域。第 5 列（綠）看得到鍵 3 … 7；kernel 只走訪與帶狀區域相交的鍵分塊。

## 數學表述

$$
s_{ij} = \frac{\mathbf q_i\cdot\mathbf k_j}{\sqrt d}, \qquad
\mathcal W_i = \{\, j : \max(0, i - w) \le j \le \min(M-1, i + w) \,\}, \qquad
O_{i,:} = \sum_{j\in\mathcal W_i} \frac{e^{s_{ij} - m_i}}{\sum_{j'\in\mathcal W_i} e^{s_{ij'} - m_i}}\,\mathbf v_j
$$

| 符號 | 意義 |
|---|---|
| $M$ | 序列長度 |
| $d$ | 頭維度 |
| $w$ | 視窗半徑（`window_size`） |
| $\mathcal W_i$ | 查詢 $i$ 可見的鍵：最多 $2w + 1$ 個 |
| $s_{ij}$ | 經縮放的分數 |
| $m_i$ | 視窗內的最大值 |
| $O_{i,:}$ | 輸出列 |

## 解題思路

使用 FlashAttention 風格的核心（每個查詢列一個 warp、每區塊 8 列、
共享記憶體中每塊 32 個鍵、每個 lane 為一個鍵評分、線上 softmax），
並做兩項帶狀範圍專用調整：

1. **區塊鍵範圍。** 涵蓋列 $[r_0, r_1]$ 的區塊只需要鍵
   $[\max(0, r_0 - w),\ \min(M-1, r_1 + w)]$。圖塊迴圈只走訪這個範圍，
   即 $\le 8 + 2w$ 個鍵，而不是 $M$ 個。
2. **逐 lane 遮罩。** 在此範圍內，lane $\ell$ 的鍵 $j$ 對該 warp
   的列 $i$ 而言，只有在 $\lvert i - j\rvert \le w$ 時才有效。
   否則分數設為 $-\infty$，權重設為 0。

圖塊迴圈邊界對整個區塊一致，因此每個 warp 都會抵達每個 `__syncthreads()`。

## 成本分析

$$
W \approx 4d\sum_{i}\lvert\mathcal W_i\rvert \le 4dM(2w+1), \qquad \text{tiles per block} = \left\lceil\frac{(r_1 - r_0 + 1) + 2w}{32}\right\rceil
$$

| 符號 | 意義 |
|---|---|
| $W$ | 可見配對的分數與 $PV$ FLOP 數 |
| $r_0,\ r_1$ | 區塊的第一個與最後一個查詢列 |

成本對 $M$ 呈線性（固定 $w$）。當 $w$ 很小時，圖塊中多數 lane 會被遮罩
（每塊有 32 個鍵，每列的帶狀範圍為 $2w+1$），因此較大的 $w$ 或每區塊更多列
可提高效率。

## 常見陷阱

- **對稱視窗。** 參考實作會遮罩兩側所有 $\lvert j - i\rvert > w$ 的位置；
  這不是因果注意力。
- **在兩端截斷。** 靠近 0 或 $M-1$ 的列能看到較少的鍵，各列分母也不同。
- **屏障一致性**，同[因果注意力](../053-casual-attention/)。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆以
`1e-5` 通過，包括 $w = 0$（每個查詢只看自己：輸出 = $V$），以及
$w \ge M$（完整注意力）。

## 延伸閱讀

- [因果注意力](../053-casual-attention/)、[Softmax 注意力](../006-softmax-attention/)、
  [含匯聚點的注意力](../112-attention-with-sinks/)。
