---
title: 縮放點積注意力
platform: Tensara
upstream: scaled-dot-attention
url: https://tensara.org/problems/scaled-dot-attention
difficulty: hard
tags: [attention, flash-attention, online-softmax, shared-memory]
status: solved
---

# 縮放點積注意力

**平台：** Tensara · **難度：** 困難 · [題目敘述](https://tensara.org/problems/scaled-dot-attention)

## 題意

對張量 $Q, K, V$ 計算非因果縮放點積注意力，其形狀為 $(B, H, S, E)$，
結果須符合不使用遮罩或 dropout 的 `F.scaled_dot_product_attention`。
測試形狀從 $(16, 32, 256, 64)$ 到 $(8, 16, 2048, 64)$ 與
$(8, 16, 512, 256)$。檢查條件為 `rtol = 2e-2`、`atol = 5e-3`。

## 圖解

![對每個（批次, head）計算縮放點積注意力：完整 softmax，不使用遮罩](figure.svg)

每個 (b, h) 都是獨立的注意力問題。一組查詢（綠色）以線上 softmax 依序掃過各鍵分塊（深淺藍色）。

## 數學表述

每個批次 $b$ 與注意力頭 $h$ 都各自獨立計算：

$$
s_{ij} = \frac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{E}}, \qquad
P_{ij} = \frac{e^{s_{ij} - m_i}}{\sum_{j'} e^{s_{ij'} - m_i}}, \qquad
\mathbf{o}_i = \sum_{j=0}^{S-1} P_{ij}\,\mathbf{v}_j, \qquad m_i = \max_j s_{ij}
$$

| 符號 | 意義 |
|---|---|
| $B, H, S, E$ | 批次、注意力頭、序列長度、注意力頭維度 |
| $\mathbf{q}_i, \mathbf{k}_j, \mathbf{v}_j$ | 第 $i$ 列（位於 $Q$），以及第 $j$ 列（位於 $K$ 與 $V$；針對一個 $(b, h)$） |
| $s_{ij}$ | 縮放後的分數 |
| $P_{ij}$ | 注意力機率（沿 $j$ 執行 softmax） |
| $m_i$ | 分數的列最大值 |
| $\mathbf{o}_i$ | 長度為 $E$ 的輸出列 |

**線上 softmax** 會以每塊 32 個鍵的區塊 $\mathcal{J}_t$ 處理鍵，
並維護目前最大值 $m$、正規化值 $\ell$ 與未正規化輸出 $\mathbf{u}$：

$$
m' = \max\bigl(m, \max_{j\in\mathcal{J}_t} s_{ij}\bigr), \quad
\ell' = \ell\,e^{m - m'} + \sum_{j\in\mathcal{J}_t} e^{s_{ij} - m'}, \quad
\mathbf{u}' = \mathbf{u}\,e^{m - m'} + \sum_{j\in\mathcal{J}_t} e^{s_{ij} - m'}\mathbf{v}_j
$$

最後得到 $\mathbf{o}_i = \mathbf{u}/\ell$。

| 符號 | 意義 |
|---|---|
| $\mathcal{J}_t$ | 第 $t$ 個、含 32 個鍵的區塊 |
| $m, \ell, \mathbf{u}$ | 目前最大值、目前指數總和、目前值的加權總和 |
| $m', \ell', \mathbf{u}'$ | 處理區塊 $t$ 後的對應值 |
| $e^{m - m'}$ | 最大值增加時套用的重新縮放係數 |

## 解題思路

使用 FlashAttention 風格的融合核心（`flashForward`），並把
$B\cdot H$ 組配對合併至網格中（注意力頭步幅為 $S\cdot E$）：

1. **每個區塊 4 個 warp，每個 warp 負責一個查詢列。**
2. **計分**：對每個含 32 個鍵的區塊，lane $l$ 計算鍵 $l$ 的分數，
   也就是與查詢列進行長度為 $E$ 的點積。
3. **串流讀取 K 與 V**：以注意力頭維度中寬度為 128 的切片通過共享記憶體，
   因此共享記憶體用量不會隨 $E$（最大 1024）增加。
4. **每個 warp 執行線上 softmax**：計算 32 個分數的 warp 最大值與總和，
   並重新縮放累加器。
5. **累加**：lane $l$ 負責輸出欄 $l, l+32, \dots$，並加上
   $\sum_j p_j v_{j,c}$；鍵 $j$ 的機率會透過 shuffle 從 lane $j$ 廣播。

程式完全不會具體建立 $S\times S$ 分數矩陣：每個注意力頭的記憶體用量
是 $O(SE)$，而非 $O(S^2)$。

## 成本分析

$$
W = 4\,BHS^2E\ \text{flops}, \qquad Q_{\min} = 4\cdot 4\,BHSE\ \text{bytes}, \qquad
Q_{\text{K,V}} \approx \frac{S}{4}\cdot 8\,BHSE\ \text{bytes through L2}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 兩次矩陣乘法（$QK^{\mathsf T}$ 與 $PV$），每次 FMA 算 2 flops |
| $Q_{\min}$ | 必要的 DRAM 位元組數：讀取 $Q, K, V$ 並寫入 $O$ |
| $Q_{\text{K,V}}$ | K/V 流量：每個區塊（4 個查詢列）都會重新串流讀取該注意力頭的 $K$ 與 $V$ |

在 $(8, 16, 2048, 64)$ 時，$W = 275$ GFLOP。由於每個區塊僅負責 4 個
查詢列，$K$ 與 $V$ 會透過 L2 重讀 $S/4$ 次。使用較大的查詢區塊
（每區塊 64–128 列，如 FlashAttention-2）與張量核心 MMA，可同時降低
L2 流量與指令數。

## 常見陷阱

- **縮放係數** $1/\sqrt{E}$ 要套用至分數，而非 $V$ 或輸出。
- **穩定性**：減去目前最大值；將 $m$ 初始化為很大的有限負值，以免出現
  $\infty - \infty$。
- **沒有遮罩**：這是雙向注意力。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [Softmax](../softmax/)、LeetGPU [Softmax 注意力](../../leetgpu/006-softmax-attention/)、
  LeetGPU [多頭注意力](../../leetgpu/012-multi-head-attention/)、
  LeetGPU [因果注意力](../../leetgpu/053-casual-attention/)。
