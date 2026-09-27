---
title: 1D 最大池化
platform: Tensara
upstream: max-pool-1d
url: https://tensara.org/problems/max-pool-1d
difficulty: easy
tags: [pooling, stencil, dilation, grid-stride]
status: solved
---

# 1D 最大池化

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/max-pool-1d)

## 題意

對 float32 張量執行 1D 最大池化，視窗邊長為 $k$、步距為 $S$、
填補為 $P$、**膨脹率**為 $\delta$，結果須與
`F.max_pool1d(x, k, S, P, dilation=δ)` 一致。填補位置視為
$-\infty$（永遠不會成為最大值）。輸入是很長的 1D 訊號，
$H$ = 2 M … 33 M，參數例如 $(k, S, P, \delta) = (7, 4, 3, 1)$
與 $(4, 2, 1, 2)$。檢查條件為 `rtol = 1e-4`、`atol = 7e-5`。

## 圖解

![帶膨脹的一維最大池化：取樣點相隔 δ，填補值永遠不會勝出](figure.svg)

膨脹為 2 時，視窗每隔一個輸入取一個：輸出 1 是 x₁、x₃、x₅ 的最大值。兩端的填補視為 −∞。

## 數學表述

膨脹視窗在每一軸上涵蓋 $\delta(k-1) + 1$ 個輸入位置，因此

$$
X_{\text{out}} = \left\lfloor \frac{X + 2P - \delta(k - 1) - 1}{S} \right\rfloor + 1
$$

| 符號 | 意義 |
|---|---|
| $X$ | 單一軸上的輸入範圍（$H$） |
| $X_{\text{out}}$ | 該軸上的輸出範圍 |
| $k$ | 視窗邊長（`kernel_size`） |
| $S$ | 步距 |
| $P$ | 每側的填補量 |
| $\delta$ | 膨脹率：相鄰視窗取樣點之間的距離 |

$$
\text{out}[t] = \max_{0 \le m < k,\ 0 \le q < H} x[q], \qquad q = tS - P + m\delta
$$

| 符號 | 意義 |
|---|---|
| $x$ | 輸入訊號，長度為 $H$ |
| $t$ | 輸出索引 |
| $m$ | 視窗內的取樣點索引 |
| $q$ | 取樣點 $m$ 的輸入位置（$q$ 在 $[0, H)$ 外即為填補） |

## 解題思路

1. **每個輸出使用一個執行緒**（以網格跨步迴圈走訪攤平後的輸出索引）；
   相鄰執行緒產生最內層軸上的相鄰輸出，因此視窗會重疊，載入由 L1/L2 提供。
2. **以邊界測試處理填補**：略過輸入範圍外的取樣點，等同將它們視為 $-\infty$。
3. **累加器**從 $-\text{FLT\_MAX}$ 開始，並以 `fmaxf` 摺疊。
   `max` 是精確運算，因此結果與 PyTorch 的位元完全相同。

## 成本分析

$$
W_{\text{ops}} = k^{1}\,H_{\text{out}}, \qquad Q \approx 4\,(\text{input size} + H_{\text{out}})\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{ops}}$ | 比較次數 |
| $Q$ | 必要的 DRAM 位元組數，假設重疊視窗會命中快取 |
| $\beta$ | DRAM 頻寬 |

當 $S < k$ 時，視窗會重疊且核心受頻寬限制；步距很大時，每個輸入仍只讀取一次。

## 常見陷阱

- **輸出大小中的膨脹率**：有效視窗大小為 $\delta(k-1)+1$，不是 $k$。
- **填補是 $-\infty$，不是 0**：若視窗只看到負值與填補，應傳回最大的負值，
  而不是 0。
- PyTorch 要求 $P \le k/2$，因此每個視窗至少包含一個真實元素，
  結果絕不會是 $-\text{FLT\_MAX}$。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [2D 最大池化](../max-pool-2d/)、[3D 最大池化](../max-pool-3d/)、
  [1D 平均池化](../avg-pool-1d/)。
