---
title: 三元組邊界損失
platform: Tensara
upstream: triplet-margin
url: https://tensara.org/problems/triplet-margin
difficulty: medium
tags: [loss, reduction, row-per-block, fp64-accumulation]
status: solved
---

# 三元組邊界損失

**平台：** Tensara · **難度：** 中等 · [題目敘述](https://tensara.org/problems/triplet-margin)

## 問題

為一批 $B$ 個三元組（錨點、正例、負例）計算三元組邊界損失。嵌入大小為
$E$（最大 $B = 1024$、$E = 16384$），邊界值在執行期指定；結果須符合
`nn.TripletMarginLoss(margin)`（L2 距離、平均歸約）。輸出為一個純量。
檢查條件為 `rtol = atol = 6e-4`。

## 公式

$$
d(\mathbf{x}, \mathbf{y}) = \Bigl\lVert \mathbf{x} - \mathbf{y} + \epsilon\mathbf{1} \Bigr\rVert_2 = \sqrt{\sum_{e=0}^{E-1} (x_e - y_e + \epsilon)^2}
$$

$$
\ell_i = \max\bigl(0,\ d(\mathbf{a}_i, \mathbf{p}_i) - d(\mathbf{a}_i, \mathbf{n}_i) + m\bigr), \qquad
\mathcal{L} = \frac{1}{B}\sum_{i=0}^{B-1} \ell_i
$$

| 符號 | 意義 |
|---|---|
| $\mathbf{a}_i, \mathbf{p}_i, \mathbf{n}_i$ | 三元組 $i$ 的錨點、正例與負例嵌入（$B\times E$ 矩陣的列） |
| $d$ | `torch.pairwise_distance`：每個分量的差加上 $\epsilon$ 後的 L2 範數 |
| $\epsilon$ | $10^{-6}$（PyTorch 預設值） |
| $m$ | 邊界 |
| $\ell_i$ | 每個三元組的 hinge |
| $\mathcal{L}$ | 純量輸出 |

只要負例與錨點之間的距離，比正例與錨點之間的距離至少多 $m$，
損失便為 0。

## 方法

1. **每個三元組使用一個區塊**（256 個執行緒）。每個執行緒跨步走訪
   $E$ 個欄，並在一次走訪中同時累加兩個距離平方
   $(a - p + \epsilon)^2$ 與 $(a - n + \epsilon)^2$：錨點列只讀取一次，
   三列都使用合併存取。
2. 兩次區塊歸約產生 $d_{ap}$ 與 $d_{an}$；執行緒 0 將 $\ell_i$
   寫入小型暫存緩衝區。
3. **最後使用單一區塊**，以 `double` 加總 $B$ 個 hinge 值，
   並寫入 $\mathcal{L}$。

## 成本分析

$$
Q = 12BE\ \text{bytes}, \qquad W = 6BE\ \text{flops}, \qquad T_{\min} = \frac{12BE}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取一次三個 $B\times E$ 矩陣 |
| $W$ | 每欄兩條減法、加法與 FMA 鏈 |
| $\beta$ | DRAM 頻寬 |

當 $B = 256$、$E = 16384$ 時，資料量為 50 MB；在 2 TB/s 下約為 25 µs。

## 注意事項

- **$\epsilon$ 位於差值內**，而非加至範數上；省略它會讓相對結果改變約
  $\epsilon\sqrt{E}$，對彼此接近的配對會造成影響。
- **對批次取平均**，不是取總和。
- **先對每個三元組取 hinge，再計算平均值**。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [Hinge 損失](../hinge-loss/)、[餘弦相似度](../cosine-similarity/)、
  [L2 範數](../l2-norm/)。
