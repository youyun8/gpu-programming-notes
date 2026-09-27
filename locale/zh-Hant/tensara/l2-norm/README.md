---
title: L2 正規化
platform: Tensara
upstream: l2-norm
url: https://tensara.org/problems/l2-norm
difficulty: easy
tags: [normalization, reduction, row-per-block]
status: solved
---

# L2 正規化

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/l2-norm)

## 題意

對 $B\times D$ float32 矩陣逐列執行 L2 正規化
（$B = 128, 256$；$D = 4096 \dots 16384$）：將每列除以其 Euclidean
範數再加上 $\epsilon = 10^{-10}$。檢查條件為
`rtol = 1e-4`、`atol = 1e-6`。

## 圖解

![逐列 L2 正規化：除以歐幾里得範數](figure.svg)

標示的列先歸約出歐幾里得範數，再除以它（加上 ε），因此每個輸出列都是單位向量。

## 數學表述

$$
n_b = \sqrt{\sum_{d=0}^{D-1} x_{bd}^2}, \qquad y_{bd} = \frac{x_{bd}}{n_b + \epsilon}
$$

| 符號 | 意義 |
|---|---|
| $B, D$ | 列數與列長度 |
| $x_{bd}, y_{bd}$ | 輸入與輸出元素 |
| $n_b$ | 第 $b$ 列的 L2 範數 |
| $\epsilon$ | $10^{-10}$，在開平方根後加上 |

之後每個輸出列都是單位向量：$\sum_d y_{bd}^2 \approx 1$。

## 解題思路

結構與 [L1 範數](../l1-norm/)相同：每列使用一個有 256 個執行緒的區塊；
第一趟由每個執行緒累加 $x^2$，並在區塊內歸約（warp shuffle 加上共享
記憶體）；第二趟乘以 $1/(\sqrt{s} + \epsilon)$，並從 L2 重新讀取該列。

## 成本分析

$$
Q_{\text{DRAM}} \approx 8BD\ \text{bytes}, \qquad W = 3BD\ \text{flops}
$$

| 符號 | 意義 |
|---|---|
| $Q_{\text{DRAM}}$ | 讀取一次（第二次讀取命中 L2），並寫入一次 |
| $W$ | 以一次 FMA 計算 $x^2$，再以一次乘法縮放 |

## 常見陷阱

- **嚴格的 `atol = 1e-6`**：輸出約為
  $O(1/\sqrt{D}) \approx 0.01$，因此 $10^{-4}$ 的相對誤差會有影響；
  在 16 K 個元素上使用 float 累加（樹狀歸約前每個執行緒處理 64 個）
  已經足夠精確。
- **$\epsilon$ 加在平方根之後**，不是加在根號內（後者是 RMS 範數的
  慣例）。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [L1 範數](../l1-norm/)、[Frobenius 範數](../frobenius-norm/)、
  [餘弦相似度](../cosine-similarity/)、[RMS 範數](../rms-norm/)。
