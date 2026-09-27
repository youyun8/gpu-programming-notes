---
title: L1 正規化
platform: Tensara
upstream: l1-norm
url: https://tensara.org/problems/l1-norm
difficulty: easy
tags: [normalization, reduction, row-per-block]
status: solved
---

# L1 正規化

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/l1-norm)

## 題意

對 $B\times D$ float32 矩陣逐列執行 L1 正規化
（$B = 128, 256$；$D = 4096 \dots 16384$）：將每列的每個元素除以
其絕對值總和再加上 $\epsilon = 10^{-10}$。檢查條件為
`rtol = 7e-4`、`atol = 5e-5`。

## 圖解

![逐列 L1 正規化：除以絕對值總和](figure.svg)

標示的列先歸約出 $s_{b}$ = Σ|x|，再把該列每個元素除以 $s_{b}$ + ε。

## 數學表述

$$
s_b = \sum_{d=0}^{D-1} \lvert x_{bd} \rvert, \qquad y_{bd} = \frac{x_{bd}}{s_b + \epsilon}
$$

| 符號 | 意義 |
|---|---|
| $B, D$ | 列數與列長度 |
| $x_{bd}, y_{bd}$ | 第 $b$ 列、第 $d$ 欄的輸入與輸出元素 |
| $s_b$ | 第 $b$ 列的 L1 範數 |
| $\epsilon$ | $10^{-10}$，依參考實作加至範數上（不是用來設定下限） |

正規化後，$\sum_d \lvert y_{bd}\rvert = s_b/(s_b + \epsilon) \approx 1$。

## 解題思路

**每列使用一個區塊**，每個區塊有 256 個執行緒：

1. 每個執行緒跨步走訪該列並加總 $\lvert x\rvert$；區塊歸約（warp
   `__shfl_xor_sync`，再經過一次共享記憶體）會將 $s_b$ 提供給所有
   執行緒。
2. 每個執行緒只計算一次 $r = 1/(s_b + \epsilon)$，再為自己負責的元素
   寫入 $y = x\,r$。該列（最大 64 KB）剛剛才被讀取，因此第二趟大多會
   命中 L2。

## 成本分析

$$
Q_{\text{DRAM}} \approx 4BD + 4BD = 8BD\ \text{bytes}, \qquad \#\text{blocks} = B
$$

| 符號 | 意義 |
|---|---|
| $Q_{\text{DRAM}}$ | 從 DRAM 讀取一次（第二次讀取來自 L2），並寫入一次 |
| #blocks | 每列一個區塊 |

在 $B = 256$、$D = 8192$ 時：共 16.8 MB，以 2 TB/s 計算約需 8 µs。
由於只有 128 至 256 個區塊，每個 SM 只分到一或兩個；此時列走訪的延遲
會成為限制。若將列拆分到一組區塊上（Hopper 的分散式共享記憶體），或以
更多執行緒讓每列使用數個 warp，可以改善效能。

## 常見陷阱

- **$\epsilon$ 是加上去的**，不是用來設定下限。
- **絕對值**：加總帶正負號的 $x$ 會得到另一種（且錯誤的）正規化。
- **乘上倒數**：與參考實作的除法相差 1 ulp。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [L2 範數](../l2-norm/)、[Frobenius 範數](../frobenius-norm/)、
  [RMS 範數](../rms-norm/)。
