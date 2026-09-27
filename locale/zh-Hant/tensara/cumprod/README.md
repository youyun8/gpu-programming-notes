---
title: 累積乘積
platform: Tensara
upstream: cumprod
url: https://tensara.org/problems/cumprod
difficulty: medium
tags: [scan, prefix-product, fp64-accumulation]
status: solved
---

# 累積乘積

**平台：** Tensara · **難度：** medium · [題目敘述](https://tensara.org/problems/cumprod)

## 題意

計算長度為 $N$（64 K … 1 M）的 float32 向量之內含式前綴乘積，行為與 `torch.cumprod(x, dim=0)` 相同。大量隨機數的乘積很快就會下溢為 0 或上溢為 $\infty$，因此檢查誤差較寬鬆
（`rtol = 1e-2`、`atol = 2e-2`）；本題的重點是掃描。

## 圖解

![累積乘積：同樣的掃描，改用乘法、單位元為 1](figure.svg)

輸出 4 是輸入 0 … 4 的乘積。每個 chunk 先在內部掃描，再乘上所有前面 chunk 的乘積（進位值）。

## 數學表述

$$
y_i = \prod_{j=0}^{i} x_j = y_{i-1}\,x_i, \qquad y_{-1} = 1
$$

| 符號 | 意義 |
|---|---|
| $x_j$ | 輸入元素 |
| $y_i$ | 輸出：前 $i+1$ 個輸入的乘積 |
| 1 | 乘法的單位元素 |

掃描適用於任何具有單位元素 $e$ 的**可結合**運算子 $\otimes$。將輸入分成長度為 $L$ 的分段 $c$：

$$
T_c = \bigotimes_{j \in c} x_j, \qquad
E_c = \bigotimes_{c' < c} T_{c'}, \qquad
y_i = E_{c(i)} \otimes \bigotimes_{j = cL}^{i} x_j
$$

| 符號 | 意義 |
|---|---|
| $\otimes, e$ | 掃描運算子及其單位元素（此處為 $\times$ 和 1） |
| $L$ | 分段長度（2048） |
| $T_c$ | 分段 $c$ 的總乘積 |
| $E_c$ | 傳入分段 $c$ 的互斥進位值：所有先前分段的總乘積 |
| $c(i)$ | 包含 $i$ 的分段 |

## 解題思路

使用與 [cumsum](../cumsum/) 相同的三核心函式**先縮減再掃描**方法，並將運算子作為範本參數（`Times`：單位元素為 1，套用 $a\cdot b$）：

1. `chunkTotals`：每個 256 執行緒的區塊將其 2048 個元素（每個執行緒 8 個）合併為 $T_c$。
2. `scanTotals`：使用一個區塊掃描
   $\lceil N/2048\rceil \le 512$ 個總乘積，得到互斥進位值 $E_c$。
3. `scanChunks`：每個區塊重新讀取其分段並掃描（每個執行緒依序掃描 8 個項目，接著進行 warp shuffle 掃描及 warp 總值掃描），再套用 $E_c$。

所有中間值皆使用 `double`。對乘積而言，這比對總和更重要：fp64 的指數範圍（$10^{\pm308}$）能讓部分乘積在更長的範圍內保持精確，並只在儲存時捨入為 float。

## 成本分析

$$
Q = 4N\ (\text{read}) + 4N\ (\text{re-read}) + 4N\ (\text{write}) = 12N\ \text{bytes}, \qquad \#\text{launches} = 3
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| #launches | 每次呼叫的核心函式啟動次數 |

當 $N = 2^{20}$ 時，$Q = 12.6$ MB，在 2 TB/s 下約需 6 µs，因此啟動額外負擔和中間的單區塊核心函式所需時間，都與資料移動時間相近。解耦回看掃描（單趟，$8N$ 位元組）是目前最先進的替代方案。

## 常見陷阱

- **單位元素是 1**，不是 0：超出 $N$ 的填補通道不可將乘積歸零。
- **fp64 乘法**在消費級 GPU 上很慢（速率為 1/64），但此處每個元素只需執行數次。
- **0 × ∞**：若前綴先達到 $\infty$，而後續元素為 0，PyTorch 和此程式碼都會得到 NaN；檢查器會將 NaN 視為相等。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 延伸閱讀

- [累積總和](../cumsum/)、[一維移動總和](../running-sum-1d/)、
  LeetGPU [前綴和](../../leetgpu/016-prefix-sum/)。
