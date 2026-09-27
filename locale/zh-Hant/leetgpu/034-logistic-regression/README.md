---
title: 邏輯斯迴歸
platform: LeetGPU
upstream: medium/34_logistic_regression
url: https://leetgpu.com/challenges/logistic-regression
difficulty: medium
tags: [optimization, newton, irls, cholesky, fp64]
status: solved
---

# 邏輯斯迴歸

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/logistic-regression)

## 題意

為二元邏輯斯迴歸進行擬合：$X \in \mathbb R^{n\times f}$，標籤為
$y_i \in \{0, 1\}$（$n \le 10^5$、$f \le 1000$；效能評測使用 $n = 16$、$f = 8$；
容許誤差為 `1e-2`）。參考實作會以很小的 L2 項執行
**Newton–Raphson（IRLS）**，直到步長範數低於 $10^{-8}$。若要讓結果在
`1e-2` 內符合其固定點，基本上必須執行相同的演算法。對接近線性可分的資料，
一般梯度下降的收斂速度太慢。

## 圖解

![以牛頓法（IRLS）求解邏輯斯迴歸：重複迭代直到步長極小](figure.svg)

一列方框就是一次牛頓迭代；紅色迴圈會重複執行，直到 ‖Δ‖ < 10⁻⁸。下方小圖是把 xᵀβ 轉成預測機率 p 的 sigmoid 函數。

## 數學表述

最大化對數概似，等價於最小化其負值再加上一個很小的脊迴歸項：

$$
J(\boldsymbol\beta) = -\sum_{i=0}^{n-1}\Bigl[y_i\log p_i + (1-y_i)\log(1-p_i)\Bigr] + \frac{\lambda}{2}\lVert\boldsymbol\beta\rVert^2,
\qquad p_i = \sigma(\mathbf x_i^{\mathsf T}\boldsymbol\beta) = \frac{1}{1 + e^{-\mathbf x_i^{\mathsf T}\boldsymbol\beta}}
$$

| 符號 | 意義 |
|---|---|
| $n,\ f$ | 樣本數與特徵數 |
| $\mathbf x_i$ | $X$ 的第 $i$ 列（樣本 $i$ 的特徵） |
| $y_i$ | 樣本 $i$ 的標籤，為 0 或 1 |
| $\boldsymbol\beta$ | 係數（輸出，長度為 $f$） |
| $\sigma$ | 邏輯斯 sigmoid 函數 |
| $p_i$ | $y_i = 1$ 的預測機率 |
| $\lambda$ | L2 正則化，$10^{-6}$ |
| $J$ | 目標函數（負對數概似加脊迴歸項） |

### Newton / IRLS 步驟

$$
\mathbf g = X^{\mathsf T}(\mathbf p - \mathbf y) + \lambda\boldsymbol\beta, \qquad
H = X^{\mathsf T} W X + \lambda I, \qquad W = \operatorname{diag}\bigl(\max(p_i(1-p_i),\ 10^{-8})\bigr), \qquad
\boldsymbol\beta \leftarrow \boldsymbol\beta - H^{-1}\mathbf g
$$

當 $\lVert H^{-1}\mathbf g\rVert_2 < 10^{-8}$ 時停止（最多 1000 次迭代）。

| 符號 | 意義 |
|---|---|
| $\mathbf g$ | $J$ 的梯度 |
| $H$ | $J$ 的 Hessian 矩陣（因為有 $\lambda I$ 與截限，所以是對稱正定矩陣） |
| $W$ | 對角權重 $p_i(1-p_i)$，截限使其不會接近 0 |
| $I$ | $f \times f$ 單位矩陣 |
| $H^{-1}\mathbf g$ | Newton 步驟，以 Cholesky 求解計算（絕不明確計算反矩陣） |

Newton 法在最佳解附近會以**二次速度**收斂，因此通常 5–15 次迭代就足夠。

## 解題思路

每次迭代使用四個核心函式，全部採用 float64：

1. **`sampleTerms`**（每個樣本一個 warp）：透過 shuffle 歸約計算
   $z_i = \mathbf x_i^{\mathsf T}\boldsymbol\beta$。執行緒束中的第 0 個
   執行緒會寫入 $W_i$ 與殘差 $r_i = p_i - y_i$。
2. **`weightedGram`**：計算 $H = X^{\mathsf T}WX + \lambda I$。這是
   [OLS](../033-ordinary-least-squares/) 的分塊 Gram 核心函式，但會先將
   $A$ 側的資料塊乘上 $W_s$。
3. **`gradient`**：計算
   $g_j = \sum_s X_{sj} r_s + \lambda\beta_j$，每個特徵使用一個執行緒。
4. **`newtonStep`**（一個執行緒區塊）：就地對 $H$ 做 Cholesky 分解，
   以前向與反向代入求出 $\boldsymbol\delta$，執行
   $\boldsymbol\beta \mathrel{-}= \boldsymbol\delta$，再以區塊歸約計算
   $\lVert\boldsymbol\delta\rVert^2$。

每次迭代只會將單一純量 $\lVert\boldsymbol\delta\rVert^2$ 複製到主機端，
以判斷是否停止。$\boldsymbol\beta$ 會留在裝置端，最後才轉為 float32。

## 成本分析

$$
\text{per iteration:}\quad W \approx \underbrace{2nf}_{z} + \underbrace{2nf^2}_{H} + \underbrace{2nf}_{\mathbf g} + \underbrace{f^3/3}_{\text{Cholesky}}, \qquad \text{total} \approx T_{\text{it}}\cdot W
$$

| 符號 | 意義 |
|---|---|
| $W$ | 每次 Newton 迭代的 FLOP 數 |
| $T_{\text{it}}$ | 迭代次數（約 10） |

效能評測（$16 \times 8$）完全受延遲限制：約 10 次迭代 ×
（4 次核心函式啟動 + 1 次小型裝置到主機複製）。若以常駐的單區塊核心函式
在裝置端執行完整迴圈，就能省去與主機端往返的成本。

## 常見陷阱

- **停止條件。** 參考實作是依據*步長*範數停止，而非梯度範數。
  使用相同條件可讓結果差異遠低於容許誤差。
- **線性可分資料。** 若沒有 $\lambda$ 與對 $W$ 的截限，當
  $p_i \to 0/1$ 時，$H$ 可能變成奇異矩陣。
- **Float32。** 在 $10^{-8}$ 的步長門檻下，使用 float32 會使收斂失敗。

## 驗證

所有 LeetGPU 測試案例均以 `1e-2` 的容許誤差在
[cuemu](../../tools/cuemu/README.md) 通過，包括近乎線性可分的資料。

## 延伸閱讀

- [普通最小平方法](../033-ordinary-least-squares/)、[類別交叉熵](../025-categorical-cross-entropy-loss/)。
