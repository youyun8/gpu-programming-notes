---
title: 普通最小平方法
platform: LeetGPU
upstream: medium/33_ordinary_least_squares
url: https://leetgpu.com/challenges/ordinary-least-squares
difficulty: medium
tags: [linear-algebra, cholesky, normal-equations, fp64]
status: solved
---

# 普通最小平方法

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/ordinary-least-squares)

## 題意

以最小平方法擬合線性模型：給定 $X \in \mathbb R^{n\times f}$ 與
$\mathbf y \in \mathbb R^n$（float32；$n \le 10^5$、$f \le 1000$、
$n \ge f$，且 $X$ 滿秩；基準測試為 $n = f = 32$），傳回係數向量
$\boldsymbol\beta$，並要求 `atol = rtol = 1e-2`。參考實作會使用
Cholesky 分解來解**正規方程式**。這是 GPU 上的小型密集線性代數管線：
先計算 Gram 矩陣（類似 GEMM），再進行分解
（沿 $k$ 循序執行，每一步內部平行），最後解兩次三角方程組。

## 圖解

![以正規方程式解最小平方法：Gram 矩陣、Cholesky 分解、兩次三角求解](figure.svg)

流程由左至右閱讀。下方是涉及的矩陣形狀：稠密且對稱的 Gram 矩陣 G，以及三角因子 L 與 Lᵀ（灰色半邊為 0）。

## 數學表述

$$
\boldsymbol\beta = \arg\min_{\boldsymbol\beta} \lVert X\boldsymbol\beta - \mathbf y\rVert_2^2
\quad\Longleftrightarrow\quad
\underbrace{X^{\mathsf T}X}_{G}\ \boldsymbol\beta = \underbrace{X^{\mathsf T}\mathbf y}_{\mathbf b}
$$

$$
G = LL^{\mathsf T}, \qquad L\mathbf z = \mathbf b\ \ (\text{forward}), \qquad L^{\mathsf T}\boldsymbol\beta = \mathbf z\ \ (\text{backward})
$$

| 符號 | 意義 |
|---|---|
| $n$ | 樣本數（$X$ 的列數） |
| $f$ | 特徵數（$X$ 的欄數） |
| $X$ | 特徵矩陣，採列優先格式；元素 $X_{sj}$ 位於 $sf + j$ |
| $\mathbf y$ | 目標向量 |
| $\boldsymbol\beta$ | 係數（輸出） |
| $\lVert\cdot\rVert_2$ | 歐幾里得範數 |
| $G$ | Gram 矩陣 $X^{\mathsf T}X$（$f\times f$；滿秩 $X$ 會使其對稱正定） |
| $\mathbf b$ | 方程式右側的 $X^{\mathsf T}\mathbf y$ |
| $L$ | 下三角 Cholesky 因子 |
| $\mathbf z$ | 前向求解的中間向量 |

### 右看式 Cholesky，第 $k = 0 \dots f-1$ 步

$$
L_{kk} = \sqrt{G_{kk}}, \qquad L_{ik} = \frac{G_{ik}}{L_{kk}}\ (i > k), \qquad G_{ij} \leftarrow G_{ij} - L_{ik}L_{jk}\ (j \le i,\ i, j > k)
$$

| 符號 | 意義 |
|---|---|
| $k$ | 目前的樞紐欄 |
| $G_{ij}$ | 尾端子矩陣，就地更新（只更新下三角） |
| $L_{ik}$ | $L$ 的第 $k$ 欄，覆寫 $G$ 的該欄 |

## 解題思路

1. **`gramTiled`**：以分塊的「$A^{\mathsf T}A$」GEMM 計算
   $G = X^{\mathsf T}X$。一個 $16\times16$ 區塊計算 $G$ 的一個
   $16\times16$ 分塊。它以 16 為切片大小走訪樣本維度，將
   $X[s_0{:}s_0{+}16,\ i_0{:}i_0{+}16]$ 與
   $X[s_0{:}s_0{+}16,\ j_0{:}j_0{+}16]$ 暫存在共享記憶體
   （填補至 17 以避免 bank 衝突），並使用 **float64** 累加。
2. **`xtY`**：每個特徵 $j$ 使用一個執行緒走訪各樣本。
   連續執行緒讀取連續的 $X_{sj}$，因此能合併存取。
3. **`choleskySolve`**（一個含 1024 個執行緒的區塊）：
   - 每個 $k$ 先由執行緒 0 計算樞紐，接著所有執行緒縮放第 $k$ 欄，
     再由所有執行緒執行秩 1 尾端更新（將 $(i, j)$ 攤平）。
     每個階段後都有屏障。
   - 前向與反向代換：對每一列執行全區塊歸約，以計算
     $\sum_j L_{ij}z_j$。

### 為何使用 float64

形成正規方程式時，條件數會平方：

$$
\kappa_2(X^{\mathsf T}X) = \kappa_2(X)^2
$$

| 符號 | 意義 |
|---|---|
| $\kappa_2(\cdot)$ | 2-範數條件數（最大奇異值與最小奇異值之比） |

當輸入範圍達 $\pm1000$ 且特徵為隨機值時，$\kappa(X)$ 約為
$10^2$–$10^3$，使 $\kappa(G)$ 達 $10^4$–$10^6$。
這會耗盡 float32 約 $\sim10^{-7}$ 的精度，但仍遠低於 float64
約 $\sim10^{-16}$ 的精度限制。

## 成本分析

$$
W_G = 2nf^2, \qquad W_{\text{chol}} \approx \frac{f^3}{3}, \qquad W_{\text{solve}} = 2f^2, \qquad \text{sync steps} = O(f)
$$

| 符號 | 意義 |
|---|---|
| $W_G$ | 形成 $G$ 的 FLOP 數（利用對稱性只需一半，但此處兩半都會計算） |
| $W_{\text{chol}}$ | Cholesky FLOP 數 |
| $W_{\text{solve}}$ | 兩次三角方程組求解 |
| Sync steps | 分解本質上必須沿 $k$ 循序執行，每步約有 3 個屏障 |

當 $n = 10^5$、$f = 1000$ 時，
$W_G = 2\times10^{11}$（float64），是主要成本。對基準測試
（$32 \times 32$）而言，所有工作都受限於延遲。對小型 $f$，
含 $O(f)$ 個屏障的單一區塊很合適。大型 $f$ 則應使用分塊、多區塊的
Cholesky（如 cuSOLVER）。

## 常見陷阱

- **float32 Gram 矩陣。** 在條件不佳的輸入上，可能無法達到
  `1e-2` 容許誤差。
- **上三角。** 只有下三角會被更新與讀取。`gram` 的上半部保留舊值，
  求解過程完全不會使用。
- **暫存緩衝區**（$f^2 + 2f$ 個 double）會在每次呼叫時配置，
  同步後再釋放。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
以 `1e-2` 容許誤差通過，包括 $n = f$（方形系統）與 $f = 1$。

## 延伸閱讀

- [邏輯斯迴歸](../034-logistic-regression/)
  （Newton 法內使用相同的 Gram + Cholesky）。
