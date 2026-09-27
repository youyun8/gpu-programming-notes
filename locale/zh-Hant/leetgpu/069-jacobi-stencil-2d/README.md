---
title: 二維 Jacobi 樣板
platform: LeetGPU
upstream: medium/69_jacobi_stencil_2d
url: https://leetgpu.com/challenges/2d-jacobi-stencil
difficulty: medium
tags: [stencil, memory-bound, pde]
status: solved
---

# 二維 Jacobi 樣板

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/2d-jacobi-stencil)

## 題意

在 `rows × cols` 的 float32 網格上執行一次五點 Laplace 樣板的 Jacobi
掃描（$\le 16\,384^2$；基準測試 $8192^2$；容許誤差 `1e-5`）。內部儲存格
變為其 4 個相鄰儲存格的平均值，邊界儲存格則直接複製。Jacobi 迭代可求解
Laplace／Poisson 方程式（熱擴散、靜電學）。此樣板是典型的
**記憶體受限、高重用率**存取模式。

## 圖解

![Jacobi 五點模板：每個內部格點變成四個鄰居的平均](figure.svg)

四個藍色格是標示輸出格的鄰居，中心點本身不參與計算。輸出寫到另一個網格，因此不會讀到已經更新過的值。

## 數學表述

$$
u'_{ij} = \begin{cases}
\frac14\bigl(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1}\bigr), & 0 < i < R-1,\ 0 < j < C-1 \\
u_{ij}, & \text{otherwise (boundary)}
\end{cases}
$$

| 符號 | 意義 |
|---|---|
| $R,\ C$ | 網格的列數與欄數 |
| $u_{ij}$ | 第 $i$ 列、第 $j$ 欄的輸入值（偏移量 $iC + j$） |
| $u'_{ij}$ | 輸出值 |

這是離散 Laplace 方程式 $\nabla^2 u = 0$ 在 Dirichlet 邊界條件下，迭代式
$u^{(t+1)} = D^{-1}(b - (L + U)u^{(t)})$ 的一步。輸入絕不會原地更新，
這正是 Jacobi 與 Gauss–Seidel 的差異。

## 解題思路

- 使用由 $32 \times 8$ 區塊組成的二維網格，每個儲存格一個執行緒，
  `threadIdx.x` 沿著欄方向。
- 邊界執行緒直接複製。內部執行緒載入 4 個相鄰值，並儲存
  $0.25\cdot\bigl(((u_{\uparrow} + u_{\downarrow}) + u_{\leftarrow}) + u_{\rightarrow}\bigr)$，
  其加總順序與參考實作的四項加總相同。
- `__restrict__` 告訴編譯器 `in` 與 `out` 不會互相別名，因此載入可透過
  唯讀路徑快取。

### 不使用共享記憶體的重用

每個輸入值最多會被 5 個執行緒讀取：自身、左、右、上、下。在一個 warp
內，左、中、右的讀取會落在相同的快取列（L1 命中）。上、下兩列則剛被
相鄰區塊列讀取過（L2 命中）。因此 DRAM 流量仍接近每個儲存格 1 次讀取加
1 次寫入；若明確使用帶有 halo 的共享記憶體 tile，主要只會省下 L1/L2
交易，而非 DRAM 位元組。

## 成本分析

$$
Q_{\min} = 8RC\ \text{bytes}, \qquad W = 4RC, \qquad I = \frac{4}{8} = 0.5\ \text{FLOP/byte}
$$

| 符號 | 意義 |
|---|---|
| $Q_{\min}$ | 必要的 DRAM 位元組數：每個儲存格各讀取、寫入一次 |
| $W$ | 每個內部儲存格 3 次加法加 1 次乘法 |
| $I$ | 算術強度 |

基準測試為 537 MB，在 2 TB/s 下約需 270 µs。迭代求解器可透過
**時間阻塞**超越單次掃描：tile 留在共享記憶體期間執行多個時間步驟，
使 $I$ 成比例提高。

## 常見陷阱

- **原地更新**（`in == out`）會讓 Jacobi 變成含有競爭條件、定義不明的
  Gauss–Seidel。
- **退化網格。** 當 `rows` 或 `cols` 小於 3 時，所有儲存格都是邊界。
  索引檢查能處理 1 × N 網格。
- 最後的 `0.25 * sum` 發生 **FMA 合併**也無妨，因為乘以 2 的冪次是精確的。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上通過，
包括 1 × 1、1 × N 和 2 × 2 網格。

## 延伸閱讀

- [二維卷積](../010-2d-convolution/)、[高斯模糊](../028-gaussian-blur/)。
