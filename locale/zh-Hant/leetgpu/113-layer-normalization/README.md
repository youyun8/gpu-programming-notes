---
title: 層正規化
platform: LeetGPU
upstream: medium/113_layer_normalization
url: https://leetgpu.com/challenges/layer-normalization
difficulty: medium
tags: [normalization, row-reduction, warp-per-row, transformer]
status: solved
---

# 層正規化

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/layer-normalization)

## 題意

對 $N\times C$ float32 矩陣執行 LayerNorm 前向傳播：每列（樣本或權杖）會在其 $C$ 個特徵上正規化，再以每個特徵的權重與偏置進行縮放和平移（$N \le 65\,536$、$C \le 4096$、$\varepsilon = 10^{-5}$；效能測試為 $N = 65\,536$、$C = 512$；容許誤差 `1e-4`）。

## 圖解

![LayerNorm：每一列在其 C 個特徵上計算統計量，再套用逐特徵的仿射轉換](figure.svg)

標示的列代表一個 token。它的平均與變異數只來自自己的 C 個值，而逐特徵的 w 與 b 由所有列共用。

## 數學表述

$$
\mu_i = \frac1C\sum_{j=0}^{C-1} x_{ij}, \qquad
\sigma^2_i = \frac1C\sum_{j=0}^{C-1}(x_{ij} - \mu_i)^2, \qquad
y_{ij} = w_j\,\frac{x_{ij} - \mu_i}{\sqrt{\sigma^2_i + \varepsilon}} + b_j
$$

| 符號 | 意義 |
|---|---|
| $N,\ C$ | 資料列數與特徵數 |
| $x_{ij}$ | 輸入 |
| $\mu_i,\ \sigma^2_i$ | 資料列平均值與有偏變異數 |
| $\varepsilon$ | 穩定常數 |
| $w_j,\ b_j$ | 每個特徵的縮放與平移（由所有資料列共用） |
| $y_{ij}$ | 輸出 |

### 為何變異數需要兩次走訪

當 $\lvert\mu\rvert \gg \sigma$ 時，單次走訪公式 $\sigma^2 = E[x^2] - \mu^2$ 會將兩個很大且極為接近的數相減。相對誤差約為 $u\cdot\mu^2/\sigma^2$，其中 $u$ 是 float32 的單位捨入誤差。若輸入最大為 100 且分布範圍很小，這會在 float32 中徹底破壞結果。中心化形式 $\frac1C\sum(x - \mu)^2$ 則沒有消去誤差。

## 解題思路

**每列使用一個 warp**（每個含 256 個執行緒的區塊處理 8 列）：

1. 第一次走訪：各 lane 以跨距方式走訪資料列（合併存取）、加總，並執行蝶形歸約，讓每個 lane 都能取得 $\mu$。
2. 第二次走訪：以相同方式歸約 $\sum (x - \mu)^2$，接著計算 $\text{rstd} = \texttt{rsqrtf}(\sigma^2 + \varepsilon)$。
3. 第三次走訪：寫入 $w_j\bigl((x - \mu)\,\text{rstd}\bigr) + b_j$。

資料列（$\le 16$ KB）在三次走訪期間都會留在 L1，因此 DRAM 流量大約是讀取一次、寫入一次。每列一個 warp 可完全避免共享記憶體與 `__syncthreads()`，而 $N = 65\,536$ 列也提供充足的平行度。

## 成本分析

$$
Q \approx 8NC + 8C\ \text{bytes}, \qquad W \approx 8NC, \qquad T_{\min} = \frac{8NC}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取 $x$ 一次（後續走訪命中 L1）、寫入 $y$，另加權重 |
| $W$ | 浮點運算次數 |
| $\beta$ | DRAM 頻寬 |

效能測試：268 MB，亦即在 2 TB/s 下約為 134 µs。

## 常見陷阱

- 此處若使用**無偏變異數**（$C - 1$）會出錯。
- 當 $C$ **非常大**時（例如 16K 以上），每列一個 warp 將無法讓資料常駐 L1，而且每列的平行度太低。此時應改為每列一個區塊。

## 驗證

所有 LeetGPU 測試案例都以 `1e-4` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $C = 1$（輸出 = 偏置）。

## 延伸閱讀

- [RMS 正規化](../050-rms-normalization/)、[群組正規化](../105-group-normalization/)、[批次正規化](../040-batch-normalization/)、[權杖嵌入](../106-token-embedding-layer/)。Tensara [層正規化](../../tensara/layer-norm/)。
