---
title: 批次正規化
platform: LeetGPU
upstream: medium/40_batch_normalization
url: https://leetgpu.com/challenges/batch-normalization
difficulty: medium
tags: [normalization, column-reduction, welford, fp64]
status: solved
---

# 批次正規化

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/batch-normalization)

## 題意

對 $N \times C$ 輸入執行 BatchNorm 正向傳播（訓練模式）：以批次平均值
與有偏變異數正規化每一**欄**（通道），再以可學習的 $\gamma, \beta$
縮放與平移（$N \le 10^4$、$C \le 1024$、$\varepsilon = 10^{-5}$；
效能評測使用 $N = 5000$；容許誤差為 `1e-5`）。統計值是對列優先矩陣
執行*欄*歸約，因此會決定執行緒配置。

## 圖解

![BatchNorm：沿批次方向，為每一欄（通道）計算統計量](figure.svg)

標示的欄位代表一個通道。先對所有 N 列計算它的平均與變異數，再用來正規化這一欄；各通道彼此獨立。

## 數學表述

$$
\mu_j = \frac1N\sum_{i=0}^{N-1} x_{ij}, \qquad
\sigma_j^2 = \frac1N\sum_{i=0}^{N-1} (x_{ij} - \mu_j)^2, \qquad
y_{ij} = \gamma_j\,\frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \varepsilon}} + \beta_j
$$

| 符號 | 意義 |
|---|---|
| $N$ | 批次大小（列數） |
| $C$ | 通道數（欄數） |
| $x_{ij}$ | 輸入的第 $i$ 列、第 $j$ 個通道（偏移為 $iC + j$） |
| $\mu_j$ | 通道 $j$ 的批次平均值 |
| $\sigma_j^2$ | 有偏批次變異數（除以 $N$，而非 $N-1$） |
| $\varepsilon$ | 數值穩定常數（$10^{-5}$） |
| $\gamma_j,\ \beta_j$ | 可學習的縮放與平移參數 |
| $y_{ij}$ | 輸出 |

### Welford 線上更新與 Chan 合併法

若累加 $\sum x$ 與 $\sum x^2$，再使用 $E[x^2] - E[x]^2$，
當 $\lvert\mu\rvert \gg \sigma$ 時會發生災難性消去。Welford 演算法會
維護持續更新的平均值與離均差平方和：

$$
n \leftarrow n + 1, \quad \delta = x - \bar x, \quad \bar x \leftarrow \bar x + \frac{\delta}{n}, \quad M_2 \leftarrow M_2 + \delta\,(x - \bar x)
$$

兩個部分狀態 $(n_a, \bar x_a, M_{2,a})$ 與
$(n_b, \bar x_b, M_{2,b})$ 的合併方式為

$$
n = n_a + n_b, \quad \delta = \bar x_b - \bar x_a, \quad \bar x = \bar x_a + \delta\frac{n_b}{n}, \quad M_2 = M_{2,a} + M_{2,b} + \delta^2 \frac{n_a n_b}{n}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 納入狀態的樣本數 |
| $\bar x$ | 持續更新的平均值 |
| $M_2$ | 相對平均值的離均差平方和；$\sigma^2 = M_2 / n$ |
| $\delta$ | 新樣本（或部分平均值）與目前平均值之差 |

## 解題思路

1. **`channelStats`**：使用 $32 \times 8$ 個執行緒的區塊，每個區塊
   負責連續 32 個通道。
   - `threadIdx.x` 代表通道。一個 warp 會讀取同一列中連續的 32 個
     float，因此能合併存取。
   - `threadIdx.y` $\in 0..7$ 跨步走訪各列：列群組 $g$ 處理
     $g, g+8, \dots$ 列，並以 float64 執行 Welford 更新。
   - 每個通道的 8 個部分狀態會放入共享記憶體，再由列群組 0 使用
     Chan 公式合併。它會寫入 $\mu_j$ 與
     $\text{rstd}_j = 1/\sqrt{\sigma_j^2 + \varepsilon}$。
2. **`normalize`**：以網格跨步的逐元素核心函式計算
   $y = \gamma_j\,((x - \mu_j)\cdot\text{rstd}_j) + \beta_j$，
   其中 $j = i \bmod C$。

## 成本分析

$$
Q = \underbrace{4NC}_{\text{stats}} + \underbrace{4NC + 4NC}_{\text{normalize}} = 12NC \ \text{bytes}, \qquad T_{\min} = \frac{12NC}{\beta_{\text{mem}}}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取 $x$ 兩次、寫入 $y$ 一次 |
| $\beta_{\text{mem}}$ | DRAM 頻寬（如此命名以免與 BN 的平移參數 $\beta$ 混淆） |

當 $N = 5000$、$C = 1024$ 時約為 61 MB，也就是在 2 TB/s 下約需
30 µs。統計核心函式只有 $\lceil C/32\rceil = 32$ 個區塊，無法充分利用
大型 GPU。若將各通道群組的列拆分給數個區塊（再以第二輪合併），
即可提高平行度。

## 常見陷阱

- **有偏與無偏變異數。** 參考實作使用 `unbiased=False`，也就是除以 $N$。
- **單輪 $\sum x^2$ 公式**會在數值相對於分散程度很大時失去精度。
- 當 $N < 8$ 時會出現**空的列群組**，合併時必須略過（計數為 0）。

## 驗證

所有 LeetGPU 測試案例均以 `1e-5` 的容許誤差在
[cuemu](../../tools/cuemu/README.md) 通過，包括 $N = 1$（變異數為 0）
與 $C$ 不是 32 倍數的情況。

## 延伸閱讀

- [RMS 正規化](../050-rms-normalization/)、[層正規化](../113-layer-normalization/)、
  [群組正規化](../105-group-normalization/)。Tensara [批次正規化](../../tensara/batch-norm/)。
