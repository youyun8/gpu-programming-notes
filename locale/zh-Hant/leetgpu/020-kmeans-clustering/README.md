---
title: K-Means 分群
platform: LeetGPU
upstream: hard/20_kmeans_clustering
url: https://leetgpu.com/challenges/k-means-clustering
difficulty: hard
tags: [clustering, atomics, privatization, iterative]
status: solved
---

# K-Means 分群

**平台：** LeetGPU · **難度：** hard · [題目敘述](https://leetgpu.com/challenges/k-means-clustering)

## 問題

對 $n$ 個二維點與 $k$ 個群集執行 $T$ 次 **Lloyd K-Means** 迭代
（$1 \le n \le 10^6$、$1 \le k \le 1000$；基準測試為 $k = 5$、
$T = 30$、$n = 10\,000$）。從指定的初始群心開始，每次迭代都先將每個點
指派給最近的群心，再將各群心移至所屬點的平均位置。輸出為最終群心與
最後一次指派的標籤，容許誤差為 `1e-4`。

## 公式

對迭代 $t = 0, \dots, T-1$：

$$
\ell_i^{(t)} = \arg\min_{0 \le c < k}\ \bigl(x_i - \mu^{(t)}_{c,x}\bigr)^2 + \bigl(y_i - \mu^{(t)}_{c,y}\bigr)^2
$$

$$
\mu^{(t+1)}_c =
\begin{cases}
\dfrac{1}{\lvert S_c\rvert}\displaystyle\sum_{i \in S_c} (x_i, y_i), & \lvert S_c\rvert > 0\\[2mm]
\mu^{(t)}_c, & \lvert S_c\rvert = 0
\end{cases}
\qquad S_c = \{\, i : \ell^{(t)}_i = c \,\}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 點數（`sample_size`） |
| $k$ | 群集數 |
| $T$ | 迭代次數（`max_iterations`） |
| $(x_i, y_i)$ | 第 $i$ 個點的座標 |
| $\mu^{(t)}_c = (\mu_{c,x}, \mu_{c,y})$ | 迭代 $t$ 時的第 $c$ 個群心；$\mu^{(0)}$ = 初始群心 |
| $\ell^{(t)}_i$ | 第 $i$ 個點的標籤（群集索引）；距離相同時選最小的 $c$ |
| $S_c$ | 指派給群集 $c$ 的點集合 |
| $\lvert S_c\rvert$ | 群集大小；空群集維持原群心 |

## 方法

每次迭代使用兩個核心函式。迭代迴圈在主機端執行，資料全程留在 GPU。

### 1. `assignPoints`（≤ 1024 個區塊 × 256 個執行緒）

- 將全部 $k$ 個群心暫存在共享記憶體。每個點都要與每個群心比較，
  因此群心讀取是廣播。
- 以網格跨步迴圈走訪各點，並使用 `__fsub_rn`、`__fmul_rn`、
  `__fadd_rn` 計算 $d = (x-\mu_x)^2 + (y-\mu_y)^2$，因此**不會收縮成
  FMA**。這可逐位元符合 PyTorch 的 `expanded_x**2 + expanded_y**2`。
  嚴格的 `<` 比較會在距離相同時選擇最低索引，與 `argmin` 完全一致。
  若接近平手時選到不同標籤，群心變化會遠大於 `1e-4`。
- **私有化累加。** 區塊將 $x$、$y$ 與 1 加入共享記憶體陣列
  $\Sigma_x, \Sigma_y, \text{cnt}$（float64 原子操作）。同步後，
  每個非空群集只需執行一次全域原子操作來寫回結果。當 $k = 5$、
  $n = 10^4$ 時，每次迭代原本 30 000 次嚴重競爭的全域原子操作，
  可降至最多 $3 \cdot 5 \cdot 40$ 次。

### 2. `updateCentroids`（$k$ 個執行緒）

若 $\text{cnt}_c > 0$，則計算
$\mu_c = \Sigma_c / \text{cnt}_c$。接著將累加器歸零供下一次迭代使用，
以省去一次獨立的 `memset` 啟動。

## 成本分析

$$
W \approx T\,(5nk + 3n), \qquad Q \approx T\,(8n + 4n) \ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 運算數：每次迭代中，每個點與群心的距離約 5 次運算，再加上每個點 3 次累加 |
| $Q$ | 每次迭代的位元組數：讀取座標（每點 8 位元組）並寫入標籤（每點 4 位元組） |

在基準測試規模下，每次迭代的傳輸量約 0.1 MB，遠低於任何吞吐量限制。
成本主要來自 **60 次核心函式啟動**（2 × 30）。CUDA Graph，
或帶有網格屏障的單一常駐核心函式，都能消除大部分開銷。

## 常見問題

- **使用 float32 原子操作累加。** 加總順序每次執行都可能不同；
  當點數達 $10^6$ 時，平均值會漂移。float64 共享記憶體原子操作
  成本低且穩定。
- **空群集**必須保留原群心（不可除以 0）。
- **標籤。** 輸出標籤來自最終群心更新前的*最後一次*指派，
  與參考實作一致。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md) 通過，
包括 $k = 1$、$k > n$（空群集）與重複點（距離相同）。

## 相關內容

- [最近鄰](../038-nearest-neighbor/)、[直方圖統計](../013-histogramming/)（私有化）。
