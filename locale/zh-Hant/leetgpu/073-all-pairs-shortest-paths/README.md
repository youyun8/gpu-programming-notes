---
title: 全點對最短路徑
platform: LeetGPU
upstream: hard/73_all_pairs_shortest_paths
url: https://leetgpu.com/challenges/all-pairs-shortest-paths
difficulty: hard
tags: [graph, floyd-warshall, blocked-algorithm, shared-memory]
status: solved
---

# 全點對最短路徑

**平台：** LeetGPU · **難度：** 困難 · [題目說明](https://leetgpu.com/challenges/all-pairs-shortest-paths)

## 問題

在稠密的 $N\times N$ 距離矩陣上，以 **Floyd–Warshall** 計算全點對最短路徑
（$N \le 4096$，$+\infty$ = 無邊，對角線為零，且沒有負權環；基準測試
$N = 2048$；容許誤差 `1e-2`）。參考實作依序透過每個中繼頂點
$k = 0, \dots, N-1$ 鬆弛所有點對。Floyd–Warshall 與 GEMM 一樣是
$O(N^3)$，但使用 $(\min, +)$ 半環。為提高快取效率而設計的分塊版本，
是經典的 GPU 實作方式。

## 公式

$$
d^{(k+1)}_{ij} = \min\bigl(d^{(k)}_{ij},\ d^{(k)}_{ik} + d^{(k)}_{kj}\bigr), \qquad d^{(0)} = \text{dist}, \qquad \text{output} = d^{(N)}
$$

| 符號 | 意義 |
|---|---|
| $N$ | 頂點數 |
| $d^{(k)}_{ij}$ | 僅使用編號 $< k$ 的中繼頂點時，$i \to j$ 的最短距離 |
| dist | 輸入的鄰接／權重矩陣（沒有邊的位置為 $+\infty$） |

將 $(+, \times)$ 換成 $(\min, +)$，即可把所有 $k$ 上的一個步驟視為
「熱帶」矩陣乘法：$d_{ij} = \min_k (d_{ik} + d_{kj})$。$k$ 迴圈是
**循序的**，因為步驟 $k$ 需要步驟 $k-1$ 的結果。當 $k$ 固定時，
所有 $(i, j)$ 點對則彼此獨立。

### 分塊 Floyd–Warshall

將矩陣切成 $T\times T$ 的 tile（$T = 32$），並以每次 $T$ 個為一組處理
$k$。第 $b$ 輪（$k$ 的範圍為 $bT \dots bT + T - 1$）：

$$
\begin{aligned}
&\text{Phase 1 (diagonal tile):} && D_{bb} \leftarrow \operatorname{FW}(D_{bb}) \\
&\text{Phase 2 (row and column panels):} && D_{bj} \leftarrow \operatorname{FW}_{\text{row}}(D_{bb}, D_{bj}), \quad D_{ib} \leftarrow \operatorname{FW}_{\text{col}}(D_{ib}, D_{bb}) \\
&\text{Phase 3 (all other tiles):} && D_{ij} \leftarrow \min\Bigl(D_{ij},\ D_{ib} \otimes D_{bj}\Bigr)
\end{aligned}
$$

| 符號 | 意義 |
|---|---|
| $T$ | Tile 大小（32） |
| $b$ | 輪次索引，$0 \le b < \lceil N/T\rceil$ |
| $D_{ij}$ | tile 第 $i$ 列、第 $j$ 欄的 tile |
| FW | 限制在單一 tile 內的 $T$ 個循序 Floyd–Warshall 步驟 |
| $\otimes$ | $(\min, +)$ 矩陣乘法：$(X\otimes Y)_{rc} = \min_k (X_{rk} + Y_{kc})$ |

階段 3 的結果是精確的，因為該輪的面板完成後，其餘 tile 只需對此輪區塊
內的 $k$ 取 $\min$，而 $\min$ 具有結合律與交換律。

## 方法

每輪啟動三個核心，執行緒區塊大小為 $32 \times 32$（每個 tile 元素一個
執行緒）：

1. **`phase1`**（1 個區塊）：將對角 tile 載入共享記憶體，執行 32 個步驟。
   每個步驟先計算 `cand`，接著執行一次同步屏障、條件式更新，再執行一次
   同步屏障。第一個屏障可確保所有執行緒都在任何人寫入步驟 $k+1$ 的值
   之前，讀完步驟 $k$ 的值。
2. **`phase2`**（$\lceil N/T\rceil\times 2$ 個區塊）：`blockIdx.y` 選擇列面板
   或欄面板。每個區塊暫存對角 tile 與自己的 tile，再執行 32 個相依步驟。
3. **`phase3`**（$\lceil N/T\rceil^2$ 個區塊）：暫存欄面板 tile $(i, b)$
   與列面板 tile $(b, j)$，每個執行緒在暫存器中計算
   $\min(d, \min_k(\text{col}[y][k] + \text{row}[k][x]))$。此處**沒有內層
   同步屏障**，因為階段 3 的 tile 彼此不相依。

超出範圍的項目會載入為 $+\infty$，因此絕不會成為最小值。

## 成本分析

$$
W = 2N^3\ (\text{add + min}), \qquad
Q_{\text{naive}} \approx N\cdot 3\cdot 4N^2, \qquad
Q_{\text{blocked}} \approx \frac{N}{T}\cdot 3\cdot 4N^2
$$

| 符號 | 意義 |
|---|---|
| $W$ | 運算次數 |
| $Q_{\text{naive}}$ | 每個 $k$ 啟動一個核心時的位元組數（各自讀取第 $k$ 列、第 $k$ 欄，並讀寫整個矩陣） |
| $Q_{\text{blocked}}$ | 分塊後的位元組數：每輪讀寫每個 tile 一次，再加上其兩個面板 |

當 $N = 2048$ 時，$W = 1.7\times10^{10}$ 次運算，而
$Q_{\text{blocked}} \approx 3.2$ GB，相較之下未分塊版本約為 100 GB。
階段 3 佔主要成本，其行為近似 $T = 32$ 的 GEMM。效能受共享記憶體載入與
`fminf` 的計算限制。如同 SGEMM，使用暫存器分塊（每個執行緒計算
2 × 2 或 4 × 4 個輸出）還能進一步加速。

## 常見陷阱

- **階段 1 與 2 內的相依性。** 每個 $k$ 步驟都必須看到上一步的結果，
  因此需要一對同步屏障；階段 3 則不需要。
- **捨入。** 參考實作與此處一樣，精確使用 float32 加法。測試中的路徑總和
  都是可精確表示的整數；其他情況則由 `1e-2` 容許誤差涵蓋結合順序差異。
- **$+\infty + x = +\infty$** 在 IEEE 算術中會正確傳遞，不需特殊處理。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-2`
通過，包括 $N$ 不是 32 的倍數，以及含有 $+\infty$ 結果的不連通圖。

## 相關內容

- [BFS 最短路徑](../046-bfs-shortest-path/)、Tensara [全點對最短路徑](../../tensara/all-pairs-shortest-path/)、
  [矩陣乘法](../002-matrix-multiplication/)（使用 $(+, \times)$ 的對應運算）。
