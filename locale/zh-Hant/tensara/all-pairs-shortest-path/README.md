---
title: 全點對最短路徑
platform: Tensara
upstream: all-pairs-shortest-path
url: https://tensara.org/problems/all-pairs-shortest-path
difficulty: medium
tags: [graph, floyd-warshall, blocked-algorithm, shared-memory]
status: solved
---

# 全點對最短路徑

**平台：** Tensara · **難度：** medium · [題目敘述](https://tensara.org/problems/all-pairs-shortest-path)

## 問題

給定一個以 $n\times n$ 鄰接矩陣表示的稠密加權有向圖，求所有點對之間的最短路徑。權重為正整數；**0 表示「沒有邊」**（對角線除外）；無法抵達的點對必須輸出為 $-1$。測試大小為
$n = 512 \dots 4096$，檢查誤差為 `rtol = 1e-4`、`atol = 1e-3`。演算法和參考實作一樣採用 Floyd–Warshall。與
[LeetGPU 版本](../../leetgpu/073-all-pairs-shortest-paths/)不同之處，在於「沒有邊」和「無法抵達」的編碼方式。

## 公式

$$
d^{(0)}_{ij} = \begin{cases} 0, & i = j \\ +\infty, & a_{ij} = 0 \\ a_{ij}, & \text{otherwise}\end{cases}, \qquad
d^{(k+1)}_{ij} = \min\bigl(d^{(k)}_{ij},\ d^{(k)}_{ik} + d^{(k)}_{kj}\bigr), \qquad
\text{out}_{ij} = \begin{cases} d^{(n)}_{ij}, & d^{(n)}_{ij} < \infty \\ -1, & \text{otherwise}\end{cases}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 頂點數量 |
| $a_{ij}$ | 邊 $i \to j$ 的輸入權重（0 = 不存在） |
| $d^{(k)}_{ij}$ | 只使用編號 $< k$ 的頂點作為中繼點時，$i\to j$ 的最短距離 |
| $\text{out}_{ij}$ | 輸出；無法抵達的點對為 $-1$ |

## 方法

1. **`prepare`**：以網格跨步方式映射。對角線設為 0、零值設為
   $+\infty$，其餘值複製到 `output`。
2. **分塊 Floyd–Warshall**：使用 32 × 32 圖塊，直接在 `output` 中更新。每一輪 $b$ 會執行三個核心函式：對角圖塊（在共享記憶體中進行 32 個相依步驟）、列／欄面板，以及其餘所有圖塊
   （兩個面板的 $(\min, +)$ 乘積，內部不設屏障）。完整推導請見 [LeetGPU 頁面](../../leetgpu/073-all-pairs-shortest-paths/)。
3. **`unreachableToMinusOne`**：將 $+\infty$ 轉成 $-1$。

## 成本分析

$$
W = 2n^3, \qquad Q \approx \frac{n}{32}\cdot 3\cdot 4n^2\ \text{bytes}, \qquad \#\text{launches} = 3\left\lceil\frac{n}{32}\right\rceil + 2
$$

| 符號 | 意義 |
|---|---|
| $W$ | 加法和最小值運算 |
| $Q$ | DRAM 位元組數：每輪讀寫每個圖塊一次，另加兩個面板 |
| #launches | 核心函式啟動次數（$n = 4096$ 時為 384） |

當 $n = 4096$ 時：$W = 1.4\times10^{11}$ 次運算，且 $Q \approx 25$ GB，因此核心函式在第 3 階段受限於運算效能。第 3 階段的暫存器圖塊化（每個執行緒處理多個輸出）是最主要的後續最佳化方向。

## 常見陷阱

- **「0 = 沒有邊」**只適用於非對角線位置；對角線必須是 0。
- **輸出編碼**：$+\infty \to -1$。
- **整數權重**可讓所有不超過 $2^{24}$ 的路徑長度在 float32 中保持精確。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在 [cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 相關內容

- LeetGPU [全點對最短路徑](../../leetgpu/073-all-pairs-shortest-paths/)、
  [最短路徑](../shortest-path/)、[最小生成樹](../min-spanning-tree/)。
