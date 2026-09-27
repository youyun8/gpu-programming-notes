---
title: BFS 最短路徑
platform: LeetGPU
upstream: hard/46_bfs_shortest_path
url: https://leetgpu.com/challenges/bfs-shortest-path
difficulty: hard
tags: [graph, bfs, persistent-kernel, atomics]
status: solved
---

# BFS 最短路徑

**平台：** LeetGPU · **難度：** 困難 · [題目敘述](https://leetgpu.com/challenges/bfs-shortest-path)

## 題意

在帶有障礙物的 `rows × cols` 網格中，求兩個可通行儲存格之間的最短路徑
長度（上下左右移動的步數）；若無法到達則回傳 $-1$
（$\text{rows}, \text{cols} \le 1000$；效能評測使用 $500\times500$）。
答案必須完全相符。BFS 的執行不規則（前緣大小變化很大），各層之間必須循序
處理，而且迷宮可能有 $O(\text{rows}\cdot\text{cols})$ 層。
這三項特性都不利於 GPU。

## 圖解

![網格上的 BFS：第 ℓ 層前沿包含所有與起點距離為 ℓ 的格子](figure.svg)

格內數字是與紅色起點的 BFS 距離，深色格是障礙物。交替的顏色代表相鄰的前沿層，終點（右下）的距離標在網格右側。

## 數學表述

將網格建模為圖 $G = (V, E)$，以可通行儲存格作為頂點，並以四鄰接關係作為
邊。BFS 逐層計算距離：

$$
F_0 = \{s\}, \qquad
F_{\ell+1} = \bigl\{\, v \in V \setminus (F_0 \cup \dots \cup F_\ell) \ :\ \exists\, u \in F_\ell,\ (u, v) \in E \,\bigr\}, \qquad
d(s, t) = \min\{\ell : t \in F_\ell\}
$$

| 符號 | 意義 |
|---|---|
| $V$ | 可通行儲存格（網格值為 0），索引為 $r\cdot\text{cols} + c$ |
| $E$ | 上／下／左／右相差一步的可通行儲存格配對 |
| $s,\ t$ | 起點與目標儲存格 |
| $F_\ell$ | 前緣：與 $s$ 的距離恰為 $\ell$ 的儲存格 |
| $d(s,t)$ | 最短路徑長度；若永遠無法抵達 $t$（前緣變空），則為 $-1$ |

每個儲存格只會進入一個前緣，因此總工作量為
$O(\lvert V\rvert + \lvert E\rvert) = O(\text{rows}\cdot\text{cols})$。

## 解題思路

### 在單一常駐區塊中執行逐層同步 BFS

- 使用一個含 1024 個執行緒的區塊執行完整搜尋。兩個全域陣列分別保存目前
  與下一個前緣，而全域 `visited` 對應表則為每個儲存格保存一個 int。
- 每一層執行：
  1. 執行緒以網格跨步方式走訪目前前緣。每個儲存格最多計算四個界內鄰居，
     並略過障礙物。
  2. `atomicCAS(&visited[nb], 0, 1) == 0` 會**認領**鄰居。每個儲存格
     只有一個執行緒能成功，因此只會透過共享計數器 `s_next_size` 上的
     `atomicAdd` 附加一次。
  3. 若認領的儲存格就是目標，設定 `s_found = 1`。
  4. 執行 `__syncthreads()`、交換佇列、增加層數，再次執行
     `__syncthreads()`。
- 找到目標或前緣變空時，迴圈就會停止。

### 為何只用一個區塊？

分隔各層需要屏障。若要在整張 GPU 上同步，就必須每層啟動一個核心函式
（每次約 3–5 µs），或使用 cooperative-groups 網格同步。
$500 \times 500$ 的蛇形迷宮約有 125 000 層，光是啟動成本就會超過
0.5 秒。在單一區塊內，屏障是只需數十奈秒的 `__syncthreads()`。
雖然只使用一個 SM，但稀疏前緣通常只有數十到數百個儲存格，
此時單一 SM 並非瓶頸。

## 成本分析

$$
W = O(\lvert V\rvert), \qquad T \approx L\cdot t_{\text{level}} + \frac{4\lvert V\rvert}{\text{throughput}_{\text{SM}}}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 總工作量：每個儲存格展開一次，並檢查 4 個鄰居 |
| $L$ | BFS 層數（$= d(s,t)$，或 $s$ 的離心率） |
| $t_{\text{level}}$ | 每層的固定成本（兩次屏障加上共享記憶體簿記） |
| Throughput$_{\text{SM}}$ | 單一 SM 每秒可執行的鄰居檢查數，受原子操作與全域記憶體延遲限制 |

開放網格的 $L \approx$ 列數 + 欄數，且前緣很寬；迷宮的 $L$ 很大，
但前緣很窄。常駐區塊不需主機端介入，就能處理這兩種情況。

## 常見陷阱

- **不使用原子操作檢查是否造訪。** 兩個執行緒可能同時看到
  `visited == 0`，並將同一個儲存格加入佇列兩次。結果仍然正確，
  但工作量可能暴增。
- **$s = t$** 時不需搜尋，直接回傳 0。
- **提早結束。** 目標可在被*認領*時偵測，比展開它早一層。
  層數計數器會在屏障後增加，因此回報的距離仍然精確。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 中完全相符，
包括無法抵達的目標、$s = t$、$1 \times 1$ 網格與狹長的蛇形迷宮。

## 延伸閱讀

- [全點對最短路徑](../073-all-pairs-shortest-paths/)、
  Tensara [最短路徑](../../tensara/shortest-path/)、
  [串流壓縮](../072-stream-compaction/)（使用掃描而非原子操作建立前緣）。
