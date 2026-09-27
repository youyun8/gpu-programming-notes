---
title: 串流壓縮
platform: LeetGPU
upstream: medium/72_stream_compaction
url: https://leetgpu.com/challenges/stream-compaction
difficulty: medium
tags: [scan, compaction, filter]
status: solved
---

# 串流壓縮

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/stream-compaction)

## 題意

穩定的**串流壓縮**：將陣列 $A$（長度 $N \le 10^8$）中的每個正數元素
依原順序複製到 `out` 前端，其餘位置填入 0（基準測試
$N = 5\times10^7$；結果須完全精確）。壓縮（「篩選」）可在 GPU 上建立
工作清單，例如作用中的光線、存活的粒子、非零項目及 BFS 前緣。

## 圖解

![串流壓縮：先算謂詞，再做排他式掃描，最後分散寫出](figure.svg)

每一排都標示出被保留的元素（A > 0）。排他式掃描 o 提供每個保留元素的目的地，因此連線不會交叉，順序也得以保持。

## 數學表述

$$
p_i = [\,A_i > 0\,], \qquad o_i = \sum_{j < i} p_j \ \ (\text{exclusive scan}), \qquad
\text{out}_{o_i} = A_i\ \ \text{whenever } p_i = 1, \qquad
\text{out}_{k..N-1} = 0,\ \ k = \sum_j p_j
$$

| 符號 | 意義 |
|---|---|
| $N$ | 輸入長度 |
| $A_i$ | 輸入值 |
| $p_i$ | 述詞（保留時為 1） |
| $o_i$ | 元素 $i$ 的輸出位置：它之前被保留的元素數 |
| $k$ | 被保留的元素總數 |
| out | 壓縮後的輸出 |

述詞的排他掃描會為每個保留元素提供一個**唯一且保持順序**的目的位置，
這正是穩定性的要求。

## 解題思路

對每 2048 個元素的區塊（256 個執行緒 × 8 個項目）先歸約再掃描：

1. **`chunkCounts`**：計算每個區塊中述詞成立的數量（區塊歸約）。
2. **`scanCounts`**（1 個區塊）：對區塊計數做排他掃描，產生各區塊的偏移量；
   同時也在裝置記憶體中產生總數 $k$。
3. **`scatter`**：每個執行緒計算其 8 個連續項目中符合條件的數量。對這些
   計數做區塊排他掃描，再加上區塊偏移量，即得該執行緒的第一個輸出位置。
   接著走訪自己的 8 個項目，連續寫入要保留的項目。
4. **填零**：將位置 $[k, N)$ 設為 0。測試框架聲稱 `out` 已預先初始化，
   但填零可讓核心不必依賴此條件。

## 成本分析

$$
Q = \underbrace{4N}_{\text{count}} + \underbrace{4N}_{\text{scatter read}} + \underbrace{4k + 4(N-k)}_{\text{writes}} = 12N\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| $k$ | 被保留的元素數 |

基準測試為 600 MB，在 2 TB/s 下約需 0.3 ms。每個執行緒的散佈寫入是連續的，
但整個 warp 並不連續（各執行緒寫入的元素數量不同），因此寫入只能部分合併。
warp 層級的變體會使用 `__ballot_sync` 和 `__popc`，為每個 lane 指派 warp
內的位置，並以一條指令寫入 32 個連續的保留值。

## 常見陷阱

- **零不是正數。** `A[i] = 0.0` 會被捨棄（使用 `> 0`，而非 `>= 0`）。
- **穩定性。** 原子操作（對全域游標使用 `atomicAdd`）雖較簡單，卻會改變
  輸出順序。參考實作要求保留原始順序。

## 驗證

所有 LeetGPU 測試案例皆與 [cuemu](../../tools/cuemu/README.md) 的結果
完全相符，包括全為正數、全為非正數及正負交錯的輸入。

## 延伸閱讀

- [前綴和](../016-prefix-sum/)、[分段前綴和](../070-segmented-prefix-sum/)、
  [Top-K](../029-top-k-selection/)（依門檻收集）。
