---
title: 平行合併
platform: LeetGPU
upstream: medium/71_parallel_merge
url: https://leetgpu.com/challenges/parallel-merge
difficulty: medium
tags: [merge, merge-path, binary-search]
status: solved
---

# 平行合併

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/parallel-merge)

## 題意

將兩個已排序的 float32 陣列 $A$（長度 $M$）與 $B$（長度 $N$）合併為
長度 $M + N$ 的已排序陣列 $C$（$M + N \le 5\times10^7$；基準測試
$M = N = 2.5\times10^7$）。結果必須完全精確。循序合併本質上是串行作業，
而 **merge path** 技術能透過二分搜尋將它切分成完全獨立的片段。

## 圖解

![Merge path：每個執行緒以二分搜尋找出自己負責輸出區段的起點](figure.svg)

紅色階梯是 A（列）對 B（欄）網格中的合併路徑。每條虛線對角線代表一個輸出位置 k；它與路徑的交點說明前 k 個輸出有幾個來自 A，這可沿對角線做二分搜尋求得。

## 數學表述

穩定合併的前 $k$ 個輸出，是由 $A$ 的前 $i$ 個元素與 $B$ 的前 $k-i$
個元素組成，其中唯一的**共同秩** $i$ 為：

$$
C[0{:}k] = \operatorname{merge}\bigl(A[0{:}i],\ B[0{:}k-i]\bigr), \qquad
i = \operatorname{corank}(k) = \min\bigl\{\, i \in [\max(0, k-N),\ \min(k, M)] : A_i > B_{k-i-1} \,\bigr\}
$$

（以 $A_M = +\infty$ 和 $B_{-1} = -\infty$ 作為哨兵值）。

| 符號 | 意義 |
|---|---|
| $M,\ N$ | $A$ 與 $B$ 的長度 |
| $k$ | 輸出位置（合併網格的一條「對角線」） |
| $i$ | 位置 $k$ 之前從 $A$ 取出的元素數 |
| $k - i$ | 從 $B$ 取出的元素數 |
| $A_i > B_{k-i-1}$ | 停止條件：$A_i$ 必須排在 $B$ 最後一個已取元素之後。相等時取 $A$（穩定且 $A$ 優先） |

此述詞對 $i$ 單調，因此可用 **二分搜尋**在
$O(\log\min(M, N))$ 個步驟內找出共同秩。

從幾何角度看：將合併過程畫成一條穿越 $M \times N$ 網格的單調路徑。
輸出位置 $k$ 位於反對角線 $i + j = k$ 上，而二分搜尋會找出路徑與它的交點。

## 解題思路

- 每個執行緒負責 8 個連續輸出，起點為 $k = 8t$。
- 它獨立以二分搜尋找出自己的起始共同秩 $i$，不需通訊；接著設定
  $j = k - i$，並以「若 $A_i \le B_j$ 就從 $A$ 取值」的規則循序合併
  8 個元素。
- 起點超過 $M + N$ 的執行緒直接結束。

由於共同秩與合併迴圈採用**相同的相等值規則**（$A$ 優先），相鄰執行緒
的範圍會恰好鋪滿輸出，沒有缺口也不會重疊。

## 成本分析

$$
W = O\!\left(\frac{M+N}{8}\log\min(M,N)\right) + O(M+N), \qquad Q \approx 4(M + N)\cdot 2\ \text{bytes} + \text{search reads}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 工作量：每 8 個輸出一次二分搜尋，再加上線性合併 |
| $Q$ | DRAM 位元組數：$A$ 與 $B$ 各讀取一次，$C$ 寫入一次；二分搜尋的探查多半命中快取 |

基準測試的必要流量為 400 MB，約需 0.2 ms。合併迴圈的讀取由資料決定
（執行緒會以不可預測的方式在 $A$ 或 $B$ 中前進），因此 warp 只能部分
合併存取。區塊層級的 merge path 可讓 DRAM 讀取完全合併：每個區塊先求
一次共同秩，將兩個輸入視窗放入共享記憶體，再於共享記憶體中求各執行緒
的共同秩。

## 常見陷阱

- 搜尋與合併若採用**不一致的相等值處理方式**，只要 $A$ 與 $B$ 有相同值，
  就會產生重複或遺漏的元素。
- **搜尋邊界。** $i \in [\max(0, k-N),\ \min(k, M)]$。超出此範圍時，
  $j$ 會是負值或超過 $N$。
- **64 位元的 $k$。** 此處 $M + N$ 可放進 `int`，但起始位置仍先以
  64 位元計算，再進行檢查。

## 驗證

所有 LeetGPU 測試案例皆與 [cuemu](../../tools/cuemu/README.md) 的結果
完全相符，包括 $M = 1$ 或 $N = 1$、所有值皆相等，以及數值範圍互不重疊。

## 延伸閱讀

- [排序](../015-sorting/)、[基數排序](../036-radix-sort/)、[Top-K](../029-top-k-selection/)。
