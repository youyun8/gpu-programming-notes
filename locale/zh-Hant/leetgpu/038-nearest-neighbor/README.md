---
title: 最近鄰
platform: LeetGPU
upstream: medium/38_nearest_neighbor
url: https://leetgpu.com/challenges/nearest-neighbor
difficulty: medium
tags: [brute-force, shared-memory, exact-arithmetic, all-pairs]
status: solved
---

# 最近鄰

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/nearest-neighbor)

## 題意

對三維空間中的每個 $N$ 個點，找出距離最近的*另一個*點之索引
（$1 \le N \le 10^5$，座標範圍為 $[-1000, 1000]$；
效能評測使用 $N = 10^4$）。檢查方式為**完全相等**：每個回傳索引都必須
與 PyTorch 的 `argmin` 相同。因此，本題不只是分塊練習，也是在處理
浮點運算的可重現性。

## 圖解

![最近鄰：每個點掃描所有其他點，保留最小的距離](figure.svg)

每個箭頭從一個點指向離它最近的另一點；互為最近鄰時會出現雙向箭頭。結果必須與 PyTorch 逐位元一致，包括平手的情況。

## 數學表述

$$
\operatorname{nn}(i) = \arg\min_{j \ne i} d_{ij}, \qquad
d_{ij} = \operatorname{fl}\Bigl(\operatorname{fl}\bigl(\operatorname{fl}(\Delta x^2) + \operatorname{fl}(\Delta y^2)\bigr) + \operatorname{fl}(\Delta z^2)\Bigr)
$$

$$
\Delta x = \operatorname{fl}(x_i - x_j), \quad \Delta y = \operatorname{fl}(y_i - y_j), \quad \Delta z = \operatorname{fl}(z_i - z_j)
$$

| 符號 | 意義 |
|---|---|
| $N$ | 點的數量 |
| $(x_i, y_i, z_i)$ | 點 $i$ 的座標（交錯儲存：`points[3i..3i+2]`） |
| $d_{ij}$ | 歐幾里得距離平方，每個運算後都以 float32 捨入 |
| $\operatorname{fl}(\cdot)$ | 一次以最接近值捨入的 float32 運算 |
| $\operatorname{nn}(i)$ | 輸出 `indices[i]`；若距離相同，取最小的 $j$（argmin 語意） |

不必計算平方根，因為 $\sqrt{\cdot}$ 是單調函數。

### 為何必須固定運算順序

PyTorch 先計算 `diff*diff`（3 次各自捨入的乘法），再執行
`sum(dim=2)`（由左至右）。若任由編譯器處理，它可能將
$\Delta x^2 + \Delta y^2$ 合併成 FMA；如此只會捨入一次，而非兩次。
這會改變 $d_{ij}$ 的最低位元；每當兩個候選值相差不到一個 ulp 時，
argmin 就可能翻轉。核心函式使用永遠不會合併的 `__fsub_rn`、
`__fmul_rn` 與 `__fadd_rn`。

## 解題思路

- 每個查詢點 $i$ 使用一個執行緒（暫存器保存 $x_i, y_i, z_i$、
  最短距離與最佳索引）。
- 候選點以每塊 256 個的方式流經共享記憶體，並採用**陣列結構**
  （`sx`、`sy`、`sz`）。區塊中的每個執行緒載入一個候選點，然後同步。
- 每個執行緒都會掃描 256 個候選點；所有執行緒會同時讀取 `sx[t]`，
  因此可使用廣播。更新規則是
  `if (j != i && (d < best || best_j < 0))`：因為 $j$ 單調遞增，
  使用嚴格的 `<` 可在距離相同時保留最低索引。
- 即使執行緒的 $i \ge N$，仍會協助載入資料塊並參與同步。

## 成本分析

$$
W \approx 9N^2 \ \text{FLOPs}, \qquad Q_{\text{DRAM}} \approx 12N\left\lceil \frac{N}{256}\right\rceil + 16N \ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 每對點進行 3 次減法、3 次乘法、2 次加法與 1 次比較 |
| $Q_{\text{DRAM}}$ | 每個區塊都串流讀取所有點（每點 12 位元組），再加上自身讀取與索引寫入 |

當 $N = 10^4$ 時，$W \approx 9\times10^8$，執行時間遠低於一毫秒，
且受運算限制。當 $N = 10^5$ 時，$W$ 會增加 100 倍。若使用空間資料結構
（均勻網格，或以 Morton 編碼進行基數排序後建立的 k-d 樹），
可將搜尋降至 $O(N\log N)$。

## 常見陷阱

- **FMA 合併**會破壞完全相符的要求（見上文）。
- **$N = 1$。** 此時沒有其他點，參考實作對全為 $+\infty$ 的值執行
  argmin 會回傳索引 0。核心函式透過 `best_j < 0 ? 0` 重現此行為。
- **重複的點。** 與另一個索引的距離為 0，仍是有效的最近鄰。
  只排除 $j = i$。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 中完全相符，
包括重複的點與 $N = 1, 2$。

## 延伸閱讀

- [多代理模擬](../014-multi-agent-sim/)（相同的分塊方式）、
  [K-Means](../020-kmeans-clustering/)。
