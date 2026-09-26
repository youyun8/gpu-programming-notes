---
title: 前綴和
platform: LeetGPU
upstream: medium/16_prefix_sum
url: https://leetgpu.com/challenges/prefix-sum
difficulty: medium
tags: [scan, prefix-sum, reduce-then-scan, warp-shuffle]
status: solved
---

# 前綴和

**平台：** LeetGPU · **難度：** medium · [題目說明](https://leetgpu.com/challenges/prefix-sum)

## 問題

計算 $N$ 個 float32 值的包含式前綴和（累積和）
（$1 \le N \le 10^8$，$\lvert x_i\rvert \le 1000$；
基準 $N = 250\,000$；容許誤差 `1e-2`）。掃描是串流壓縮、基數排序、
稀疏矩陣建構與許多遞迴背後的平行基本元件。與歸約不同，
它會產生 $N$ 個輸出，每個輸出都依賴先前所有輸入。

## 公式

$$
y_i = \sum_{j=0}^{i} x_j, \qquad 0 \le i < N
$$

| 符號 | 意義 |
|---|---|
| $N$ | 陣列長度 |
| $x_j$ | 輸入值（float32） |
| $y_i$ | 包含式前綴和（float32 輸出） |

### 先歸約再掃描的分解

將陣列分成每塊 $C = 2048$ 個元素。令分塊總和為 $S_b$、
不包含自身的分塊偏移量為 $O_b$：

$$
S_b = \sum_{j=bC}^{bC + C - 1} x_j, \qquad O_b = \sum_{b' < b} S_{b'}, \qquad
y_i = O_{\lfloor i/C\rfloor} + \sum_{j = C\lfloor i/C\rfloor}^{i} x_j
$$

| 符號 | 意義 |
|---|---|
| $C$ | 分塊大小：256 個執行緒 × 8 個項目 = 2048 |
| $b$ | 分塊（區塊）索引 |
| $S_b$ | 分塊 $b$ 的總和 |
| $O_b$ | 分塊總和的不包含式前綴：分塊 $b$ 之前的所有值 |

在分塊內，每個執行緒處理 8 個連續項目。其前綴值是每執行緒總和的
不包含式掃描，使用 warp shuffle 計算：

$$
\text{warp inclusive scan:}\quad v \leftarrow v + \mathbb{1}[\ell \ge \delta]\cdot \texttt{shfl\_up}(v, \delta), \qquad \delta = 1, 2, 4, 8, 16
$$

| 符號 | 意義 |
|---|---|
| $\ell$ | Lane 索引 |
| $\delta$ | 每個 $\log_2 32 = 5$ 步驟的 shuffle 距離 |
| $\texttt{shfl\_up}(v, \delta)$ | Lane $\ell - \delta$ 所持有的 $v$ 值 |

這是 Hillis–Steele 掃描：工作量為 $O(n\log n)$，但對暫存器中的
32 個元素，5 個步驟比其他做法都便宜。

## 方法

1. **`blockTotals`**（每個分塊一個區塊）：將分塊加總成 $S_b$（float64）。
2. **`scanTotals`**（1 個區塊）：每次處理 256 個值，搭配持續更新的
   進位值，以 float64 將 $S_0, S_1, \dots$ 不包含式掃描成 $O_b$。
3. **`scanChunks`**（每個分塊一個區塊）：
   1. 以合併存取將分塊載入共享記憶體。
   2. 每個執行緒在暫存器中循序掃描自己的 8 個連續項目。
   3. 對每執行緒總和做區塊級掃描：先以 warp shuffle 掃描，
      再掃描 8 個 warp 總和。
   4. 加入執行緒前綴與 $O_b$，將結果寫回共享記憶體，
      再以合併存取儲存。

繞經共享記憶體可讓兩次全域存取都合併，同時讓每個執行緒處理
8 個*連續*項目（循序掃描的工作量最佳）。

### 精度

若以 float32 加總 $10^8$ 個絕對值最高 1000 的值，誤差會在約
50 000 個分塊偏移量間累積。以 float64 保存 $S_b$、$O_b$ 與區塊級掃描，
可讓 float32 只在 8 項循序掃描與最後儲存時取整。

## 成本分析

$$
Q = \underbrace{4N}_{\text{pass 1}} + \underbrace{4N + 4N}_{\text{pass 3}} = 12N \ \text{bytes}, \qquad W = O(N)
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 流量：輸入讀取兩次、輸出寫入一次 |
| $W$ | 加法次數，與 $N$ 成線性（工作量有效率） |

單遍「解耦回看」掃描（如 CUB）會透過全域旗標串接分塊前綴，
達到 $8N$ 位元組。基準大小（1 MB）完全位於 L2，
且啟動延遲占主導，因此三個簡單核心是良好取捨。

## 常見問題

- **不包含式與包含式。** 本題是包含式（$y_0 = x_0$）；
  分塊偏移量則是不包含式。
- **重用屏障。** `blockInclusiveScan` 會重用共享的 `warp_totals` 陣列，
  所以在下一次呼叫覆寫之前，需要尾端的 `__syncthreads()`。
- **暫存緩衝區。** $O_b$ 位於大小為 $\lceil N/C\rceil$ 的
  `cudaMalloc` 緩衝區，在最後一次同步後釋放。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上通過，
包括 $N = 1$ 與 $N$ 非 2048 倍數。$N = 10^8$ 的壓力測試也在容許誤差內
符合 float64 參考實作。

## 相關內容

- [分段前綴和](../070-segmented-prefix-sum/)、[串流壓縮](../072-stream-compaction/)、
  [基數排序](../036-radix-sort/)、[線性遞迴](../082-linear-recurrence/)。
- Tensara [累積和](../../tensara/cumsum/)、[累積乘積](../../tensara/cumprod/)。
