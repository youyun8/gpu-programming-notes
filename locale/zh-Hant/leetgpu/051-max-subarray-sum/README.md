---
title: 最大子陣列總和
platform: LeetGPU
upstream: medium/51_max_subarray_sum
url: https://leetgpu.com/challenges/max-subarray-sum
difficulty: medium
tags: [scan, prefix-sum, sliding-window, integer]
status: solved
---

# 最大子陣列總和

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/max-subarray-sum)

## 問題

找出 int32 陣列中所有**恰好**包含 $w$ 個元素的連續視窗之最大總和；
陣列長度為 $N$（$N \le 5\times10^4$，值域為 $[-10, 10]$；
基準測試使用 $N = 5\times10^4$）。結果必須完全精確。循序
滑動視窗是 $O(N)$，但無法平行執行。平行版本則使用**前綴和**。

## 公式

$$
\text{out} = \max_{0 \le i \le N - w}\ \sum_{t = i}^{i + w - 1} x_t
= \max_{0 \le i \le N - w}\ \bigl(P_{i+w} - P_i\bigr), \qquad P_0 = 0,\quad P_{j} = \sum_{t=0}^{j-1} x_t
$$

| 符號 | 意義 |
|---|---|
| $N$ | 陣列長度 |
| $w$ | 視窗大小（`window_size`），$1 \le w \le N$ |
| $x_t$ | 輸入值（int32） |
| $i$ | 視窗起點 |
| $P_j$ | 排他前綴和：前 $j$ 個元素的總和（$P$ 有 $N+1$ 個項目） |
| out | 最大視窗總和 |

每個視窗總和都可化為兩個前綴值的**一次減法**。掃描完成後，
問題就成了對 $N - w + 1$ 個獨立差值進行資料平行的最大值歸約。

## 方法

$N \le 50\,000$ 可輕鬆放入**含 1024 個執行緒的單一區塊**，因此不需要
跨區塊掃描機制。

1. **以 1024 個元素為一批掃描。** 每個執行緒載入一個元素。區塊內含式掃描
   （warp `__shfl_up_sync` 掃描、掃描 32 個 warp 總和，再加回 warp 位移）
   加上前一批的累計 `carry`，即可得到 $P_{i+1}$，並寫入全域 `prefix`
   陣列。最後一個執行緒發布新的 carry，且前後都設置屏障。
2. **求視窗最大值。** 執行緒 $t$ 掃描 $i = t, t + 1024, \dots$，
   計算 $P_{i+w} - P_i$ 並保留最大值。最後以 warp
   `__shfl_xor_sync` 最大值歸約，再於共享記憶體中處理 32 個 warp 最大值。

## 成本分析

$$
W_{\text{naive}} = w\,(N - w + 1), \qquad W_{\text{scan}} = O(N), \qquad Q \approx 4N + 4(N+1) + 8(N-w+1) \ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{naive}}$ | 每個視窗都從頭加總時的加法次數（最高為 $6.25\times10^8$，此時 $w = N/2$） |
| $W_{\text{scan}}$ | 掃描與求最大值的工作量 |
| $Q$ | 位元組數：讀取輸入、寫入前綴，以及每個視窗讀取兩個前綴值 |

所有資料都能放入 L2（約 600 KB）。執行時間是數微秒的單區塊工作，再加上啟動成本。

## 注意事項

- **前綴的差一錯誤。** 使用 $P$，令其含 $N + 1$ 個項目且
  $P_0 = 0$，可讓視窗 $[i, i+w)$ 恰好等於 $P_{i+w} - P_i$。
- **輸入全為負數。** 最大值可能是負數，因此要以 `INT_MIN` 初始化，而非 0。
- **Carry 競爭。** 掃描時每個執行緒都會讀取 `carry`，所以更新必須位於兩個屏障之間。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆完全相等，
包括 $w = 1$、$w = N$ 與全負數陣列。

## 相關內容

- [前綴和](../016-prefix-sum/)、[子陣列總和](../047-subarray-sum/)。
