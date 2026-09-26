---
title: 分段排他前綴和
platform: LeetGPU
upstream: medium/70_segmented_prefix_sum
url: https://leetgpu.com/challenges/segmented-exclusive-prefix-sum
difficulty: medium
tags: [scan, segmented-scan, monoid]
status: solved
---

# 分段排他前綴和

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/segmented-exclusive-prefix-sum)

## 問題

計算會在**每個分段起點重新開始**的排他前綴和。`flags[i] = 1` 代表新分段
開始（且 `flags[0] = 1` 恆成立），輸出 $i$ 是*同一分段內*先前值的總和
（$N \le 10^8$，值位於 $[-100, 100]$；基準測試 $N = 5\times10^7$；
容許誤差 `1e-3`）。分段掃描可在一次啟動中處理許多長度不一的序列，例如
稀疏矩陣的逐列運算、不規則序列批次或分組彙總。

## 公式

$$
y_i = \sum_{j = h(i)}^{i-1} x_j, \qquad h(i) = \max\{\, j \le i : f_j = 1 \,\}
$$

| 符號 | 意義 |
|---|---|
| $N$ | 陣列長度 |
| $x_j$ | 值（float32） |
| $f_j$ | 旗標：分段起點為 1，否則為 0 |
| $h(i)$ | 包含 $i$ 的分段起點索引 |
| $y_i$ | 輸出：排他分段前綴；每個起點皆有 $y_i = 0$ |

### 將分段掃描視為一般掃描

以運算子

$$
(f_1, s_1) \oplus (f_2, s_2) = \bigl(f_1 \lor f_2,\ \ f_2\ ?\ s_2 : s_1 + s_2\bigr)
$$

掃描配對 $(f, s)$。

| 符號 | 意義 |
|---|---|
| $(f, s)$ | 「包含起點」旗標，以及自最後一個起點（或範圍開頭）以來的總和 |
| $\lor$ | 邏輯 OR |
| $f_2\,?\,s_2 : s_1+s_2$ | 若右側範圍包含起點，便捨棄左側總和 |

$\oplus$ 具有**結合律**（單位元素為 $(0, 0)$），因此
[前綴和](../016-prefix-sum/) 的完整先歸約再掃描機制可原封不動地套用，
只有結合函式不同。

## 方法

以 2048 個元素為一個區塊（256 個執行緒 × 8 個連續項目）：

1. **`chunkAggregates`**：每個執行緒將自己的 8 個項目折疊成一個配對，
   區塊層級的 $\oplus$ 掃描會得到區塊彙總值 $(F_c, S_c)$。
2. **`scanAggregates`**（1 個區塊）：對區塊彙總值執行排他 $\oplus$ 掃描，
   並在每組 256 個區塊之間傳遞進位。其 `sum` 部分是傳入各區塊的值；
   若區塊的第一個元素不是分段起點，它就代表持續累加的分段總和。
3. **`scanChunks`**：重新計算每個執行緒的彙總值、執行區塊掃描，並透過
   共享記憶體中前一個執行緒的包含式結果，再與區塊進位結合，導出每個
   執行緒的排他前綴。接著依序走訪 8 個項目：遇到旗標時將 `running`
   重設為 0、寫入 `running`，再加上該值。

所有總和皆使用 float64，與參考實作相同。

### 使用不可交換運算子的區塊掃描

warp 步驟對配對的兩個部分使用含 `__shfl_up_sync` 的 Hillis–Steele：
`v = combine(other, v)`，較低 lane 的值位於**左側**。順序很重要，因為
$\oplus$ 具有結合律，卻*不具*交換律。

## 成本分析

$$
Q = \underbrace{8N}_{\text{pass 1: values + flags}} + \underbrace{8N + 4N}_{\text{pass 3}} = 20N\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：值與旗標各讀取兩次，輸出寫入一次 |

基準測試為 1 GB，在 2 TB/s 下約需 0.5 ms。

## 常見陷阱

- **`combine` 的運算元順序**（左側 = 較早）。交換順序會悄悄破壞跨越
  執行緒或 warp 邊界的分段。
- **排他語意。** 起點元素輸出 0，且進位絕不跨越分段起點。
- **float64 前綴。** 長分段會加總許多值；參考實作本身也使用 double。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上通過，
包括全部都是起點（輸出全為零）、單一分段（一般排他掃描），以及跨越
區塊邊界的分段。

## 相關內容

- [前綴和](../016-prefix-sum/)、[串流壓縮](../072-stream-compaction/)、
  [線性遞迴](../082-linear-recurrence/)（另一種不可交換掃描）。
