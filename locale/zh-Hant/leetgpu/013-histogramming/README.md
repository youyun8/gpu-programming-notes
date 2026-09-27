---
title: 直方圖統計
platform: LeetGPU
upstream: medium/13_histogramming
url: https://leetgpu.com/challenges/histogramming
difficulty: medium
tags: [histogram, atomics, privatization, shared-memory]
status: solved
---

# 直方圖統計

**平台：** LeetGPU · **難度：** medium · [題目說明](https://leetgpu.com/challenges/histogramming)

## 問題

計算每個整數值 $v \in [0, B)$ 在長度為 $N$ 的 int32 陣列中出現幾次
（$1 \le N \le 10^8$，$1 \le B \le 1024$；基準
$N = 5\times10^7$、$B = 256$）。輸出為包含 $B$ 個計數的 int32 陣列，
且必須完全相符。直方圖是**寫入競爭**的經典案例：許多執行緒同時想增加
少數幾個計數器。

## 公式

$$
h_v = \sum_{i=0}^{N-1} [\,x_i = v\,], \qquad 0 \le v < B
$$

| 符號 | 意義 |
|---|---|
| $N$ | 輸入值數量 |
| $B$ | 分箱數（`num_bins`） |
| $x_i$ | 第 $i$ 個輸入值，$0 \le x_i < B$ |
| $h_v$ | 分箱 $v$ 的計數 |
| $[\cdot]$ | Iverson 括號：條件成立時為 1，否則為 0 |

### 私有化

將輸入分配給 $G$ 個區塊，分別計數後再加總部分直方圖：

$$
h_v = \sum_{g=0}^{G-1} h^{(g)}_v, \qquad h^{(g)}_v = \sum_{i \in \mathcal{P}_g} [\,x_i = v\,]
$$

| 符號 | 意義 |
|---|---|
| $G$ | 區塊數（$\le 1024$） |
| $\mathcal{P}_g$ | 區塊 $g$ 處理的索引（其網格步進切片） |
| $h^{(g)}_v$ | 區塊 $g$ 對分箱 $v$ 的私有計數，保存在共享記憶體 |

## 方法

1. `cudaMemset(histogram, 0, …)`。測試框架不會清零輸出緩衝區。
2. **私有直方圖。** 每個區塊將共享記憶體中的 `s_hist[B]` 清零，
   再執行網格步進迴圈，以 `atomicAdd(&s_hist[x_i], 1)` 累加。
   共享記憶體原子操作在 SM 內執行，只需幾個週期，且只與同區塊的
   256 個執行緒競爭。
3. **合併。** 經 `__syncthreads()` 後，執行緒 $t$ 以 `atomicAdd`
   將 $h^{(g)}_t, h^{(g)}_{t+256}, \dots$ 加入全域記憶體，並略過 0。

### 為何更快

| 版本 | 全域原子操作 | 競爭 |
|---|---|---|
| 樸素：每個元素一次全域原子操作 | $N = 5\times10^7$ | 整個 GPU 的每個執行緒競爭 256 個位址 |
| 私有化 | $\le G \cdot B = 262\,144$ | 每個位址只有 1024 個區塊，各一次 |

全域原子操作會在 L2 以每個位址固定的吞吐量處理，因此樸素版本會在熱門
分箱上序列化。私有化把超過 99% 的更新移到共享記憶體。

## 成本分析

$$
Q \approx 4N + 4GB \ \text{bytes}, \qquad T_{\min} = \frac{4N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 流量：每個輸入讀一次，加上 $G$ 個部分直方圖的合併 |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | 頻寬下限（讀取輸入） |

基準大小下，$Q \approx 200$ MB，在 2 TB/s 時約為 $100\ \mu s$。
實際吞吐量取決於資料分布。若所有值都落在一個分箱，區塊內的共享記憶體
原子操作會序列化；此時 warp 聚合原子操作
（`__match_any_sync` + 每組一次加法）或每個 warp 的子直方圖會有幫助。

## 常見問題

- **未將輸出清零。** 結果會是垃圾值，而且每次執行都不同。
- **共享陣列大小。** 它依最大 $B = 1024$ 配置，只使用前 $B$ 項，
  每個迴圈也以 $B$ 為界。
- **超出範圍的值。** 約束保證 $0 \le x_i < B$，核心仍會檢查，
  與參考實作的遮罩一致。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上得到
完全相符的整數結果，包括 $B = 1$（每個元素都在同一分箱，競爭最大）。

## 相關內容

- [計算陣列元素](../043-count-array-element/)、[Top-K](../029-top-k-selection/)、
  [基數排序](../036-radix-sort/)（其中的數字計數就是直方圖）。
- Tensara [直方圖](../../tensara/histogram/)。
