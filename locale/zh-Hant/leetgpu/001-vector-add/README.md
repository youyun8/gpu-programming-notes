---
title: 向量加法
platform: LeetGPU
upstream: easy/1_vector_add
url: https://leetgpu.com/challenges/vector-addition
difficulty: easy
tags: [elementwise, memory-bound, coalescing]
status: solved
---

# 向量加法

**平台：** LeetGPU · **難度：** easy · [題目說明](https://leetgpu.com/challenges/vector-addition)

## 問題

將兩個 float32 向量逐元素相加。`A`、`B` 和 `C` 都是指向長度為
$N$ 的陣列的裝置指標（$1 \le N \le 10^8$；效能以
$N = 2.5\times10^7$ 測量）。結果必須寫入 `C`。這是 GPU 程式設計的
「Hello World」，值得仔細實作：它是最純粹的**頻寬受限**核心，
而這裡的執行緒索引模式會在本站所有逐元素問題中重複使用。

## 公式

$$
C_i = A_i + B_i, \qquad i = 0, 1, \dots, N-1
$$

| 符號 | 意義 |
|---|---|
| $N$ | 每個向量的元素數量 |
| $i$ | 元素索引（從 0 開始） |
| $A_i,\ B_i$ | 第 $i$ 個輸入元素，IEEE-754 float32 |
| $C_i$ | 第 $i$ 個輸出元素，float32 |

每個輸出都只依賴兩個輸入中的各一個元素，因此這個問題
**非常適合平行化**：任何兩個輸出都不共用工作。

產生 $C_i$ 的執行緒可由標準的一維全域索引找出：

$$
i = b \cdot T + t, \qquad G = \left\lceil \frac{N}{T} \right\rceil
$$

| 符號 | 意義 |
|---|---|
| $t$ | 區塊內的執行緒索引（`threadIdx.x`），$0 \le t < T$ |
| $b$ | 網格內的區塊索引（`blockIdx.x`），$0 \le b < G$ |
| $T$ | 每個區塊的執行緒數（`blockDim.x`），此處 $T = 256$ |
| $G$ | 啟動的區塊數（`gridDim.x`） |
| $\lceil\cdot\rceil$ | 向上取整；最後一個區塊可能有部分執行緒閒置，因此需要 $i < N$ 的防護條件 |

## 方法

### 平行分解

每個元素由一個執行緒處理。以 `vectorAdd<<<G, 256>>>` 啟動，
其中 $G = \lceil N/256 \rceil$。每個執行緒計算自己的 $i$；
若 $i \ge N$ 就返回，否則從 `A` 和 `B` 各載入一次、執行一次加法，
再將結果儲存至 `C`。

### 為何記憶體存取模式很重要

一個 warp 包含 32 個連續執行緒，因此會存取 32 個連續的浮點數
$A_{32w}, \dots, A_{32w+31}$，也就是連續的 128 位元組。記憶體系統
會以一筆完全利用的交易（4 個 32 位元組的區段）處理它。這稱為
**合併存取**，也是此處唯一真正重要的最佳化：取回的每個位元組都會被使用。

### 為何不讓每個執行緒處理更多工作？

網格步進迴圈或 `float4` 向量化載入（每個執行緒處理 4 個元素）
能減少指令數，對非常大的 $N$ 可能帶來幾個百分點的改善。
但它們不會改變限制執行時間的位元組數。當 $N$ 很大時，
簡單的每元素一執行緒版本已能使 DRAM 飽和，因此此處保留它以維持清楚易懂。

## 成本分析

$$
W = N, \qquad Q = 3 \cdot 4N = 12N \ \text{bytes}, \qquad I = \frac{W}{Q} = \frac{1}{12}\ \text{FLOP/byte},
\qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 工作量：浮點運算次數（每個元素一次加法） |
| $Q$ | DRAM 流量（位元組）：讀取 $A$ 和 $B$、寫入 $C$，每個值 4 位元組 |
| $I$ | 算術強度，即每位元組 DRAM 流量所做的 FLOP |
| $\beta$ | GPU 的可持續 DRAM 頻寬（位元組/秒） |
| $T_{\min}$ | 記憶體流量所造成的核心執行時間下限 |

現代 GPU 通常需要約 10–100 FLOP/byte 的 $I$，算術運算才會成為限制，
因此 $I = 1/12$ 的這個核心**嚴重受記憶體限制**。在基準大小
$N = 2.5\times10^7$ 時，$Q = 300$ MB。若 GPU 的
$\beta \approx 2\ \text{TB/s}$，則 $T_{\min} \approx 150\ \mu s$。
所以衡量此核心應看實際達成的 GB/s，而不是 GFLOP/s。

## 常見問題

- **缺少邊界檢查。** 當 $N$ 不是 $T$ 的倍數時，最後一個區塊會有
  $i \ge N$ 的執行緒，這些執行緒不得讀寫。
- **索引溢位。** `blockIdx.x * blockDim.x` 以 32 位元 `int` 計算。
  在 $N < 2^{31}$ 時是安全的（本題符合）；超過時請使用 `size_t`。
- **計時。** 啟動是非同步的。在啟動後呼叫 `cudaDeviceSynchronize()`
  可讓測試程式確認工作完成（也能顯示啟動錯誤）。

## 驗證

在每個 LeetGPU 測試案例上，以平台本身的參考實作（`torch.add`）
和 `atol = rtol = 1e-5` 比對；案例包含大小不是 256 倍數的情況，
並使用每個緩衝區後方都有防護頁的 [cuemu](../../tools/cuemu/README.md)
CPU 模擬器。

## 相關內容

- [矩陣加法](../008-matrix-addition/)、[ReLU](../021-relu/)、
  [色彩反轉](../007-color-inversion/)：相同模式在二維或其他逐元素函式中的應用。
- [教學 01－執行模型](../../tutorials/01-execution-model.md)和
  [02－記憶體階層](../../tutorials/02-memory-hierarchy.md)。
