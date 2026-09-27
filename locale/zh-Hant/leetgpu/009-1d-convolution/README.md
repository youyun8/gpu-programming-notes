---
title: 一維卷積
platform: LeetGPU
upstream: easy/9_1d_convolution
url: https://leetgpu.com/challenges/1d-convolution
difficulty: easy
tags: [convolution, shared-memory, dynamic-shared-memory, register-blocking]
status: solved
---

# 一維卷積

**平台：** LeetGPU · **難度：** easy · [題目說明](https://leetgpu.com/challenges/1d-convolution)

## 題意

以長度 $K$ 的核心對長度 $L$ 的 float32 訊號執行「有效」一維卷積
（嚴格來說是互相關，因為核心未翻轉）
（$1 \le K \le 2047$，$K \le L \le 1.5\times10^6$；基準
$L = 1.5\times10^6$、$K = 2047$）。輸出長度為 $L - K + 1$，
容許誤差為 `1e-4`。當 $K$ 達數千時，樸素核心會從全域記憶體重讀
每個輸入元素 $K$ 次。解法是將**輸入視窗平鋪到共享記憶體**。

## 圖解

![有效（valid）一維卷積：每個輸出是 K 個連續輸入的加權和](figure.svg)

輸出 y₃（深綠）由輸入 x₃、x₄、x₅（深藍）與權重 w 組合而成。相鄰輸出共用大部分輸入，這正是「共享記憶體分塊加 halo」所利用的重用。

## 數學表述

$$
y_i = \sum_{j=0}^{K-1} x_{i+j}\, w_j, \qquad 0 \le i < L - K + 1
$$

| 符號 | 意義 |
|---|---|
| $L$ | 輸入長度（`input_size`） |
| $K$ | 核心長度（`kernel_size`） |
| $x_n$ | 輸入訊號，$0 \le n < L$ |
| $w_j$ | 核心權重，$0 \le j < K$ |
| $y_i$ | 輸出樣本；共有 $L - K + 1$ 個（僅「有效」位置） |

### 平鋪

區塊 $b$ 產生 $T = 1024$ 個輸出
$y_{bT}, \dots, y_{bT+T-1}$，只依賴以下輸入視窗：

$$
x_{bT},\ \dots,\ x_{bT + T + K - 2} \qquad (\text{length } T + K - 1)
$$

| 符號 | 意義 |
|---|---|
| $b$ | 區塊索引 |
| $T$ | 每區塊輸出數：256 個執行緒 × 每執行緒 4 個輸出 = 1024 |
| $T + K - 1$ | 暫存於共享記憶體的輸入視窗（輸出加上核心「邊暈」） |

## 解題思路

1. **暫存。** 區塊將全部 $K$ 個權重與自己的 $T + K - 1$ 個輸入值
   複製到**動態**共享記憶體。大小取決於 $K$，因此以第三個啟動參數傳入：
   $(2K - 1 + T)\cdot 4$ 位元組；$K = 2047$ 時為 20.4 KB。
   超出輸入尾端的視窗元素補 0；它們只供給永遠不會寫入的輸出。
2. `__syncthreads()`。
3. **計算。** 執行緒 $t$ 負責區塊內的輸出
   $t,\ t{+}256,\ t{+}512,\ t{+}768$。每個 $j$ 只載入一次 $w_j$
   （**廣播**：所有執行緒讀同一字組，只需一筆交易），並與
   `s_input[t + 256r + j]` 執行 4 次 FMA。固定 $j$ 和 $r$ 時，
   連續執行緒讀取連續字組，因此沒有 bank 衝突。
4. 經邊界檢查後儲存 4 個結果。

每執行緒處理 4 個輸出，可讓每個權重從暫存器重用 4 次，
也提供 4 條獨立 FMA 鏈，協助隱藏 FMA 延遲。

## 成本分析

$$
W = 2K(L-K+1), \qquad
Q_{\text{naive}} \approx 4\cdot 2K(L-K+1), \qquad
Q_{\text{tiled}} \approx 4\left(L\,\frac{T+K-1}{T} + (L-K+1) + K\left\lceil\tfrac{L-K+1}{T}\right\rceil\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP（每個輸出的每個 tap 各一次乘加） |
| $Q_{\text{naive}}$ | 若每個 tap 都從全域記憶體讀取 $x$ 與 $w$ 的位元組數 |
| $Q_{\text{tiled}}$ | 平鋪後的位元組數：因邊暈重疊，每個輸入約讀取 $(T+K-1)/T$ 次；每個輸出寫一次；每區塊讀一次核心 |
| $T$ | 每區塊輸出數（1024） |

基準大小下，$W \approx 6.1\times10^9$ FLOP，而
$Q_{\text{tiled}} \approx 4(1.5\text{M}\cdot 3 + 1.5\text{M} + 3\text{M}) \approx 36$ MB。
強度約為 170 FLOP/byte，因此核心受共享記憶體載入與 FMA 的
**計算限制**。每次共享載入約執行一次 FMA。更大的暫存器分塊
（例如每執行緒 8 個輸出）可更接近 FMA roofline。

## 常見陷阱

- **動態共享記憶體大小。** 忘記第三個啟動參數會得到 0 位元組，
  所有存取都會越界。超過 48 KB 時需使用
  `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)`；本題最大僅 20 KB。
- **卷積與互相關。** 參考實作是 `unfold` + `einsum`，不會翻轉核心。
- **輸出長度。** 是 $L - K + 1$，不是 $L$；沒有填補。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上以
`1e-4` 通過，包括 $K = 1$、$K = L$（單一輸出）與最大值 $K = 2047$。

## 延伸閱讀

- [二維卷積](../010-2d-convolution/)、[三維卷積](../011-3d-convolution/)、
  [因果深度卷積 Conv1D](../090-causal-depthwise-conv1d/)。
- Tensara [一維卷積](../../tensara/conv-1d/)。
