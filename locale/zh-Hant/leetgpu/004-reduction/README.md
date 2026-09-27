---
title: 歸約
platform: LeetGPU
upstream: medium/4_reduction
url: https://leetgpu.com/challenges/reduction
difficulty: medium
tags: [reduction, warp-shuffle, two-pass, deterministic]
status: solved
---

# 歸約

**平台：** LeetGPU · **難度：** medium · [題目說明](https://leetgpu.com/challenges/reduction)

## 題意

將包含 $N$ 個 float32 值的陣列加總成單一浮點數
（$1 \le N \le 10^8$，$\lvert x_i \rvert \le 1000$；基準
$N = 4\,194\,304$）。參考實作以 **float64** 加總後只取整一次，
容許誤差為 `atol = rtol = 1e-5`。歸約是「多個輸入 → 一個輸出」
的典型模式。Softmax、正規化、損失函式、點積及本站所有 `*-dim`
問題都會重用其基本元件。

## 圖解

![平行歸約：以平衡樹合併部分和](figure.svg)

最上排是每個執行緒的部分和。每往下一層，數值個數減半：先在 warp 內以 shuffle 合併，再透過共享記憶體跨 warp 合併，最後跨區塊合併。

## 數學表述

$$
S = \sum_{i=0}^{N-1} x_i
$$

| 符號 | 意義 |
|---|---|
| $N$ | 輸入元素數量 |
| $x_i$ | 第 $i$ 個輸入值（float32） |
| $S$ | 總和，以 float32 寫入 `output[0]` |

實數算術中的加法具有結合律，因此可用任何**樹狀結構**計算總和。
平行歸約會將它重新分組成多層部分和：

$$
S = \sum_{b=0}^{B-1} \underbrace{\sum_{w=0}^{W-1} \underbrace{\sum_{\ell=0}^{31} \underbrace{\sum_{i \in \mathcal{I}(b,w,\ell)} x_i}_{\text{thread}}}_{\text{warp}}}_{\text{block } b}
$$

| 符號 | 意義 |
|---|---|
| $B$ | 第一階段的區塊數（$\le 1024$） |
| $W$ | 每個區塊的 warp 數（$256/32 = 8$） |
| $\ell$ | warp 內的 lane 索引，$0..31$ |
| $\mathcal{I}(b,w,\ell)$ | 該執行緒的網格步進迴圈所走訪的索引：$i \equiv g \pmod{P}$，其中 $g$ 是全域執行緒 ID，$P = 256B$ 是執行緒總數 |

浮點加法**不具**結合律，因此不同樹狀結構會得到最後幾位略有不同的答案。
以精度 $u$ 加總 $N$ 項的誤差上限為

$$
\lvert \hat S - S \rvert \le \gamma_{h}\sum_i \lvert x_i \rvert, \qquad \gamma_h = \frac{h\,u}{1 - h\,u}
$$

| 符號 | 意義 |
|---|---|
| $\hat S$ | 計算出的總和 |
| $u$ | 單位捨入誤差：float32 為 $2^{-24}$，float64 為 $2^{-53}$ |
| $h$ | 加總樹高度：循序迴圈為 $N-1$，平衡樹約為 $\log_2 N$ |
| $\gamma_h$ | 標準誤差成長常數（Higham） |

因此樹狀計算不只更快，也比循序迴圈*更精確*。上層使用 float64
後，其誤差貢獻可忽略不計。

## 解題思路

### 第一階段：`partialSums`（≤ 1024 個區塊 × 256 個執行緒）

1. **搭配 `float4` 載入的網格步進迴圈。** 執行緒 $g$ 讀取從
   $g, g+P, g+2P, \dots$ 開始、每組 4 個浮點數的向量
   （每次載入 16 位元組並完整合併存取），在 float32 暫存器中累加。
   純量尾端迴圈處理最後 $N \bmod 4$ 個元素。
2. **Warp 層級。** 使用偏移量 16、8、4、2、1 的五個
   `__shfl_down_sync` 步驟，以 float64 將 32 個 lane 的值歸併至
   lane 0。Shuffle 直接交換暫存器，不需要共享記憶體或屏障。
3. **區塊層級。** 每個 warp 的 lane 0 將值寫入 `warp_sums[w]`。
   經一次 `__syncthreads()` 後，warp 0 用相同的 shuffle 歸約 8 個值。
4. 執行緒 0 將區塊的 float64 部分和寫入 `__device__` 陣列
   `g_partials[b]`。

### 第二階段：`finalSum`（1 個區塊）

使用相同的區塊歸約，以 float64 加總 $B$ 個部分和，再取整一次為
float32。第二個核心會在第一個核心完成後才啟動（同一串流），
因此不需要全網格同步。

### 為何使用兩階段而非 `atomicAdd`？

每個區塊以 `atomicAdd(output, partial)` 加入部分和的單一核心也能運作。
但浮點原子操作完成順序不固定，因此每次執行的末位可能不同。
兩階段設計具**確定性**，而第二階段只需幾微秒。

## 成本分析

$$
Q = 4N \ \text{bytes}, \qquad W = N - 1, \qquad I = \frac{W}{Q} \approx \frac{1}{4}, \qquad T_{\min} = \frac{4N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 讀取 DRAM 的位元組數（每個輸入只讀一次） |
| $W$ | 加法次數 |
| $I$ | 算術強度（FLOP/byte） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | 執行時間的頻寬下限 |

在基準大小下，$Q = 16.8$ MB，因此在 2 TB/s 時
$T_{\min} \approx 8\ \mu s$。此大小下，啟動成本（每個核心約 2–5 µs）
與傳輸時間相當，因此網格上限設為 1024 個區塊，且只啟動兩次。

## 常見陷阱

- **未對齊的 `float4`。** `input` 來自 `cudaMalloc`，會對齊 256 位元組，
  因此轉型成 `float4*` 是安全的；任意偏移的指標則不一定安全。
- **區塊步驟中未啟用的 warp。** 只有 `warp_sums` 的前 $W$ 項有效。
  `threadIdx.x >= W` 的執行緒必須貢獻 0。
- **在 `__syncthreads()` 前提早返回。** 每個執行緒都必須到達
  `blockReduceSum` 內的屏障，即使沒有處理任何元素。
- **很小的 $N$。** $N < 4$ 時向量數為零；純量尾端會處理所有元素，
  且網格至少會限制為一個區塊。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上通過
float64 參考實作，包括 $N = 1$、非 4 倍數及接近互相抵消的輸入。
它們也通過 `--reverse` 執行緒排程，以確認沒有遺漏屏障。

## 延伸閱讀

- [點積](../017-dot-product/)、[Softmax](../005-softmax/)、[RMS 正規化](../050-rms-normalization/)。
- [教學 03－平行歸約](../../tutorials/03-parallel-reduction.md)。
