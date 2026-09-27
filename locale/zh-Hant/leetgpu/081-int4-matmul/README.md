---
title: 僅權重 INT4 量化矩陣乘法
platform: LeetGPU
upstream: medium/81_int4_matmul
url: https://leetgpu.com/challenges/int4-weight-only-quantized-matmul
difficulty: medium
tags: [gemm, int4, quantization, tensor-cores, wmma, w4a16]
status: solved
---

# 僅權重 INT4 量化矩陣乘法

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/int4-weight-only-quantized-matmul)

## 題意

實作僅權重 INT4 GEMM（「**W4A16**」），這是 GPTQ／AWQ 類 LLM 推論的核心。
$x$ 是 fp16 活化矩陣（$M\times K$）。$W$（$N\times K$）以封裝的 4 位元
整數儲存，每個位元組包含兩個值；沿 $K$ 的每組 $g$ 個連續權重共用一個
fp16 縮放比例。以 fp16 計算 $y = xW^{\mathsf T}$
（$M, N, K \le 8192$、$g \in \{2..128\}$；基準測試為 $4096^3$、
$g = 128$；容許誤差 `1e-2`）。

## 圖解

![W4A16：每個位元組存兩個 4 位元權重，每 g 個權重共用一個 fp16 縮放係數](figure.svg)

左圖：一個位元組存放兩個權重；每個 nibble 減 8 再乘上所屬群組的縮放係數，就是實際權重。右圖：沿 K 方向，每 g 個權重共用一個縮放係數。

## 數學表述

$$
W_{nk} = \bigl(q_{nk} - 8\bigr)\cdot s_{n,\lfloor k/g\rfloor}, \qquad
y_{mn} = \operatorname{fp16}\!\Bigl(\sum_{k=0}^{K-1} x_{mk}\, W_{nk}\Bigr)
$$

$$
q_{n,2i} = \bigl\lfloor b_{ni} / 16 \bigr\rfloor \ (\text{high nibble}), \qquad q_{n,2i+1} = b_{ni} \bmod 16 \ (\text{low nibble})
$$

| 符號 | 意義 |
|---|---|
| $M,\ N,\ K$ | Token 數（$x$ 的列）、輸出特徵數、輸入特徵數 |
| $x_{mk}$ | fp16 活化值 |
| $b_{ni}$ | 權重第 $n$ 列的第 $i$ 個封裝位元組（`w_q`，形狀為 $N \times K/2$） |
| $q_{nk}$ | 無號 4 位元編碼，$0..15$ |
| $q - 8$ | 位於 $[-8, 7]$ 的有號權重（偏移編碼） |
| $g$ | 沿 $K$ 的量化群組大小 |
| $s_{n,j}$ | 第 $n$ 列第 $j$ 組的 fp16 縮放比例（形狀為 $N\times K/g$） |
| $W_{nk}$ | 反量化後的權重 |
| $y_{mn}$ | fp16 輸出 |

### 為何僅權重量化有效

LLM 解碼時，$M$ 很小（只有幾個 token），GEMM 的瓶頸在於**讀取權重**。
INT4 權重比 fp16 小 4 倍，因此記憶體受限的解碼最多可加速 4 倍，同時活化值
與數學運算仍保持 fp16。群組縮放比例將量化誤差限制在局部：每 128 個權重
都有自己的動態範圍。

## 解題思路

採用 [GEMM（fp16）](../022-gemm/)的 tensor core GEMM，並將**反量化融合進
共享記憶體的暫存階段**：

1. **暫存 $x$**：如常將一個 $64\times32$ 的 fp16 tile 以零補齊。
2. **暫存 $W$**：每個執行緒讀取**一個封裝位元組**，也就是同一列中兩個
   連續權重。它拆出兩個 nibble、減去 8、查詢兩個群組縮放比例
   （當 $g = 2$ 且此配對跨越群組邊界時，兩者可能不同）、相乘，再將兩個
   fp16 值寫入 `w_s[n][k]`。全域記憶體中從不會出現反量化形式的 $W$。
3. **MMA**：`w_s` 以 $[n][k]$ 保存 $W$，也就是 $W^{\mathsf T}$ 為欄優先。
   它以 **`wmma::col_major`** 載入為 B fragment，因此不需轉置。每個 warp
   使用 fp32 累加器計算 2 × 2 個 fragment。
4. **結尾運算**：共享記憶體中的 float tile → 具有邊界檢查的 fp16 寫入。

## 成本分析

$$
W_{\text{flop}} = 2MNK, \qquad
Q_W = \frac{NK}{2} + 2\frac{NK}{g}\ \text{bytes}\ (\text{vs. } 2NK \text{ for fp16}), \qquad
Q_x = 2MK
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{flop}}$ | tensor core 的 FLOP 數 |
| $Q_W$ | 權重讀取一次的位元組數：每個權重 0.5 位元組，加上縮放比例 |
| $Q_x$ | 活化值的讀取位元組數 |

在基準測試（$4096^3$、$g = 128$）中，$Q_W = 8.4$ MB，而 fp16 為
33.5 MB；$W_{\text{flop}} = 137$ GFLOP，因此這個大 $M$ 案例受計算限制。
記憶體節省會在小 $M$（解碼）時發揮作用，此時同一核心在 $Q_W$ 上受頻寬
限制。該情況更適合 GEMV 風格的核心（每個輸出欄一個 warp）。

## 常見陷阱

- **Nibble 順序。** **高位** nibble 保存*偶數*索引 $2i$。這與較常見的
  低位優先封裝相反（可比較 Tensara FP4 題目）。
- **偏移編碼。** 有號值是 $q - 8$，不是二補數 nibble。
- 當 $g = 2$ 時，一對位元組內可能出現**群組邊界**：必須分別取得每個
  權重的縮放比例。
- 在 MMA 前將反量化權重進行 **fp16 捨入**，相較於參考實作的 float32
  反量化會增加誤差，但仍遠低於 `1e-2`。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-2`
通過，涵蓋每種群組大小，且 $M, N$ 不必為 64 的倍數。

## 延伸閱讀

- [INT8 量化矩陣乘法](../032-int8-quantized-matmul/)、[權重反量化](../064-weight-dequantization/)、
  [GEMM（fp16）](../022-gemm/)。Tensara [NVFP4 GEMM](../../tensara/nvfp4-gemm/)、
  [MXFP4 GEMM](../../tensara/mxfp4-gemm/)。
