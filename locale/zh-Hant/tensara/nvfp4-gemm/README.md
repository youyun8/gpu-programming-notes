---
title: NVFP4 GEMM
platform: Tensara
upstream: nvfp4-gemm
url: https://tensara.org/problems/nvfp4-gemm
difficulty: hard
tags: [matmul, nvfp4, block-scaled, low-precision]
status: solved
---

# NVFP4 GEMM

**平台：** Tensara · **難度：** 困難 · [題目說明](https://tensara.org/problems/nvfp4-gemm)

## 題意

計算 $C = \hat{A}\hat{B}^{\mathsf T}$，輸出為 FP16；其中 $A$（$M\times K$）和 $B$（$N\times K$）皆為 NVFP4：元素是封裝的 E2M1，每 16 個元素使用一個 E4M3 縮放值（交錯排列），且每個運算元各有一個全域編碼因子（$g_A$、$g_B$）。參考實作為 `torch._scaled_mm`。檢查條件為 `rtol = 2e-2`、`atol = 5e-2`。

## 圖解

![NVFP4 GEMM：16 個一組的 E2M1 區塊、E4M3 縮放，以及每個運算元一個全域係數](figure.svg)

區塊只有 16 個元素，縮放能緊密追蹤局部數值大小。兩個全域係數只在累加結束時套用一次。

## 數學表述

**NVFP4** 沿 $K$ 使用 16 元素區塊，並採用兩層縮放：每個區塊有一個 FP8（E4M3）縮放值，每個張量另有一個 FP32 全域因子，讓整個張量落在 E4M3 × E2M1 可表示的範圍內（$448\times6 = 2688$）。第 $i$ 列中元素 $\ell$ 的反量化值為

$$
\hat{a}_{i\ell} = \frac{\operatorname{e2m1}(c_{i\ell})\cdot\operatorname{e4m3}\bigl(s_{i,\lfloor\ell/16\rfloor}\bigr)}{g}
$$

| 符號 | 意義 |
|---|---|
| $c_{i\ell}$ | 4 位元 E2M1 編碼（每個位元組存兩個，低半位元組在前） |
| $s_{i\beta}$ | 區塊 $\beta$ 的 E4M3 縮放值位元組，儲存在交錯配置中 |
| $g$ | 全域編碼因子 `sf_g`（FP32）；$1/g$ 是全域解碼縮放值 |
| $\hat{a}_{i\ell}$ | 編碼所代表的值 |

$$
c_{ij} = \sum_{\ell=0}^{K-1} \hat{A}_{i\ell}\,\hat{B}_{j\ell}, \qquad
\hat{A}_{i\ell} = \operatorname{e2m1}(c^A_{i\ell})\operatorname{e4m3}(s^A_{i,\lfloor\ell/16\rfloor})/g_A, \qquad \hat{B}_{j\ell} = \operatorname{e2m1}(c^B_{j\ell})\operatorname{e4m3}(s^B_{j,\lfloor\ell/16\rfloor})/g_B
$$

| 符號 | 意義 |
|---|---|
| $\hat{A}$ | 反量化後的 $A$，$M\times K$ |
| $\hat{B}$ | 反量化後的 $B$，儲存為 $N\times K$（因此乘積是 $\hat{A}\hat{B}^{\mathsf T}$，即「NT」GEMM） |
| $c$ | 輸出，$M\times N$（FP16） |
| $g_A, g_B$ | 全域編碼因子（`sf_g_a`、`sf_g_b`） |

由於每個 16 元素區塊共用一個縮放值，可依區塊重新組合總和；張量核心的區塊縮放 MMA 正是採用這種方式：

$$
c_{ij} = \frac{1}{g_A g_B}\sum_{\beta=0}^{K/16 - 1} \sigma^{A}_{i\beta}\,\sigma^{B}_{j\beta} \sum_{\ell \in \beta} x^{A}_{i\ell}\,x^{B}_{j\ell}
$$

| 符號 | 意義 |
|---|---|
| $\beta$ | 沿 $K$ 的區塊索引 |
| $\sigma^A_{i\beta}, \sigma^B_{j\beta}$ | 兩個區塊縮放值 |
| $x^A, x^B$ | 解碼後、套用縮放前的元素值 |

**交錯的縮放值配置。**區塊縮放張量核心 MMA（cuBLAS / CUTLASS、TorchAO `is_swizzled_scales=True`、FlashInfer）會把含 $R$ 列、$C$ 個縮放值欄的矩陣，儲存在 512 位元組的 $128\times4$ 單元中：

$$
\operatorname{idx}(r, c) = \Bigl(\bigl\lfloor \tfrac{r}{128} \bigr\rfloor \Bigl\lceil \tfrac{C}{4} \Bigr\rceil + \bigl\lfloor \tfrac{c}{4} \bigr\rfloor\Bigr)\cdot 512 +
(r \bmod 32)\cdot 16 + \Bigl\lfloor \tfrac{r \bmod 128}{32} \Bigr\rfloor\cdot 4 + (c \bmod 4)
$$

| 符號 | 意義 |
|---|---|
| $r$ | 矩陣列 |
| $c$ | 縮放值欄（沿 $K$ 的區塊索引） |
| $C$ | 縮放值欄數，$K/\text{block}$ |
| idx | 縮放值 $(r, c)$ 的位元組偏移量（`swizzledScaleIndex`） |

在一個單元內，$r, r+32, r+64, r+96$ 各列會交錯排列，因此一次 16 位元組載入就能提供某個執行緒所需的 4 列、共 4 個縮放值。

## 解題思路

採用 Tensara 各矩陣乘法頁面所使用、以暫存器分塊的 SGEMM 之區塊縮放版本（`blockScaledGemm`）：

1. **$64\times64$ 輸出圖塊**，256 個執行緒，每個執行緒負責 $4\times4$ 個輸出。
2. **大小為 32 的 K 切片（兩個 NVFP4 區塊）**：將 $A$ 與 $B$ 面板暫存到共享記憶體時，每個執行緒解碼元素編碼（半位元組 → `e2m1ToFloat` 乘以解碼後的 E4M3 區塊縮放值），再乘上透過交錯索引查得的區塊縮放值。圖塊儲存 FP32，因此內層迴圈就是一般的 FMA 外積。
3. **結尾處理**：乘一次 $1/(g_Ag_B)$，再轉換為 FP16（`__float2half_rn`）。

反量化後的矩陣不會寫入全域記憶體；相較於 FP32 GEMM，唯一的額外成本是暫存時的解碼工作。在 Blackwell（sm_100）上，同一份資料可直接送入 `tcgen05.mma` 區塊縮放指令，由硬體讀取這些交錯縮放值配置；此可攜式核心則使用 CUDA 核心。

## 成本分析

$$
W = 2MNK, \qquad Q_{\min} = 0.5\,(MK + NK) + \frac{MK + NK}{16} + 2\,MN\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數 |
| $Q_{\min}$ | 必要的 DRAM 位元組數：封裝運算元、縮放值與輸出 |

在結尾處理中只套用一次全域因子，而非逐元素套用，可省下 $2MNK/32$ 次乘法，且每個暫存值少一次捨入。

## 常見陷阱

- **輸出為 FP16**（`float16*`）：在 FP32 中累加，最後只轉換一次。
- 有**兩個全域因子**，且都是*編碼*因子：應除以兩者的乘積。
- 使用 **16 元素區塊**，不是 MX 的 32 元素區塊。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [NVFP4 GEMV](../nvfp4-gemv/)、[MXFP4 GEMM](../mxfp4-gemm/)、[NVFP4 量化](../nvfp4-quantize/)。
