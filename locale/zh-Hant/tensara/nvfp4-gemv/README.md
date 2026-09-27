---
title: NVFP4 GEMV
platform: Tensara
upstream: nvfp4-gemv
url: https://tensara.org/problems/nvfp4-gemv
difficulty: hard
tags: [gemv, nvfp4, low-precision, warp-per-row, bandwidth-bound]
status: solved
---

# NVFP4 GEMV

**平台：** Tensara · **難度：** 困難 · [題目說明](https://tensara.org/problems/nvfp4-gemv)

## 題意

計算 $\mathbf{y} = \hat{A}\hat{\mathbf{x}}$，輸出為 FP16；其中矩陣 $A$（$M\times K$）和向量 $\mathbf{x}$（長度 $K$）皆為 NVFP4（封裝的 E2M1、每 16 個元素一個且交錯排列的 E4M3 區塊縮放值，以及全域因子 $g_A$、$g_x$）。參考實作使用 FlashInfer 反量化，再執行 FP32 矩陣乘法。檢查條件為 `rtol = 2e-2`、`atol = 5e-2`。

## 圖解

![NVFP4 GEMV：一個 warp 逐區塊串流讀取量化後的一列與量化向量](figure.svg)

A 的列與向量 x 都是 NVFP4。對應的區塊（深色）解碼後相乘，再乘上兩個區塊縮放，最後在 warp 內合併部分和。

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
y_i = \frac{1}{g_A g_x}\sum_{\beta=0}^{K/16 - 1} \operatorname{e4m3}(s^A_{i\beta})\operatorname{e4m3}(s^x_{\beta})\sum_{\ell\in\beta} \operatorname{e2m1}(c^A_{i\ell})\operatorname{e2m1}(c^x_{\ell})
$$

| 符號 | 意義 |
|---|---|
| $y_i$ | 輸出元素（FP16） |
| $\beta$ | 沿 $K$ 的 16 元素區塊 |
| $s^A_{i\beta}, s^x_\beta$ | 矩陣列與向量的區塊縮放值 |
| $c^A, c^x$ | 元素編碼 |
| $g_A, g_x$ | 全域編碼因子 |

矩陣每列需使用 $K/2$ 位元組的編碼和 $K/16$ 個縮放值位元組：

$$
\text{bytes per weight} = \frac{1}{2} + \frac{1}{16} = 0.5625
$$

| 符號 | 意義 |
|---|---|
| 0.5625 | 每個 NVFP4 元素的儲存空間；FP32 則為 4 位元組（減少 7.1 倍） |

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

1. **每列使用一個 warp。**lane $l$ 處理該列的區塊 $l, l + 32, \dots$：讀取區塊的 8 個編碼位元組與縮放值位元組，以及向量中對應的 8 個位元組與縮放值（每個 warp 都讀取相同向量，因此可使用快取），接著解碼兩者並累加 16 個乘積。
2. 每個區塊的部分和乘上 $\operatorname{e4m3}(s^A)\operatorname{e4m3}(s^x)$，再以 FP32 累加。
3. 經過五步 shuffle 縮減後，lane 0 乘上 $1/(g_Ag_x)$ 並以 FP16 儲存。

向量（約 $0.56K$ 位元組）會保留在 L1/L2；矩陣只串流讀取一次。

## 成本分析

$$
Q \approx 0.5625\,MK + 0.5625\,K + 2M\ \text{bytes}, \qquad W = 2MK, \qquad T_{\min} = \frac{Q}{\beta_{\text{mem}}}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數，主要來自封裝矩陣 |
| $W$ | 浮點運算次數（另加解碼工作） |
| $\beta_{\text{mem}}$ | DRAM 頻寬 |

這正是 FP4 權重對 LLM 解碼很重要的原因：GEMV 受頻寬限制，因此只要解碼速度跟得上，縮小 7 倍的矩陣便可比 FP32 快最多 7 倍（比 FP16 快 3.6 倍）。每 9 位元組可解碼 16 個值，平均每個值約需 2–3 個整數指令；目前的 GPU 能以滿頻寬維持此速度。

## 常見陷阱

- $A$ 與 $\mathbf{x}$ 都使用**交錯縮放值**（向量視為 $1\times K$ 矩陣，並填補至 128 列的單元）。
- **輸出為 FP16**，以 FP32 累加。
- **全域縮放值只在最後套用一次**。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [矩陣向量乘法](../matrix-vector/)、[NVFP4 GEMM](../nvfp4-gemm/)、[NVFP4 反量化](../nvfp4-dequantize/)。
