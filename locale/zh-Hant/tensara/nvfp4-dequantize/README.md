---
title: NVFP4 反量化
platform: Tensara
upstream: nvfp4-dequantize
url: https://tensara.org/problems/nvfp4-dequantize
difficulty: medium
tags: [quantization, nvfp4, low-precision, elementwise]
status: solved
---

# NVFP4 反量化

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/nvfp4-dequantize)

## 題意

依照 FlashInfer `e2m1_and_ufp8sf_scale_to_float` 的語意，將 NVFP4 矩陣（封裝的 E2M1、交錯排列的 E4M3 區塊縮放值，以及全域因子 $g$）還原成 FP32。尺寸最大為 $8192\times4096$；檢查條件為 `rtol = atol = 1e-3`。

## 圖解

![NVFP4 反量化：â = e2m1(code) · e4m3(s) / g](figure.svg)

每 16 個編碼共用一個 E4M3 縮放，整個張量共用全域係數 g。此例中每個編碼都乘上 3.25 / 2 = 1.625。

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
\text{out}_{i\ell} = \operatorname{e2m1}(c_{i\ell})\cdot\operatorname{e4m3}\bigl(s[\operatorname{idx}(i, \lfloor\ell/16\rfloor)]\bigr)\cdot\frac{1}{g}
$$

| 符號 | 意義 |
|---|---|
| out | FP32 結果，$M\times K$ |
| idx | 交錯的縮放值索引（如下） |

**E2M1（FP4）**包含 1 個符號位元、2 個指數位元和 1 個尾數位元（偏差值為 1）。其八種大小與解碼規則為

$$
\operatorname{e2m1}(c) = (-1)^{c_3}\cdot\begin{cases} \tfrac{1}{2}\,m, & m < 4 \\ (2 + (m \bmod 2))\cdot 2^{\lfloor m/2 \rfloor - 2}, & m \ge 4 \end{cases}
\in \pm\{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\}, \qquad m = c \mathbin{\&} 7
$$

| 符號 | 意義 |
|---|---|
| $c$ | 4 位元編碼；每個位元組存兩個編碼，元素 $2i$ 位於**低**半位元組 |
| $c_3$ | 符號位元（第 3 位元） |
| $m$ | 3 位元大小編碼，0 … 7 |

**E4M3（FP8）**包含 1 個符號位元、4 個指數位元和 3 個尾數位元，偏差值為 7，沒有無限大，而編碼 `0x7F`/`0xFF` 代表 NaN：

$$
\operatorname{e4m3}(b) = (-1)^{s}\cdot\begin{cases} \dfrac{f}{8}\cdot 2^{-6}, & e = 0 \ (\text{subnormal}) \\ \Bigl(1 + \dfrac{f}{8}\Bigr) 2^{e - 7}, & 1 \le e \le 15 \end{cases}, \qquad \lvert\operatorname{e4m3}\rvert \le 448
$$

| 符號 | 意義 |
|---|---|
| $b$ | 位元組 |
| $s, e, f$ | 符號位元、4 位元指數欄位、3 位元尾數欄位 |
| 448 | 最大有限值（$e = 15$、$f = 6$） |

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

每個位元組使用一個執行緒：解碼兩個半位元組、透過交錯索引取得區塊的 E4M3 縮放值（8 個執行緒共用，會命中快取）、乘上 $\operatorname{e4m3}(s)$ 與 $1/g$（在主機上只計算一次），再儲存一個 `float2`。

## 成本分析

$$
Q = 0.5\,MK + \frac{MK}{16}\ (\text{read}) + 4MK\ (\text{write})\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數，主要來自 FP32 輸出 |
| $\beta$ | DRAM 頻寬 |

約為 $4.56\,MK$ 位元組，主要來自 FP32 輸出。

## 常見陷阱

- **縮放值已經交錯排列**：不要再次交錯，也不要按列優先讀取。
- **全域因子是*編碼*因子**：應除以 $g$。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [NVFP4 量化](../nvfp4-quantize/)、[MXFP4 反量化](../mxfp4-dequantize/)。
