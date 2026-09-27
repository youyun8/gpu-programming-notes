---
title: NVFP4 量化
platform: Tensara
upstream: nvfp4-quantize
url: https://tensara.org/problems/nvfp4-quantize
difficulty: medium
tags: [quantization, nvfp4, low-precision, half-warp]
status: solved
---

# NVFP4 量化

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/nvfp4-quantize)

## 題意

給定全域編碼因子 $g$（`sf_g`），將 $M\times K$ **FP16** 矩陣量化為 NVFP4：元素為封裝的 E2M1，每 16 個元素使用一個 E4M3 縮放值，並採用交錯的 128×4 配置，行為依照 FlashInfer 的 `nvfp4_quantize`。檢查條件（將兩邊反量化後）為 `rtol = atol = 1e-3`。

## 圖解

![NVFP4 量化：在全域係數 g 之上，每 16 個值再配一個 E4M3 縮放](figure.svg)

區塊縮放為 g·α/6 捨入到 E4M3（此例為 1.75）。接著把元素乘上 g 並除以這個捨入後的縮放，再捨入到 E2M1。

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

第 $i$ 列區塊 $\beta$ 的量化方式：

$$
\alpha_{i\beta} = \max_{\ell\in\beta}\lvert a_{i\ell}\rvert, \qquad
s_{i\beta} = \operatorname{e4m3}_{\text{RNE,sat}}\Bigl(g\cdot\frac{\alpha_{i\beta}}{6}\Bigr), \qquad
c_{i\ell} = \operatorname{e2m1}_{\text{RNE,sat}}\Bigl(a_{i\ell}\cdot\frac{g}{\operatorname{e4m3}(s_{i\beta})}\Bigr)
$$

| 符號 | 意義 |
|---|---|
| $a_{i\ell}$ | 轉換為 FP32 的 FP16 輸入元素 |
| $\alpha_{i\beta}$ | 區塊絕對值最大值 |
| $\alpha/6$ | 將區塊最大值對應到 E2M1 最大值 6 的解碼縮放值 |
| $\operatorname{e4m3}_{\text{RNE,sat}}$ | 捨入至最接近的偶數，並飽和至 $\pm448$（「satfinite」） |
| $\operatorname{e2m1}_{\text{RNE,sat}}$ | 在 8 種大小中捨入至最接近的偶數，並飽和於 6 |

元素編碼時使用*捨入後*的縮放值 $\operatorname{e4m3}(s)$（而非 $\alpha/6$），才能讓 decode(encode(x)) 的結果一致。

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

1. **`zeroScales`** 清除整個已填補的縮放值緩衝區（交錯配置會將列數填補至 128 的倍數，並將縮放值欄數填補至 4 的倍數；填補區必須為 0）。
2. **`quantizeNvfp4`**：每個區塊使用一個**半 warp（16 個 lane）**。每個 lane 載入一個 FP16 元素；在 16 個 lane 間執行四步 shuffle 最大值運算，以取得 $\alpha$；每個 lane 都計算 E4M3 縮放值位元組（逐位元精確的 RNE 與飽和處理），將它解碼回 FP32，再編碼自己的元素。偶數 lane 會把自己與相鄰奇數 lane 的值封裝成一個位元組。半 warp 的 lane 0 將縮放值寫入 `swizzledScaleIndex(row, blk)`。

## 成本分析

$$
Q = 2MK\ (\text{read}) + \frac{MK}{2} + \frac{MK}{16}\ (\text{write})\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：輸入 FP16，輸出封裝的 FP4 與 FP8 縮放值 |

約為 $2.56\,MK$ 位元組；受頻寬限制。

## 常見陷阱

- 輸入為 **FP16**（`float16*`），與 MX 問題不同。
- 縮放值使用**飽和有限值 E4M3**：$g\alpha/6$ 可能超過 448。
- **交錯縮放值需以零填補**。
- **除以解碼後的 E4M3 縮放值**，而非 $\alpha/6$。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [NVFP4 反量化](../nvfp4-dequantize/)、[NVFP4 GEMM](../nvfp4-gemm/)、[NVFP4 GEMV](../nvfp4-gemv/)、[MXFP4 量化](../mxfp4-quantize/)。
