---
title: MXFP8 反量化
platform: Tensara
upstream: mxfp8-dequantize
url: https://tensara.org/problems/mxfp8-dequantize
difficulty: easy
tags: [quantization, mxfp8, low-precision, elementwise]
status: solved
---

# MXFP8 反量化

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/mxfp8-dequantize)

## 題意

依照 TorchAO 的語意，將 MXFP8 矩陣（E4M3 位元組，加上每 32 個元素一個、按列優先排列的 E8M0 縮放值）還原成 FP32。尺寸最大為 $8192\times4096$；檢查條件為 `rtol = atol = 1e-3`。

## 圖解

![MXFP8 反量化：每個 E4M3 位元組乘上區塊的 2 的冪縮放](figure.svg)

每個位元組解碼成一個 E4M3 值，再乘上區塊縮放 2^(u − 127)；兩個步驟都是精確的。

## 數學表述

$$
\text{out}_{ij} = \operatorname{e4m3}\bigl(q_{ij}\bigr)\cdot\operatorname{e8m0}\bigl(u_{i,\lfloor j/32\rfloor}\bigr)
$$

| 符號 | 意義 |
|---|---|
| $q_{ij}$ | 元素 $(i, j)$ 的 E4M3 位元組 |
| $u$ | 縮放值位元組，$M\times K/32$ |
| out | FP32 結果 |

### E4M3（FP8）格式

**E4M3（FP8）**包含 1 個符號位元、4 個指數位元和 3 個尾數位元，偏差值為 7，沒有無限大，而編碼 `0x7F`/`0xFF` 代表 NaN：

$$
\operatorname{e4m3}(b) = (-1)^{s}\cdot\begin{cases} \dfrac{f}{8}\cdot 2^{-6}, & e = 0 \ (\text{subnormal}) \\ \Bigl(1 + \dfrac{f}{8}\Bigr) 2^{e - 7}, & 1 \le e \le 15 \end{cases}, \qquad \lvert\operatorname{e4m3}\rvert \le 448
$$

| 符號 | 意義 |
|---|---|
| $b$ | 位元組 |
| $s, e, f$ | 符號位元、4 位元指數欄位、3 位元尾數欄位 |
| 448 | 最大有限值（$e = 15$、$f = 6$） |

### E8M0 區塊縮放值

**E8M0**（MX 區塊縮放值）是純粹的 2 次方：

$$
\operatorname{e8m0}(u) = 2^{\,u - 127}, \qquad u \in [0, 254], \quad u = 255 \Rightarrow \text{NaN}
$$

| 符號 | 意義 |
|---|---|
| $u$ | 縮放值位元組（帶偏差的指數） |

## 解題思路

每個元素使用一個執行緒（網格跨步）：以整數運算解碼位元組（`e4m3ToFloat`：取出指數與尾數欄位，並分別處理次正規數和 NaN）、乘上區塊縮放值，再寫入。E4M3 解碼也可在共享記憶體中使用含 256 個項目的查詢表，但算術版本的成本已隱藏在記憶體流量之後。

## 成本分析

$$
Q = 1\,MK + \frac{MK}{32}\ (\text{read}) + 4MK\ (\text{write})\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數，主要來自 FP32 輸出 |
| $\beta$ | DRAM 頻寬 |

當尺寸為 $8192\times4096$ 時：約 169 MB，在 2 TB/s 下約需 85 µs。

## 常見陷阱

- **次正規數**：$e = 0$ 代表 $f/8\cdot2^{-6}$，而非 $(1 + f/8)2^{-7}$。
- **NaN** 只有 `0x7F`/`0xFF`；E4M3 沒有無限大。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [MXFP8 量化](../mxfp8-quantize/)、[MXFP8 GEMM](../mxfp8-gemm/)、[MXFP4 反量化](../mxfp4-dequantize/)。
