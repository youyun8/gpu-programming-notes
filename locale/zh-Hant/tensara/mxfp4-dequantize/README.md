---
title: MXFP4 反量化
platform: Tensara
upstream: mxfp4-dequantize
url: https://tensara.org/problems/mxfp4-dequantize
difficulty: easy
tags: [quantization, mxfp4, low-precision, elementwise]
status: solved
---

# MXFP4 反量化

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/mxfp4-dequantize)

## 題意

依照 TorchAO `MXTensor.to_dtype` 的語意，將 MXFP4 矩陣（封裝的 E2M1 編碼，加上每 32 個元素一個、按列優先排列的 E8M0 縮放值）還原成 $M\times K$ FP32 矩陣。尺寸最大為 $8192\times4096$；檢查條件為 `rtol = atol = 1e-3`。

## 圖解

![MXFP4 反量化：每個位元組解出兩個 E2M1 碼，再乘上區塊共用的 2 的冪](figure.svg)

每個位元組存放兩個 4 位元碼（低 nibble 在前）。解碼後得到較小的數值，整個區塊再乘上共用的縮放 2^(u − 127)。

## 數學表述

$$
\text{out}_{ij} = \operatorname{e2m1}\bigl(c_{ij}\bigr)\cdot \operatorname{e8m0}\bigl(u_{i,\lfloor j/32\rfloor}\bigr), \qquad
c_{ij} = \begin{cases} q_{i,j/2} \mathbin{\&} \text{0xF}, & j \text{ even} \\ q_{i,(j-1)/2} \gg 4, & j \text{ odd}\end{cases}
$$

| 符號 | 意義 |
|---|---|
| $q$ | 封裝的酬載，共 $M\times K/2$ 位元組 |
| $c_{ij}$ | 元素 $(i, j)$ 的 4 位元編碼 |
| $u$ | 縮放值位元組，$M\times K/32$，按列優先排列 |
| out | FP32 結果，$M\times K$ |

### E2M1（FP4）元素格式

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

### E8M0 區塊縮放值

**E8M0**（MX 區塊縮放值）是純粹的 2 次方：

$$
\operatorname{e8m0}(u) = 2^{\,u - 127}, \qquad u \in [0, 254], \quad u = 255 \Rightarrow \text{NaN}
$$

| 符號 | 意義 |
|---|---|
| $u$ | 縮放值位元組（帶偏差的指數） |

## 解題思路

**每個位元組使用一個執行緒**（兩個元素）：解碼兩個半位元組、乘上區塊縮放值 $2^{u - 127}$（使用 `ldexpf(1, u - 127)`，也能正確產生次正規數 $2^{-127}$），再寫入一個 `float2`（合併存取的 8 位元組寫入）。共用同一區塊的 16 個執行緒會讀取同一個縮放值位元組，因此可命中快取。每個 E2M1 值乘以 2 的次方，在 FP32 中都能精確表示，所以結果與參考實作逐位元相同。

## 成本分析

$$
Q = 0.5\,MK + \frac{MK}{32}\ (\text{read}) + 4MK\ (\text{write})\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數，主要來自 FP32 輸出 |
| $\beta$ | DRAM 頻寬 |

當尺寸為 $8192\times4096$ 時：約 153 MB，在 2 TB/s 下約需 77 µs。反量化會將資料膨脹 8 倍；在實際模型中，應將它融合進使用資料的運算（請見 [MXFP4 GEMM](../mxfp4-gemm/)）。

## 常見陷阱

- **半位元組順序**：低半位元組對應偶數欄。
- **E8M0 = 0** 代表 $2^{-127}$，即 float 次正規數；若以 float 位元（`u << 23`）建立縮放值，結果會變成 0，因此應使用 `ldexpf`。
- **縮放值 255** 代表 NaN。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [MXFP4 量化](../mxfp4-quantize/)、[MXFP8 反量化](../mxfp8-dequantize/)、[NVFP4 反量化](../nvfp4-dequantize/)。
