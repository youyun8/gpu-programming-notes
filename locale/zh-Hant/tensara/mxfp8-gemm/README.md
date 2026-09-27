---
title: MXFP8 GEMM
platform: Tensara
upstream: mxfp8-gemm
url: https://tensara.org/problems/mxfp8-gemm
difficulty: hard
tags: [matmul, mxfp8, block-scaled, low-precision]
status: solved
---

# MXFP8 GEMM

**平台：** Tensara · **難度：** 困難 · [題目說明](https://tensara.org/problems/mxfp8-gemm)

## 題意

以 FP32 計算 $C = \hat{A}\hat{B}^{\mathsf T}$，其中 $A$（$M\times K$）和 $B$（$N\times K$）皆為 MXFP8 張量：每個元素是 E4M3，每 32 個元素共用一個 E8M0 縮放值，縮放值採用 `to_mx(..., is_swizzled_scales=True)` 產生的**交錯 128×4 配置**。參考實作為 `torch._scaled_mm`。檢查條件為 `rtol = 2e-2`、`atol = 5e-2`。

## 圖解

![MXFP8 GEMM：32 個一組的 E4M3 區塊，搭配 2 的冪縮放，C = Â B̂ᵀ](figure.svg)

結構與 MXFP4 相同，只是元素改為 E4M3：區塊乘積在低精度 Tensor Core 上計算，每個區塊貢獻 σᴬ σᴮ 乘上其部分和。

## 數學表述

$$
c_{ij} = \sum_{\ell=0}^{K-1} \hat{A}_{i\ell}\,\hat{B}_{j\ell}, \qquad
\hat{A}_{i\ell} = \operatorname{e4m3}(q^A_{i\ell})\,2^{u^A_{i,\lfloor\ell/32\rfloor} - 127}, \qquad \hat{B}_{j\ell} = \operatorname{e4m3}(q^B_{j\ell})\,2^{u^B_{j,\lfloor\ell/32\rfloor} - 127}
$$

| 符號 | 意義 |
|---|---|
| $\hat{A}$ | 反量化後的 $A$，$M\times K$ |
| $\hat{B}$ | 反量化後的 $B$，儲存為 $N\times K$（因此乘積是 $\hat{A}\hat{B}^{\mathsf T}$，即「NT」GEMM） |
| $c$ | 輸出，$M\times N$（FP32） |
| $u^A, u^B$ | E8M0 縮放值位元組（交錯排列） |

由於每個 32 元素區塊共用一個縮放值，可依區塊重新組合總和；張量核心的區塊縮放 MMA 正是採用這種方式：

$$
c_{ij} = \sum_{\beta=0}^{K/32 - 1} \sigma^{A}_{i\beta}\,\sigma^{B}_{j\beta} \sum_{\ell \in \beta} x^{A}_{i\ell}\,x^{B}_{j\ell}
$$

| 符號 | 意義 |
|---|---|
| $\beta$ | 沿 $K$ 的區塊索引 |
| $\sigma^A_{i\beta}, \sigma^B_{j\beta}$ | 兩個區塊縮放值 |
| $x^A, x^B$ | 解碼後、套用縮放前的元素值 |

**E4M3（FP8）**包含 1 個符號位元、4 個指數位元和 3 個尾數位元，偏差值為 7，沒有無限大，而編碼 `0x7F`/`0xFF` 代表 NaN：

$$
\operatorname{e4m3}(b) = (-1)^{s}\cdot\begin{cases} \dfrac{f}{8}\cdot 2^{-6}, & e = 0 \ (\text{subnormal}) \\ \Bigl(1 + \dfrac{f}{8}\Bigr) 2^{e - 7}, & 1 \le e \le 15 \end{cases}, \qquad \lvert\operatorname{e4m3}\rvert \le 448
$$

| 符號 | 意義 |
|---|---|
| $b$ | 位元組 |
| $s, e, f$ | 符號位元、4 位元指數欄位、3 位元尾數欄位 |
| 448 | 最大有限值（$e = 15$、$f = 6$） |

**E8M0**（MX 區塊縮放值）是純粹的 2 次方：

$$
\operatorname{e8m0}(u) = 2^{\,u - 127}, \qquad u \in [0, 254], \quad u = 255 \Rightarrow \text{NaN}
$$

| 符號 | 意義 |
|---|---|
| $u$ | 縮放值位元組（帶偏差的指數） |

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
2. **大小為 32 的 K 切片（一個縮放區塊）**：將 $A$ 與 $B$ 面板暫存到共享記憶體時，每個執行緒解碼元素編碼（`e4m3ToFloat`），並乘上透過交錯索引查得的區塊縮放值。圖塊儲存 FP32，因此內層迴圈就是一般的 FMA 外積。
3. **結尾處理**：直接寫入 FP32（`global_scale = 1`）。

反量化後的矩陣不會寫入全域記憶體；相較於 FP32 GEMM，唯一的額外成本是暫存時的解碼工作。在 Blackwell（sm_100）上，同一份資料可直接送入 `tcgen05.mma` 區塊縮放指令，由硬體讀取這些交錯縮放值配置；此可攜式核心則使用 CUDA 核心。

## 成本分析

$$
W = 2MNK, \qquad Q_{\min} = 1\,(MK + NK) + \frac{MK + NK}{32} + 4\,MN\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數 |
| $Q_{\min}$ | 必要的 DRAM 位元組數：封裝運算元、縮放值與輸出 |

FP8 運算元比 FP32 小 4 倍，因此對中等尺寸而言，此核心甚至比 SGEMM 更受運算量限制；每個暫存元素只需幾個整數解碼運算，且其成本可分攤到 64 次 FMA。

## 常見陷阱

- **交錯縮放值**：若以按列優先方式建立索引，所有 $r \bmod 128 \ge 32$ 的區塊都會讀到錯誤縮放值。
- **$B$ 是 $N\times K$**：這是「NT」乘積。
- **容許誤差**：參考實作在張量核心上使用不同順序累加；`atol = 5e-2` 可涵蓋這項差異。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [MXFP4 GEMM](../mxfp4-gemm/)、[NVFP4 GEMM](../nvfp4-gemm/)、[MXFP8 量化](../mxfp8-quantize/)、[矩陣乘法](../matrix-multiplication/)。
