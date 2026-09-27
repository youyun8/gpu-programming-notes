---
title: MXFP4 GEMM
platform: Tensara
upstream: mxfp4-gemm
url: https://tensara.org/problems/mxfp4-gemm
difficulty: hard
tags: [matmul, mxfp4, block-scaled, low-precision]
status: solved
---

# MXFP4 GEMM

**平台：** Tensara · **難度：** 困難 · [題目說明](https://tensara.org/problems/mxfp4-gemm)

## 題意

以 FP32 計算 $C = \hat{A}\hat{B}^{\mathsf T}$，其中 $A$（$M\times K$）和 $B$（$N\times K$）皆為 MXFP4 張量：元素是封裝的 E2M1，每 32 個元素共用一個 E8M0 縮放值，並採用交錯的 128×4 配置。參考實作為 `torch._scaled_mm`。檢查條件為 `rtol = 2e-2`、`atol = 5e-2`。

## 圖解

![MXFP4 GEMM：以區塊縮放的內積計算 C = Â B̂ᵀ（FP32）](figure.svg)

沿 K 方向，兩個列都被切成 32 個元素一組的區塊，各有自己的縮放 σ。區塊內直接相乘編碼值，兩個縮放係數每個區塊只套用一次。

## 數學表述

$$
c_{ij} = \sum_{\ell=0}^{K-1} \hat{A}_{i\ell}\,\hat{B}_{j\ell}, \qquad
\hat{A}_{i\ell} = \operatorname{e2m1}(c^A_{i\ell})\,2^{u^A_{i,\lfloor\ell/32\rfloor} - 127}, \qquad \hat{B}_{j\ell} = \operatorname{e2m1}(c^B_{j\ell})\,2^{u^B_{j,\lfloor\ell/32\rfloor} - 127}
$$

| 符號 | 意義 |
|---|---|
| $\hat{A}$ | 反量化後的 $A$，$M\times K$ |
| $\hat{B}$ | 反量化後的 $B$，儲存為 $N\times K$（因此乘積是 $\hat{A}\hat{B}^{\mathsf T}$，即「NT」GEMM） |
| $c$ | 輸出，$M\times N$（FP32） |
| $c^A, c^B$ | 4 位元編碼，每個位元組存兩個（低半位元組對應偶數 $\ell$） |
| $u^A, u^B$ | E8M0 縮放值位元組（交錯排列） |

### 依區塊重組總和

由於每個 32 元素區塊共用一個縮放值，可依區塊重新組合總和；張量核心的區塊縮放 MMA 正是採用這種方式：

$$
c_{ij} = \sum_{\beta=0}^{K/32 - 1} \sigma^{A}_{i\beta}\,\sigma^{B}_{j\beta} \sum_{\ell \in \beta} x^{A}_{i\ell}\,x^{B}_{j\ell}
$$

| 符號 | 意義 |
|---|---|
| $\beta$ | 沿 $K$ 的區塊索引 |
| $\sigma^A_{i\beta}, \sigma^B_{j\beta}$ | 兩個區塊縮放值 |
| $x^A, x^B$ | 解碼後、套用縮放前的元素值 |

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

### 交錯的縮放值配置

區塊縮放張量核心 MMA（cuBLAS / CUTLASS、TorchAO `is_swizzled_scales=True`、FlashInfer）會把含 $R$ 列、$C$ 個縮放值欄的矩陣，儲存在 512 位元組的 $128\times4$ 單元中：

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
2. **大小為 32 的 K 切片（一個縮放區塊）**：將 $A$ 與 $B$ 面板暫存到共享記憶體時，每個執行緒解碼元素編碼（先取半位元組，再呼叫 `e2m1ToFloat`），並乘上透過交錯索引查得的區塊縮放值。圖塊儲存 FP32，因此內層迴圈就是一般的 FMA 外積。
3. **結尾處理**：直接寫入 FP32。

反量化後的矩陣不會寫入全域記憶體；相較於 FP32 GEMM，唯一的額外成本是暫存時的解碼工作。在 Blackwell（sm_100）上，同一份資料可直接送入 `tcgen05.mma` 區塊縮放指令，由硬體讀取這些交錯縮放值配置；此可攜式核心則使用 CUDA 核心。

## 成本分析

$$
W = 2MNK, \qquad Q_{\min} = 0.5\,(MK + NK) + \frac{MK + NK}{32} + 4\,MN\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數 |
| $Q_{\min}$ | 必要的 DRAM 位元組數：封裝運算元、縮放值與輸出 |

FP4 運算元所占位元組數只有 FP32 的 1/8，因此運算元流量很小；當 $K$ 較小時，FP32 輸出往往主導 DRAM 流量。每個 E2M1 值乘以 2 的次方縮放值，在 FP32 中都能精確表示，因此只有累加時會發生捨入。

## 常見陷阱

- **半位元組順序**與**交錯縮放值**，與其他 MX 頁面相同。
- **位元組定址**：封裝酬載的第 $i$ 列從位元組 $iK/2$ 開始。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [MXFP8 GEMM](../mxfp8-gemm/)、[NVFP4 GEMM](../nvfp4-gemm/)、[MXFP4 量化](../mxfp4-quantize/)、LeetGPU [INT4 矩陣乘法](../../leetgpu/081-int4-matmul/)。
