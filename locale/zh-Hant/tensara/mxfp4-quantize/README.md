---
title: MXFP4 量化
platform: Tensara
upstream: mxfp4-quantize
url: https://tensara.org/problems/mxfp4-quantize
difficulty: medium
tags: [quantization, mxfp4, low-precision, warp-per-block]
status: solved
---

# MXFP4 量化

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/mxfp4-quantize)

## 題意

將 $M\times K$ FP32 矩陣量化為 **MXFP4**：每個位元組存兩個 4 位元 E2M1 元素，且沿 $K$ 每 32 個元素使用一個 E8M0 縮放值，結果需符合 TorchAO 的 `MXTensor` 參考路徑。縮放值輸出按列優先排列（$M\times K/32$，不交錯）。尺寸最大為 $8192\times4096$。檢查器會反量化兩邊的輸出，再以 `rtol = atol = 1e-3` 比較，因此實際上編碼必須完全一致。

## 圖解

![MXFP4 量化：每 32 個值共用一個 2 的冪縮放，元素採 E2M1](figure.svg)

區塊最大值 10.2 決定指數 E = 1，因此縮放為 2。除以 2 後四捨五入到最接近的 E2M1 值，就得到下排的編碼值。

## 數學表述

**MX**（OCP Microscaling）張量會沿 $K$ 將每一列切成連續的 32 元素區塊；每個區塊共用一個 2 次方縮放值。依照 TorchAO `to_mx` 預設的 FLOOR 縮放值捨入方式：

$$
\alpha_b = \max_{t \in b} \lvert a_t \rvert, \qquad
E_b = \operatorname{clamp}\Bigl(\bigl\lfloor \log_2 \alpha_b \bigr\rfloor - e_{\max},\ -127,\ 128\Bigr), \qquad
u_b = E_b + 127
$$

$$
q_t = \operatorname{round}_{\text{fmt}\Bigl(\frac{a_t}{2^{E_b}\Bigr), \qquad \hat{a}_t = \operatorname{fmt}(q_t)\cdot 2^{E_b}
$$

| 符號 | 意義 |
|---|---|
| $b$ | 同一列中的 32 元素區塊 |
| $a_t$ | 輸入元素 |
| $\alpha_b$ | 區塊絕對值最大值 |
| $\lfloor\log_2\alpha_b\rfloor$ | float 的無偏差指數，從第 23–30 位元讀取 |
| $e_{\max}$ | 元素格式可表示之最大 2 次方的指數（E2M1 為 2，其最大值為 $6 = 1.5\cdot 2^2$） |
| $E_b$ | 共用區塊指數 |
| $u_b$ | 儲存的 E8M0 縮放值位元組 |
| $q_t$ | 元素編碼，以最接近值捨入（同距時取偶數）至元素格式，並進行飽和處理 |
| $\hat{a}_t$ | 編碼所代表的值（檢查器反量化後比較的值） |

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

因為 E2M1 最大的 2 次方是 $2^2$，縮放後的區塊最大值 $\alpha_b / 2^{E_b}$ 會落在 $[4, 8)$；大於 6 的值會飽和為 6。

## 解題思路

1. **每個 32 元素區塊使用一個 warp**（以網格跨步方式處理區塊）：lane $l$ 載入元素 $l$（合併存取的 128 位元組），經過五步 `__shfl_xor_sync` 最大值運算，讓每個 lane 都取得 $\alpha_b$。
2. **從位元讀取指數**：`(__float_as_uint(amax) >> 23) & 0xFF` 就是帶偏差的指數，因此取得 $\lfloor\log_2\alpha_b\rfloor$ 不需呼叫 `log2f`（次正規的 $\alpha_b$ 會比照 TorchAO 進行截限）。
3. **編碼**：$v = a_t / 2^{E_b}$（除以 2 的次方，因此精確），接著 `floatToE2M1` 會將 $\lvert v\rvert$ 與中點 $0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5$ 比較；透過嚴格或非嚴格比較實作同距取偶數（例如 0.25 → 0、0.75 → 1）。
4. **封裝**：`__shfl_down_sync(code, 1)` 取得相鄰奇數 lane 的編碼；偶數 lane 寫入 `code | (odd << 4)`。lane 0 寫入縮放值位元組（若區塊包含 NaN，則寫入 255）。

格式專屬部分（E2M1、E4M3、E8M0 編解碼器與交錯排列）皆以整數位元操作實作，因此不依賴 `cuda_fp4.h` 或特定架構。

## 成本分析

$$
Q = 4MK\ (\text{read}) + \frac{MK}{2} + \frac{MK}{32}\ (\text{write})\ \text{bytes} \approx 4.53\,MK, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| $\beta$ | DRAM 頻寬 |

當尺寸為 $8192\times4096$ 時：共 152 MB，在 2 TB/s 下約需 76 µs。寫入採用半個 warp 的位元組儲存（每個 warp 16 位元組）；若每 8 個 lane 封裝成 32 位元字組，便能加寬寫入。

## 常見陷阱

- **縮放值捨入模式**：FLOOR（TorchAO 的預設值）與 OCP 規格建議的捨入方式，會讓部分區塊得到不同指數。
- **同距值**：2.5 必須捨入為 2（偶數編碼），而非 3。
- **半位元組順序**：元素 $2i$ 位於低 4 位元。
- **全零區塊**：$\alpha_b = 0$ 的指數欄位為 0，會截限成 $E_b = -127$；所有編碼均為 0。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [MXFP4 反量化](../mxfp4-dequantize/)、[MXFP4 GEMM](../mxfp4-gemm/)、[MXFP8 量化](../mxfp8-quantize/)、[NVFP4 量化](../nvfp4-quantize/)、LeetGPU [權重反量化](../../leetgpu/064-weight-dequantization/)。
