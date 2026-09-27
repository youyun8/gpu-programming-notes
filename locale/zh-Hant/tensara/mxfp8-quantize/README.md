---
title: MXFP8 量化
platform: Tensara
upstream: mxfp8-quantize
url: https://tensara.org/problems/mxfp8-quantize
difficulty: medium
tags: [quantization, mxfp8, low-precision, warp-per-block]
status: solved
---

# MXFP8 量化

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/mxfp8-quantize)

## 題意

將 $M\times K$ FP32 矩陣量化為 **MXFP8**：每個元素使用一個 E4M3 位元組，且沿 $K$ 每 32 個元素使用一個 E8M0 縮放值（縮放值按列優先排列），結果需符合 TorchAO `to_mx`。尺寸最大為 $8192\times4096$。檢查器會反量化兩邊的輸出，再以 `rtol = atol = 1e-3` 比較。

## 圖解

![MXFP8 量化：每 32 個值共用一個 2 的冪縮放，元素採 E4M3](figure.svg)

區塊最大值 10.2 得到 E = −5，縮放為 1/32。縮放後的值有 3 個尾數位元，因此捨入得較細（例如 326.4 變成 320）。

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
| $e_{\max}$ | 元素格式可表示之最大 2 次方的指數（E4M3 為 8，其最大值為 $448 = 1.75\cdot 2^8$） |
| $E_b$ | 共用區塊指數 |
| $u_b$ | 儲存的 E8M0 縮放值位元組 |
| $q_t$ | 元素編碼，以最接近值捨入（同距時取偶數）至元素格式，並進行飽和處理 |
| $\hat{a}_t$ | 編碼所代表的值（檢查器反量化後比較的值） |

**E4M3（FP8）**包含 1 個符號位元、4 個指數位元和 3 個尾數位元，偏差值為 7，沒有無限大，而編碼 `0x7F`/`0xFF` 代表 NaN：

$$
\operatorname{e4m3}(b) = (-1)^{s}\cdot\begin{cases} \dfrac{f}{8}\cdot 2^{-6}, & e = 0 \ (\text{subnormal}) \\ \Bigl(1 + \dfrac{f}{8}\Bigr) 2^{e - 7}, & 1 \le e \le 15 \end{cases}, \qquad \lvert\operatorname{e4m3}\rvert \le 448
$$

| 符號 | 意義 |
|---|---|
| $b$ | 位元組 |
| $s, e, f$ | 符號位元、4 位元指數欄位、3 位元尾數欄位 |
| 448 | 最大有限值（$e = 15$、$f = 6$） |

縮放後的區塊最大值 $\alpha_b/2^{E_b}$ 會落在 $[256, 512)$，因此捨入前會將超過 448 的值截限為 $\pm448$。

## 解題思路

採用與 [MXFP4 量化](../mxfp4-quantize/) 相同的每區塊一個 warp 架構：以 warp shuffle 求最大值、從 float 位元讀取指數、截限至 $[-127, 128]$，再除以 $2^{E_b}$。E4M3 編碼器會截限至 $\pm448$ 並捨入至最接近的偶數，也會處理次正規數（$\lvert v\rvert < 2^{-6}$，間距為 $2^{-9}$）。每個 lane 寫入自己的位元組（每個 warp 進行一次合併存取的 32 位元組寫入），lane 0 則寫入縮放值。

## 成本分析

$$
Q = 4MK + MK + \frac{MK}{32}\ \text{bytes} \approx 5.03\,MK, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：輸入 FP32，每個元素輸出一個位元組，另加縮放值 |
| $\beta$ | DRAM 頻寬 |

當尺寸為 $8192\times4096$ 時：共 169 MB，在 2 TB/s 下約需 85 µs。

## 常見陷阱

- **$e_{\max} = 8$**，不是 7：E4M3 最大的正規 2 次方是 $2^8$。
- 捨入前先進行**飽和處理**（縮放後會出現 $(448, 512)$ 內的值）。
- **NaN 編碼**：在 E4M3 中，`0x7F` 是 NaN，因此最大的有限值編碼是 `0x7E`（448）。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [MXFP8 反量化](../mxfp8-dequantize/)、[MXFP8 GEMM](../mxfp8-gemm/)、[MXFP4 量化](../mxfp4-quantize/)。
