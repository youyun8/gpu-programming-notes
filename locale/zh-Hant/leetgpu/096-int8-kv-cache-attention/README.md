---
title: INT8 KV 快取注意力
platform: LeetGPU
upstream: medium/96_int8_kv_cache_attention
url: https://leetgpu.com/challenges/int8-kv-cache-attention
difficulty: medium
tags: [attention, decode, flash-decoding, int8, split-k]
status: solved
---

# INT8 KV 快取注意力

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/int8-kv-cache-attention)

## 題意

**解碼階段**的多頭注意力：每個頭的一個新查詢權杖，會對儲存為 **int8** 且每個權杖各有縮放值的長 KV 快取進行注意力運算（$H \le 64$ 個頭、$S \le 32\,768$ 個快取權杖、頭維度 $8 \le D \le 256$；效能測試為 $H = 32$、$S = 8192$、$D = 128$；容許誤差 `1e-3`）。相較於 fp16，int8 快取可將記憶體流量減半；相較於 fp32，則只需四分之一。解碼注意力完全受記憶體頻寬限制，因此流量降低會直接轉化成速度提升。

## 圖解

![在 int8 KV 快取上做解碼注意力，並分散到多個區塊（flash-decoding）](figure.svg)

快取被切成數個片段，各由一個區塊處理並產生部分 softmax 狀態 (m, ℓ, a)，最後再精確地合併成最終輸出。

## 數學表述

$$
K_{h,j,c} = \kappa_{h,j}\,\hat K_{h,j,c}, \qquad V_{h,j,c} = \nu_{h,j}\,\hat V_{h,j,c}
$$

$$
s_{h,j} = \frac{\mathbf q_h\cdot K_{h,j,:}}{\sqrt D} = \frac{\kappa_{h,j}}{\sqrt D}\sum_c q_{h,c}\,\hat K_{h,j,c}, \qquad
\mathbf o_h = \sum_{j} \frac{e^{s_{h,j} - m_h}}{\sum_{j'} e^{s_{h,j'} - m_h}}\,\nu_{h,j}\,\hat V_{h,j,:}
$$

| 符號 | 意義 |
|---|---|
| $H,\ S,\ D$ | 頭數、快取序列長度、頭維度 |
| $\hat K,\ \hat V$ | 位於 $[-128, 127]$ 的 int8 快取值 |
| $\kappa_{h,j},\ \nu_{h,j}$ | 每個權杖的浮點縮放值（`k_scale`、`v_scale`） |
| $\mathbf q_h$ | 頭 $h$ 的查詢（長度 $D$） |
| $s_{h,j}$ | 快取權杖 $j$ 的注意力分數 |
| $m_h$ | 頭 $h$ 的最大分數 |
| $\mathbf o_h$ | 頭 $h$ 的輸出 |

縮放因子可從每個權杖的內部加總中提出，因此內積能直接對原始 int8 值運算，每個權杖只需縮放一次。

### Flash-Decoding（沿序列切分）

由於每個頭只有一個查詢，因此獨立問題只有 $H$ 個，例如在 108–132 個 SM 上只有 32 個區塊。將鍵切成由 256 個權杖組成的分段 $\mathcal C_1, \mathcal C_2, \dots$，為每個分段計算部分 softmax，再合併：

$$
m = \max_i m_i, \qquad \ell = \sum_i \ell_i\,e^{m_i - m}, \qquad \mathbf o = \frac{1}{\ell}\sum_i e^{m_i - m}\,\mathbf a_i
$$

| 符號 | 意義 |
|---|---|
| $m_i,\ \ell_i$ | 分段 $i$ 內的最大分數，以及 $e^{s - m_i}$ 的總和 |
| $\mathbf a_i$ | 未正規化的部分輸出 $\sum_{j\in\mathcal C_i} e^{s_j - m_i}V_j$ |
| $m,\ \ell,\ \mathbf o$ | 合併後的統計量與最終輸出 |

這與 [Softmax 注意力](../006-softmax-attention/)中可結合的 $(m, \ell, \mathbf a)$ 合併相同，只是此處在區塊之間套用，而非分塊之間。

## 解題思路

1. **`partialAttention`**，在效能測試中使用網格 $(\lceil S/256\rceil, H)$ = 1024 個區塊：
   - **分數**：每個鍵使用一個 warp，各 lane 處理 $D$ 維；將 int8 載入值轉成 float，以 shuffle 歸約，再乘上 $\kappa_j/\sqrt D$ 並存入共享記憶體；
   - **區域 softmax**：計算區塊最大值、指數及區塊總和；
   - **$PV$**：執行緒 $c$（沿 $D$ 維）累加 $\sum_j p_j\,\nu_j\hat V_{j,c}$。對每個鍵，$D$ 個執行緒會讀取連續的 int8 位元組，因此可合併存取。
   - 將 $(m_i, \ell_i, \mathbf a_i)$ 寫入暫存空間。
2. **`combine`**：每個頭使用一個區塊，依上述公式合併各分段的部分結果。

## 成本分析

$$
Q \approx 2HSD + 8HS + 4HD\cdot 2 + \text{partials}, \qquad W \approx 4HSD
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 位元組數：int8 K 與 V（各 1 位元組）、每個權杖兩個浮點縮放值、查詢與輸出 |
| $W$ | 浮點運算次數（分數與加權總和） |

效能測試：$Q \approx 67$ MB，而 float32 快取需 268 MB；在 2 TB/s 下約為 34 µs。$W/Q \approx 2$ FLOP/byte，因此核心明顯受頻寬限制，int8 快取相較於 fp32 確實可獲得 4 倍效益。

## 常見陷阱

- 若不切分，區塊數量會**太少**（每個頭一個區塊），導致多數 SM 閒置。
- **每個權杖各有縮放值**：應將 $\kappa$ 套用到整個內積，並將 $\nu$ 套用到值資料列，而不是對每個元素重複套用兩次。
- **不會有空分段**（$S \ge 1$），但最後一個分段可能未滿。

## 驗證

所有 LeetGPU 測試案例都以 `1e-3` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $S = 1$ 與 $S$ 無法被 256 整除的情況。

## 延伸閱讀

- [GQA](../080-grouped-query-attention/)、[Softmax 注意力](../006-softmax-attention/)、[INT8 矩陣乘法](../032-int8-quantized-matmul/)。
