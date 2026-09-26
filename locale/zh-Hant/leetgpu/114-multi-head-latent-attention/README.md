---
title: 多頭潛在注意力解碼
platform: LeetGPU
upstream: hard/114_multi_head_latent_attention
url: https://leetgpu.com/challenges/multi-head-latent-attention-decode
difficulty: hard
tags: [attention, mla, deepseek, decode, weight-absorption]
status: solved
---

# 多頭潛在注意力解碼

**平台：** LeetGPU · **難度：** 困難 · [題目說明](https://leetgpu.com/challenges/multi-head-latent-attention-decode)

## 問題

DeepSeek-V2/V3 **多頭潛在注意力**的一個**解碼步驟**。KV 快取不儲存每個頭各自的鍵與值，而是為每個位置儲存一個低秩潛在向量 $\mathbf c_t$（寬度 $R$，`kv_lora_rank`），以及一個較小、共用的旋轉鍵 $\mathbf k^{\text{pe}}_t$（寬度 $r$）。每個頭的向上投影 $W_{UK}$ 與 $W_{UV}$ 會隱式重建鍵和值。利用權重吸收後，注意力可完全在潛在空間中運算（容許誤差 `1e-3`）。與 MHA 相比，MLA 可將 KV 快取記憶體減少超過一個數量級。

## 公式

對頭 $h$，其查詢為 $\mathbf q_h = [\mathbf q^{\text{nope}}_h\,|\,\mathbf q^{\text{pe}}_h]$，而快取資料列 $t$ = $[\mathbf c_t\,|\,\mathbf k^{\text{pe}}_t]$：

$$
\tilde{\mathbf q}_h = \mathbf q^{\text{nope}}_h\,W_{UK,h}, \qquad
s_{h,t} = \frac{\tilde{\mathbf q}_h\cdot\mathbf c_t + \mathbf q^{\text{pe}}_h\cdot\mathbf k^{\text{pe}}_t}{\sqrt{d_h + r}}, \qquad
\mathbf o_h = \Bigl(\sum_{t} \operatorname{softmax}_t(s_h)_t\,\mathbf c_t\Bigr)\,W_{UV,h}
$$

| 符號 | 意義 |
|---|---|
| $H$ | 頭數 |
| $T$ | 快取位置數（`seq_len`） |
| $R$ | 潛在寬度（`kv_lora_rank`，DeepSeek-V3 中為 512） |
| $d_h$ | 內容部分的頭維度（`head_dim`） |
| $r$ | 旋轉維度（`rope_dim`，64） |
| $\mathbf q^{\text{nope}}_h,\ \mathbf q^{\text{pe}}_h$ | 頭 $h$ 查詢的內容部分與旋轉部分 |
| $\mathbf c_t$ | 位置 $t$ 的潛在向量，由所有頭共用（同時充當鍵和值） |
| $\mathbf k^{\text{pe}}_t$ | 共用的旋轉鍵 |
| $W_{UK,h}$ | 頭 $h$ 的鍵向上投影，$d_h\times R$ |
| $W_{UV,h}$ | 頭 $h$ 的值向上投影，$R\times d_h$ |
| $\tilde{\mathbf q}_h$ | 潛在空間中「吸收後」的查詢（長度 $R$） |
| $s_{h,t}$ | 分數 |
| $\mathbf o_h$ | 頭 $h$ 的輸出（長度 $d_h$） |

### 權重吸收

頭 $h$ 重建後的鍵原本是 $\mathbf k_{h,t} = \mathbf c_tW_{UK,h}^{\mathsf T}$，因此 $\mathbf q^{\text{nope}}_h\cdot\mathbf k_{h,t} = (\mathbf q^{\text{nope}}_hW_{UK,h})\cdot\mathbf c_t$。將 $W_{UK}$ 移到（唯一的）查詢上，只需一次小型 GEMV，無須重建 $T$ 個鍵。同理，$\sum_t a_t(\mathbf c_tW_{UV,h}) = (\sum_t a_t\mathbf c_t)W_{UV,h}$ 可在加權總和後只套用 $W_{UV}$ **一次**。

## 方法

吸收後，問題就是使用一個共用 KV 頭的**一般注意力（MQA）**：鍵向量為 $[\mathbf c_t\,|\,\mathbf k^{\text{pe}}_t]$（寬度 $R + r \le 576$），值向量為 $\mathbf c_t$（寬度 $R \le 512$）。

1. **`absorbQuery`**：串接查詢 $[\tilde{\mathbf q}_h\,|\,\mathbf q^{\text{pe}}_h]$ 的每個 $(h, i)$ 使用一個執行緒。當 $i < R$ 時，計算 $W_{UK,h}$ 第 $i$ 欄與寬度 $d_h$ 向量的內積；否則複製旋轉部分。
2. **`latentAttention`**：Flash 風格，**每區塊 4 個頭**（各一個 warp）。所有頭共用快取，因此每個暫存在共享記憶體中的 32 列快取分塊可供 4 個頭使用（MQA 重複使用）。資料列會以寬度 128 的欄切片串流處理分數（最大寬度 576）和值（寬度 512，每個 lane 使用 16 個暫存器累加器）。照常使用線上 softmax。輸出為潛在向量 $\sum_t a_{h,t}\mathbf c_t$。
3. **`upProject`**：每個 $(h, j)$ 使用一個執行緒，計算與 $W_{UV,h}$ 第 $j$ 欄的 $R$ 維內積。

## 成本分析

$$
\text{cache bytes per token} = 4(R + r)\ \ \text{vs.}\ \ 4\cdot 2Hd_h\ \text{(MHA)}, \qquad
W \approx 2Hd_hR + 2HT(2R + r) + 2HRd_h
$$

| 符號 | 意義 |
|---|---|
| 快取位元組數 | 每個位置的 KV 快取用量：MLA 儲存一個潛在資料列；MHA 則為每個頭儲存鍵和值 |
| $W$ | 浮點運算次數：吸收、對 $T$ 個位置計算分數與加權總和、向上投影 |

使用 DeepSeek-V3 的數值（$H = 128$、$d_h = 128$、$R = 512$、$r = 64$）時，每個權杖只需 576 個值，而非 32 768 個，亦即縮小 57 倍。解碼只需為所有頭讀取一次快取，因此核心受 $4T(R + r)$ 位元組的記憶體頻寬限制。

## 常見問題

- **縮放。** 使用 $1/\sqrt{d_h + r}$，也就是*重建後*的頭寬度加上 rope 寬度，而非 $R$。
- **共用旋轉鍵。** 所有頭使用相同的 $\mathbf k^{\text{pe}}$，只有查詢的 rope 部分是每個頭獨有。
- **暫存器預算。** 每個 lane 使用 16 個值累加器（$R = 512$），再加上分數迴圈；由於切片迴圈具有編譯期上限，因此仍可符合限制。

## 驗證

所有 LeetGPU 測試案例都以 `1e-3` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，並以類似 DeepSeek 的大小（$R = 512$、$r = 64$）進行壓力測試。

## 相關內容

- [GQA](../080-grouped-query-attention/)、[INT8 KV 快取注意力](../096-int8-kv-cache-attention/)、[RoPE](../061-rope-embedding/)。
