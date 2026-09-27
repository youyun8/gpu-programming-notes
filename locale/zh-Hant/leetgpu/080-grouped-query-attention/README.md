---
title: 分組查詢注意力
platform: LeetGPU
upstream: medium/80_grouped_query_attention
url: https://leetgpu.com/challenges/grouped-query-attention
difficulty: medium
tags: [attention, gqa, flash-attention, llm]
status: solved
---

# 分組查詢注意力

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/grouped-query-attention)

## 問題

實作 LLaMA-2/3 70B、Mistral 與 Gemma 所使用的分組查詢注意力
（Grouped-Query Attention，GQA）。$H_q$ 個 query 頭共用 $H_{kv}$ 個
key/value 頭，每組 $G = H_q/H_{kv}$ 個連續 query 頭會注意同一個 KV 頭。
$Q$ 的形狀為 $(H_q, S, D)$，$K$ 與 $V$ 的形狀為 $(H_{kv}, S, D)$
（$H_{kv} \le H_q \le 64$、$S \le 4096$、$8 \le D \le 256$；基準測試
$H_q = 32$、$H_{kv} = 8$、$S = 1024$、$D = 128$；容許誤差 `1e-4`）。
GQA 能以很小的品質損失，將 KV 快取縮小 $G\times$。

## 公式

$$
O_h = \operatorname{softmax}_{\text{row}}\!\Bigl(\frac{Q_h K_{g(h)}^{\mathsf T}}{\sqrt D}\Bigr)\,V_{g(h)}, \qquad g(h) = \Bigl\lfloor \frac{h}{G} \Bigr\rfloor, \qquad G = \frac{H_q}{H_{kv}}
$$

| 符號 | 意義 |
|---|---|
| $H_q,\ H_{kv}$ | Query 頭與 key/value 頭的數量 |
| $G$ | 群組大小（每個 KV 頭對應的 query 頭數） |
| $S$ | 序列長度 |
| $D$ | 每頭維度 |
| $Q_h$ | 第 $h$ 頭的 $S\times D$ query 矩陣（偏移量 $hSD$） |
| $K_{g},\ V_{g}$ | KV 頭 $g$ 的 key/value 矩陣 |
| $g(h)$ | query 頭 $h$ 使用的 KV 頭（連續的 query 頭共用一個） |
| $O_h$ | 第 $h$ 頭的輸出 |

**特殊情況：** $G = 1$ 是標準多頭注意力（MHA），而 $H_{kv} = 1$ 是
多查詢注意力（MQA）。

參考實作以 `repeat_interleave` 具體展開群組，也就是複製 $G$ 份 $K$ 與
$V$。核心只在計算指標時進行 $h \mapsto g(h)$ 對應。

## 方法

使用 FlashAttention 風格的單階段核心，不建立 $S\times S$ 矩陣：

- **網格**為 $\lceil S/8\rceil \times H_q$。一個區塊處理第 $h$ 頭的
  8 個連續 query 列（每列一個 warp），並透過共享記憶體，以每次 32 個
  key 的 tile 串流處理 KV 頭 $g(h)$。
- **每個 tile**：lane $\ell$ 對 key $\ell$ 計分（與共享記憶體中預先縮放
  的 query 計算完整 $D$ 維內積），接著執行一次 warp 最大值與一次 warp
  總和、線上 softmax 重新縮放，再以 `__shfl_sync` 廣播 $p_j$ 來更新 $PV$。
- **累加器**：lane $\ell$ 負責輸出欄 $\ell, \ell+32, \dots$；當 $D = 256$
  時最多使用 8 個暫存器。
- **動態共享記憶體**：$32(D+1) + 32D + 8D$ 個 float；當 $D = 256$ 時約為
  74 KB。這超過預設的 48 KB，因此啟動器透過
  `cudaFuncSetAttribute(…MaxDynamicSharedMemorySize…)` 明確選用較大容量。
  K tile 使用奇數跨距 $D+1$ 來避免 bank 衝突。

### GQA 在實務上為何很快

一個群組中的 $G$ 個 query 頭會讀取相同的 $K_g$ 與 $V_g$。它們的區塊會在
相近時間執行（`blockIdx.y` 連續），因此第一個區塊之後的 KV tile 會來自
L2，而非 DRAM。KV 流量最多可除以 $G$。對解碼（$S_q = 1$）而言，
這是主要的節省來源。

## 成本分析

$$
W = 4H_qS^2D, \qquad Q_{\min} = 4\bigl(2H_qSD + 2H_{kv}SD\bigr)
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP 數（每個 query 頭的分數與 $PV$） |
| $Q_{\min}$ | 必要位元組數：讀取 $Q$ 並寫入輸出（$H_q$ 個頭），讀取 $K$ 與 $V$（$H_{kv}$ 個頭） |

基準測試的 $W \approx 17$ GFLOP，且 $Q_{\min} = 42$ MB；若 MHA 使用
$H_{kv} = H_q$，則為 67 MB。此核心受計算限制，因此 GQA 的好處主要反映
在記憶體容量與解碼階段。

## 常見陷阱

- **注意力頭對應。** 連續 query 頭共用一個 KV 頭（`repeat_interleave`），
  即 $g = \lfloor h/G\rfloor$，而不是 $h \bmod H_{kv}$。
- **超過 48 KB 的共享記憶體**需要明確選用屬性，否則啟動會失敗。
- **縮放比例**為 $1/\sqrt D$。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-4`
通過，包括 $G = 1$（MHA）、$H_{kv} = 1$（MQA）及 $D = 256$。

## 相關內容

- [多頭注意力](../012-multi-head-attention/)、[多頭潛在注意力](../114-multi-head-latent-attention/)、
  [LLaMA 區塊](../093-llama-transformer-block/)、[INT8 KV 快取注意力](../096-int8-kv-cache-attention/)。
