---
title: Llama Transformer 區塊
platform: LeetGPU
upstream: hard/93_llama_transformer_block
url: https://leetgpu.com/challenges/llama-transformer-block
difficulty: hard
tags: [transformer, llama, gqa, rope, swiglu, rmsnorm, fusion]
status: solved
---

# Llama Transformer 區塊

**平台：** LeetGPU · **難度：** 困難 · [題目說明](https://leetgpu.com/challenges/llama-transformer-block)

## 問題

以 float32 實作一個 **LLaMA 風格的解碼器區塊**：$d = 512$，有 8 個查詢頭與 2 個寬度為 64 的 KV 頭（GQA），並使用 RoPE、因果注意力，以及隱藏寬度為 1408 的 SwiGLU MLP。所有投影都沒有偏置。輸入為 $x$（$S\times512$）、一個封裝的權重緩衝區，以及預先計算的 RoPE $\cos$/$\sin$ 表（$S \times 32$）；容許誤差 `1e-3`。相較於 [GPT-2 區塊](../074-gpt2-block/)，這裡的每個元件都是「現代」版本。

## 公式

$$
\begin{aligned}
X_1 &= \operatorname{RMSNorm}(X;\ \mathbf w_1), & Q &= X_1W_Q^{\mathsf T},\ \ K = X_1W_K^{\mathsf T},\ \ V = X_1W_V^{\mathsf T} \\
\tilde Q &= \operatorname{RoPE}(Q),\ \ \tilde K = \operatorname{RoPE}(K), & A_h &= \operatorname{softmax}\!\Bigl(\operatorname{mask}_{\text{causal}}\tfrac{\tilde Q_h\tilde K_{\lfloor h/4\rfloor}^{\mathsf T}}{8}\Bigr)V_{\lfloor h/4\rfloor} \\
X' &= X + \operatorname{Concat}(A_0..A_7)\,W_O^{\mathsf T}, & Y &= X' + \Bigl(\operatorname{SiLU}(X_2W_g^{\mathsf T})\odot X_2W_u^{\mathsf T}\Bigr)W_{\text{down}}^{\mathsf T},\ \ X_2 = \operatorname{RMSNorm}(X';\ \mathbf w_2)
\end{aligned}
$$

$$
\operatorname{RMSNorm}(\mathbf z; \mathbf w) = \frac{\mathbf z}{\sqrt{\frac1d\sum_i z_i^2 + 10^{-5}}}\odot\mathbf w, \qquad
\operatorname{RoPE}([\mathbf q_1\,|\,\mathbf q_2]) = [\mathbf q_1\odot\mathbf c - \mathbf q_2\odot\mathbf s\ \,|\ \,\mathbf q_1\odot\mathbf s + \mathbf q_2\odot\mathbf c]
$$

| 符號 | 意義 |
|---|---|
| $S$ | 序列長度 |
| $d$ | 模型寬度，512 |
| $X$ | 輸入，$S\times d$ |
| $\mathbf w_1,\ \mathbf w_2$ | RMSNorm 權重（長度 $d$） |
| $W_Q$ | $512\times512$（8 個頭 × 64），`nn.Linear` 配置（輸出、輸入） |
| $W_K,\ W_V$ | 各為 $128\times512$（2 個 KV 頭 × 64） |
| $\tilde Q_h,\ \tilde K_g$ | 經 RoPE 旋轉的查詢頭 $h$ 與鍵頭 $g$ |
| $\lfloor h/4\rfloor$ | GQA 對應：每個 KV 頭由 4 個查詢頭共用 |
| 8 | $\sqrt{64}$，注意力的縮放除數 |
| $\operatorname{mask}_{\text{causal}}$ | 對角線上方為 $-\infty$ |
| $\mathbf c,\ \mathbf s$ | 權杖的 RoPE $\cos$/$\sin$ 資料列（長度 32，由前後兩半共用） |
| $\mathbf q_1,\ \mathbf q_2$ | 一個頭向量中前後各 32 個元素 |
| $W_O$ | $512\times512$ 的輸出投影 |
| $W_g,\ W_u$ | $1408\times512$ 的閘門與向上投影 |
| $W_{\text{down}}$ | $512\times1408$ 的向下投影 |
| $Y$ | 區塊輸出 |

## 方法

| # | 核心 | 輸出 | 備註 |
|---|---|---|---|
| 1 | `rmsNormRows` | $X_1$ | 每列一個 warp |
| 2 | NT-GEMM，768 欄 | $[Q \mid K \mid V]$ | $W_Q, W_K, W_V$ 在緩衝區中**連續**排列，因此一次 GEMM 即可計算三者 |
| 3 | `applyRope` | 就地旋轉 $Q$ 與 $K$ 頭 | 每個執行緒處理配對 $(j, j+32)$ |
| 4 | Flash attention | $A$ | 因果、GQA `group = 4`，從封裝的 `qkv` 資料列進行跨距讀取 |
| 5 | NT-GEMM + `ResidualEpi` | $X' = X + AW_O^{\mathsf T}$ | 融合殘差 |
| 6 | `rmsNormRows` | $X_2$ | – |
| 7 | NT-GEMM，2816 欄 | $[G \mid U]$ | $W_g$ 與 $W_u$ 連續排列，因此只需一次 GEMM |
| 8 | `swiglu` | $\operatorname{SiLU}(G)\odot U$ | 逐元素運算 |
| 9 | NT-GEMM + `ResidualEpi` | $Y = X' + HW_{\text{down}}^{\mathsf T}$ | 融合殘差 |

**串接投影。** 將共用相同輸入的權重矩陣堆疊成一個較高的矩陣，可把 3 個（或 2 個）狹長 GEMM 合併成一個較大的 GEMM。輸入分塊只需載入一次，也能讓更多區塊平行執行。所有正式環境的 LLM 實作都以這種方式處理 QKV 與 gate/up。

**從封裝資料列計算注意力。** $Q$ 的頭 $h$ 從寬度 768 的資料列第 $64h$ 欄開始，$K$ 的頭 $g$ 從第 $512 + 64g$ 欄開始，$V$ 則從第 $640 + 64g$ 欄開始。注意力核心會收到這些位移與 768 的跨距，因此不需要 reshape 或 transpose 核心（請參閱 [GQA](../080-grouped-query-attention/)）。

## 成本分析

$$
W \approx 2S d\,(512 + 256) + 2Sd^2 + 2Sd\,(2\cdot1408) + 2S\cdot1408\,d + 2\cdot 8\cdot 64\,S^2
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數：QKV 投影、輸出投影、gate/up、down，以及因果注意力（約為密集 $4S^2 \cdot 512$ 的一半） |

當 $S = 2048$ 時：投影約需 9.8 GFLOP，注意力約需 4.3 GFLOP。GEMM 效率仍是主要影響因素。

## 常見問題

- **權重配置。** 所有投影都採用 `nn.Linear` 風格的 $(\text{out}, \text{in})$，亦即 $XW^{\mathsf T}$，因此使用 NT GEMM（與 GPT-2 問題相反）。
- **RoPE 表**有 32 欄，由每個寬度為 64 的頭之前後兩半共用。
- **GQA 對應**為 $h \mapsto \lfloor h/4 \rfloor$（連續的查詢頭會共用）。
- **所有位置都沒有偏置**，而且 RMSNorm 沒有 $\beta$。

## 驗證

所有 LeetGPU 測試案例都以 `1e-3` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過。

## 相關內容

- [GPT-2 區塊](../074-gpt2-block/)、[GQA](../080-grouped-query-attention/)、[RoPE](../061-rope-embedding/)、[SwiGLU MLP](../084-swiglu-mlp-block/)、[融合殘差 + RMSNorm](../083-fused-residual-add-rms-norm/)。
