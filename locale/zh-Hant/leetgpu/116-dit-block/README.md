---
title: 擴散 Transformer 區塊
platform: LeetGPU
upstream: hard/116_dit_block
url: https://leetgpu.com/challenges/diffusion-transformer-block
difficulty: hard
tags: [transformer, diffusion, adaln, fusion, attention]
status: solved
cuemu_max_elements: 16777216
---

# 擴散 Transformer 區塊

**平台：** LeetGPU · **難度：** 困難 · [題目說明](https://leetgpu.com/challenges/diffusion-transformer-block)

## 問題

對一批圖塊權杖序列 $x \in \mathbb R^{B\times S\times 512}$ 執行一個 **DiT 區塊**（Diffusion Transformer：DiT、Stable Diffusion 3、Flux），並以每個樣本的向量 $c \in \mathbb R^{B\times512}$（時間步 + 類別／文字嵌入）作為條件。封裝的權重緩衝區包含 adaLN、QKV、輸出與 MLP 權重（容許誤差 `1e-3`）。與 LLM 區塊不同，正規化**沒有可學習的仿射參數**。其縮放、平移與殘差**閘門**會依每個樣本從 $c$ 預測（adaLN-Zero），因此批次中的每個樣本都會以不同方式正規化。

## 公式

調節（每個樣本一次）：

$$
\bigl[\boldsymbol\beta_1\,|\,\boldsymbol\gamma_1\,|\,\mathbf g_1\,|\,\boldsymbol\beta_2\,|\,\boldsymbol\gamma_2\,|\,\mathbf g_2\bigr] = \operatorname{SiLU}(\mathbf c)\,W_{\text{ada}}^{\mathsf T} + \mathbf b_{\text{ada}} \in \mathbb R^{6\cdot512}
$$

區塊（逐樣本，將六個向量廣播至全部 $S$ 個權杖）：

$$
\begin{aligned}
H &= \operatorname{LN}(X)\odot(1 + \boldsymbol\gamma_1) + \boldsymbol\beta_1, &
X' &= X + \mathbf g_1\odot\bigl(\operatorname{MHA}(H)\,W_o^{\mathsf T} + \mathbf b_o\bigr) \\
H' &= \operatorname{LN}(X')\odot(1 + \boldsymbol\gamma_2) + \boldsymbol\beta_2, &
Y &= X' + \mathbf g_2\odot\Bigl(\operatorname{GELU}_{\tanh}\bigl(H'W_1^{\mathsf T} + \mathbf b_1\bigr)W_2^{\mathsf T} + \mathbf b_2\Bigr)
\end{aligned}
$$

| 符號 | 意義 |
|---|---|
| $B,\ S$ | 批次大小與每個樣本的權杖數 |
| $X$ | 一個樣本的輸入權杖，$S\times512$ |
| $\mathbf c$ | 樣本的條件向量 |
| $W_{\text{ada}},\ \mathbf b_{\text{ada}}$ | adaLN 調節層，$3072\times512$ |
| $\boldsymbol\beta_k,\ \boldsymbol\gamma_k,\ \mathbf g_k$ | 子區塊 $k$ 的平移、縮放與閘門（MSA = 1、MLP = 2），各長 512 |
| LN | **不含**仿射參數的 LayerNorm |
| MHA | 8 頭自注意力，含 QKV 投影 $W_{qkv}$（$1536\times512$）與偏置，$d_h = 64$ |
| $W_o,\ \mathbf b_o$ | 注意力輸出投影 |
| $W_1,\ W_2$ | 含偏置的 MLP $512\to2048\to512$ |
| $\operatorname{GELU}_{\tanh}$ | tanh 近似 GELU |
| $\odot$ | 逐元素相乘，並跨權杖廣播 |
| $Y$ | 區塊輸出 |

**為何稱為「Zero」。** DiT 會初始化 $W_{\text{ada}}$，使 $\mathbf g_k = 0$。如此每個區塊一開始都是恆等映射，可穩定訓練非常深的擴散 Transformer。

## 方法

| # | 核心 | 融合 |
|---|---|---|
| 1 | 調節 GEMM $\operatorname{SiLU}(c)W_{\text{ada}}^{\mathsf T}$ | 載入 $c$ 時套用 SiLU，在尾聲加入偏置 |
| 2 | LayerNorm + **調節** | 以每列一個 warp 的核心計算 $\operatorname{LN}(x)(1+\gamma_1)+\beta_1$；由樣本索引選擇調節資料列 |
| 3 | QKV GEMM（+偏置） | – |
| 4 | 批次 Flash attention | grid.z = 批次，從封裝的資料列跨距讀取 Q/K/V |
| 5 | $W_o$ GEMM | 尾聲：$x + g_1\odot(v + b_o)$，即**閘控殘差** |
| 6 | LayerNorm + 調節 | 使用 $\gamma_2, \beta_2$ |
| 7 | FC1 GEMM | 尾聲：偏置 + GELU |
| 8 | FC2 GEMM | 尾聲：$x' + g_2\odot(v + b_2)$ |

尾聲函式物件會取得輸出資料列索引。它們透過 $b = \lfloor \text{row}/S\rfloor$ 推導樣本索引，再讀取正確的閘門向量，因此每樣本廣播不會產生額外成本。

## 成本分析

$$
W \approx BS\Bigl(2\cdot512\cdot1536 + 2\cdot512^2 + 2\cdot2\cdot512\cdot2048\Bigr) + 4BS^2\cdot512 + 2B\cdot512\cdot3072
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數：QKV、輸出投影、MLP（逐權杖）、注意力（二次方）、調節（逐樣本） |

當 $B = 4$ 且 $S = 1024$（32 × 32 潛在圖塊網格）時：投影約需 25 GFLOP，注意力約需 8.6 GFLOP。GEMM 占主要成本。融合可省去約 10 次逐元素走訪，資料範圍涵蓋 $B\cdot S\cdot 512$ 至寬度 2048 的張量。

## 常見問題

- **調節順序。** 寬度 3072 的輸出會依序切成 `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`。
- 使用 **$(1 + \gamma)$，而不是 $\gamma$。** 預測出的縮放值是以 1 為基準的殘差。
- **LayerNorm 沒有仿射參數**：不需套用 $w$/$b$ 向量。
- **每樣本廣播**：每個融合核心都必須將資料列對應至樣本索引。

## 驗證

所有 LeetGPU 測試案例都以 `1e-3` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，且大型案例使用提高後的 `cuemu_max_elements`。

## 相關內容

- [GPT-2 區塊](../074-gpt2-block/)、[LLaMA 區塊](../093-llama-transformer-block/)、[ViT 圖塊嵌入](../118-vit-patch-embedding/)、[群組正規化](../105-group-normalization/)。
