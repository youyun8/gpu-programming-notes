---
title: Vision Transformer 圖塊嵌入
platform: LeetGPU
upstream: medium/118_vit_patch_embedding
url: https://leetgpu.com/challenges/vision-transformer-patch-embedding
difficulty: medium
tags: [gemm, im2col, vision, vit]
status: solved
---

# Vision Transformer 圖塊嵌入

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/vision-transformer-patch-embedding)

## 問題

**Vision Transformer**（ViT、CLIP、SigLIP、DiT）的圖塊嵌入前端。將 $B$ 張影像（$C\times H\times W$，NCHW）各自切成不重疊的 $P\times P$ 圖塊，把每個攤平的圖塊投影至 $D$ 維，在前方加上可學習的 CLS 權杖，再加上可學習的位置嵌入（容許誤差 `1e-4`）。這相當於核心大小 = 跨距 = $P$ 的卷積，並以 GEMM 實作。

## 公式

$$
g_h = \frac HP,\quad g_w = \frac WP,\quad N = g_hg_w, \qquad n = p_y g_w + p_x
$$

$$
t_{b,n,d} = \beta_d + \sum_{c=0}^{C-1}\sum_{i=0}^{P-1}\sum_{j=0}^{P-1} I_{b,c,\ p_yP + i,\ p_xP + j}\ \ W_{d,c,i,j}
$$

$$
Y_{b,0,:} = \mathbf{cls} + E_{0,:}, \qquad Y_{b,n+1,:} = \mathbf t_{b,n,:} + E_{n+1,:}
$$

| 符號 | 意義 |
|---|---|
| $B,\ C,\ H,\ W$ | 批次、通道數、影像高度與寬度 |
| $P$ | 圖塊大小（$H$、$W$ 可被 $P$ 整除） |
| $g_h,\ g_w$ | 圖塊網格的高度與寬度 |
| $N$ | 每張影像的圖塊數 |
| $n,\ (p_y, p_x)$ | 圖塊索引與其網格位置（列優先） |
| $I_{b,c,y,x}$ | 輸入像素 |
| $W_{d,c,i,j}$ | 投影權重（$D\times C\times P\times P$），亦即卷積核心 |
| $\beta_d$ | 投影偏置 |
| $t_{b,n,d}$ | 圖塊權杖 |
| $\mathbf{cls}$ | 可學習的 CLS 向量（不進行投影） |
| $E$ | 位置嵌入，$(N+1)\times D$ |
| $Y$ | 輸出，$B\times(N+1)\times D$ |

### 以 GEMM 表示（隱式 im2col）

依 $(c, i, j)$ 順序將每個圖塊攤平成長度 $K = CP^2$ 的資料列。圖塊矩陣 $\mathcal P \in \mathbb R^{BN\times K}$ 會得到

$$
T = \mathcal P\,W_{\text{flat}}^{\mathsf T} + \boldsymbol\beta, \qquad W_{\text{flat}} \in \mathbb R^{D\times CP^2}
$$

| 符號 | 意義 |
|---|---|
| $\mathcal P$ | im2col 矩陣：資料列 $bN + n$ 是影像 $b$ 的圖塊 $n$ |
| $W_{\text{flat}}$ | 重塑為 $D\times CP^2$ 的權重（列優先，因此這是「NT」GEMM） |

由於跨距 = 核心大小，圖塊不會重疊，im2col 也不會複製任何像素：$\mathcal P$ 純粹是影像的排列。

## 方法

- **`patchGemm`**：64 × 64 暫存器分塊的 NT-GEMM。其 **A 分塊載入器**會根據上述索引運算，從（資料列、$k$）計算來源像素 $(b, c, p_yP + i, p_xP + j)$（`patchPixel`），並直接從影像讀取。圖塊矩陣**永遠不會具體建立**。
- **尾聲**會加上 $\beta_d$ 與 $E_{n+1,d}$，並寫入輸出資料列 $b(N+1) + n + 1$，略過 CLS 位置。
- 一個小型核心會寫入 $B$ 個 CLS 資料列 $\mathbf{cls} + E_0$。

## 成本分析

$$
W = 2BN\cdot CP^2\cdot D = 2BCHW\cdot D, \qquad Q_{\min} = 4\bigl(BCHW + DCP^2 + B(N+1)D\bigr)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數；注意 $NP^2 = HW$ |
| $Q_{\min}$ | 必要位元組數：影像、權重、輸出 |

對 $224^2$ 影像上的 ViT-B/16（$C = 3$、$P = 16$、$D = 768$、$N = 196$），每張影像需要 $2\cdot3\cdot224^2\cdot768 = 231$ MFLOP。當 $K = 768$ 時，GEMM 稍微受運算能力限制。A 分塊的擷取載入會沿 $j$ 合併，也就是影像資料列中連續的 $P$ 個像素。

## 常見問題

- **攤平順序**必須符合權重配置 $(c, i, j)$，採通道優先，與參考實作的 `permute(0,2,4,1,3,5)` 相同。
- 每張影像的輸出區塊都要為 CLS 權杖保留 **$+1$ 的資料列位移**。
- **CLS 不會投影。** 它只會複製並加上 $E_0$。

## 驗證

所有 LeetGPU 測試案例都以 `1e-4` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $P = 1$（每個像素都是一個圖塊）與 $P = H = W$（只有一個圖塊）。

## 相關內容

- [DiT 區塊](../116-dit-block/)、[權杖嵌入](../106-token-embedding-layer/)、[二維卷積](../010-2d-convolution/)。Tensara [二維卷積](../../tensara/conv-2d/)。
