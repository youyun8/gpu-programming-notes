---
title: SwiGLU MLP 區塊
platform: LeetGPU
upstream: medium/84_swiglu_mlp_block
url: https://leetgpu.com/challenges/swiglu-mlp-block
difficulty: medium
tags: [gemm, fusion, mlp, dual-gemm, llm]
status: solved
---

# SwiGLU MLP 區塊

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/swiglu-mlp-block)

## 題意

實作 LLaMA／Mistral／Gemma 的前饋區塊。輸入
$X \in \mathbb R^{M\times d}$ 會經過兩個平行投影（「gate」與「up」）擴展到
寬度 $d_f$，再經過 SiLU 閘門與下投影回到 $d$
（$M \le 65\,536$、$d \le 8192$、$d_f \le 32\,768$；基準測試
$M = 512$、$d = 4096$、$d_f = 14\,336$，即 LLaMA-3 8B 的 MLP；
容許誤差 `1e-4`）。三個 GEMM 約佔 LLM FLOP 的三分之二。可融合的部分，
就是它們之間的逐元素閘門。

## 圖解

![SwiGLU MLP：兩個輸入投影、一個融合的閘控、一個輸出投影](figure.svg)

gate 與 up 兩個投影讀取同一個 X；SiLU 閘控在該 GEMM 的 epilogue 中完成，因此在 down 投影之前，G 與 U 從不寫回記憶體。

## 數學表述

$$
G = XW_g, \qquad U = XW_u, \qquad H = \operatorname{SiLU}(G)\odot U, \qquad Y = HW_d
$$

$$
H_{mj} = \frac{G_{mj}}{1 + e^{-G_{mj}}}\cdot U_{mj}
$$

| 符號 | 意義 |
|---|---|
| $M$ | Token 數 |
| $d$ | 模型寬度（`d_model`） |
| $d_f$ | MLP 隱藏寬度（`d_ffn`，通常約為經捨入的 $\approx \tfrac83 d$） |
| $X$ | 輸入，$M\times d$ |
| $W_g,\ W_u$ | Gate 與 up 投影，$d\times d_f$（以 (in, out) 儲存） |
| $W_d$ | 下投影，$d_f\times d$ |
| $G,\ U$ | Gate 與 up 活化值，$M\times d_f$ |
| $\odot$ | 逐元素乘積 |
| $H$ | 經閘門處理的隱藏活化值 |
| $Y$ | 輸出，$M\times d$ |

## 解題思路

### 核心 1：具有融合閘門的雙 GEMM

`gemm<true>` 同時計算 $G$ **與** $U$ 的 tile：

- 每個 $K$ 切片只暫存**一次** $X$ 的 $64\times16$ tile。相對應的
  $W_g$ 與 $W_u$ 的 $16\times64$ tile 會並排暫存。
- 每個執行緒保留**兩組** $4\times4$ 累加器（32 個暫存器），並對載入的
  每個 $A$ 值執行 2 次 FMA。
- 結尾運算計算 $\frac{g}{1+e^{-g}}\cdot u$，且只寫入 $H$。

若不使用雙 GEMM，就需要啟動兩個 GEMM（讀取 $X$ 兩次）、寫入 $G$ 與 $U$
（$2Md_f$ 個 float），再由逐元素核心重新讀取兩者。

### 核心 2：$Y = HW_d$

以單一模式使用相同的暫存器分塊範本。

## 成本分析

$$
W = 2Md\,d_f\cdot 2 + 2Md_f\,d = 6Md\,d_f, \qquad
Q_{\text{saved}} = 4\cdot\bigl(2Md_f\ \text{(write } G, U) + 2Md_f\ \text{(read } G, U) + Md\ \text{(re-read } X)\bigr)
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP 數：兩個 $d\times d_f$ 投影加上一個 $d_f\times d$ 投影 |
| $Q_{\text{saved}}$ | 雙 GEMM 融合省下的 DRAM 位元組數 |

基準測試的 $W = 6\cdot512\cdot4096\cdot14336 \approx 180$ GFLOP，而
$Q_{\text{saved}} \approx 120$ MB。當 $M = 512$ 時，權重佔主要流量
（$3\cdot 4\cdot d\,d_f = 705$ MB，每個 64 列 tile 帶大約讀取一次）。
核心受 fp32 FMA 的計算限制。正式環境中的核心會使用 bf16 tensor core，
且在 $M$ 較小時切分 $K$ 以填滿 GPU。

## 常見陷阱

- **權重配置**為 (in, out)：計算 $XW$，而非 $XW^{\mathsf T}$。
- **SiLU 只套用於 gate**，再乘上*未活化*的 up 投影。
- **暫存器壓力。** 兩組 16-float 累加器加上 fragment 仍低於 255 個
  暫存器的限制，因此不會溢出到記憶體。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-4`
通過。

## 延伸閱讀

- [SwiGLU](../054-swiglu/)、[LLaMA Transformer 區塊](../093-llama-transformer-block/)、
  [GPT-2 區塊](../074-gpt2-block/)（GELU MLP）、[MoE Top-k 閘門](../067-moe-topk-gating/)。
