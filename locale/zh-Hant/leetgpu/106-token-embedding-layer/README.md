---
title: 權杖嵌入層
platform: LeetGPU
upstream: medium/106_token_embedding_layer
url: https://leetgpu.com/challenges/token-embedding-layer
difficulty: medium
tags: [embedding, layernorm, gather, fusion]
status: solved
---

# 權杖嵌入層

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/token-embedding-layer)

## 題意

BERT 類模型的輸入層：對 $B\times T$ 個權杖中的每一個，擷取其權杖嵌入與位置嵌入，兩者相加後，再套用具有可學習 $\gamma, \beta$ 的 LayerNorm（$B \le 64$、$T \le 1024$、$V \le 50\,000$、$P \le 4096$、$D \le 1024$；效能測試為 $B = 32$、$T = 512$、$D = 768$；容許誤差 `1e-4`）。

## 圖解

![詞嵌入層：從兩張表各取一列相加，再對總和做 LayerNorm](figure.svg)

token 編號 3 從詞嵌入表取出一列，位置 1 從位置嵌入表取出一列。兩者相加後做 LayerNorm，只寫出一次。

## 數學表述

$$
\mathbf s_{b,t} = E_T[\tau_{b,t}] + E_P[\pi_t] \in \mathbb R^{D}, \qquad
\mu_{b,t} = \frac1D\sum_{d} s_{b,t,d}, \qquad
\sigma^2_{b,t} = \frac1D\sum_d\bigl(s_{b,t,d} - \mu_{b,t}\bigr)^2
$$

$$
y_{b,t,d} = \gamma_d\,\frac{s_{b,t,d} - \mu_{b,t}}{\sqrt{\sigma^2_{b,t} + \varepsilon}} + \beta_d
$$

| 符號 | 意義 |
|---|---|
| $B,\ T$ | 批次與序列長度 |
| $V,\ P$ | 詞彙表大小與位置數 |
| $D$ | 嵌入寬度 |
| $E_T \in \mathbb R^{V\times D}$ | 權杖嵌入表 |
| $E_P \in \mathbb R^{P\times D}$ | 位置嵌入表 |
| $\tau_{b,t}$ | 權杖 $(b, t)$ 的權杖 ID |
| $\pi_t$ | 時間步 $t$ 的位置 ID（由所有批次資料列共用） |
| $\mathbf s_{b,t}$ | 相加後的嵌入 |
| $\mu,\ \sigma^2$ | 資料列平均值與有偏變異數 |
| $\gamma_d,\ \beta_d$ | LayerNorm 的縮放與平移 |
| $y_{b,t,d}$ | 輸出，形狀為 $(B, T, D)$ |

## 解題思路

**每個權杖使用一個 warp**，並將整列保存在暫存器中：

1. 讀取 $\tau_{b,t}$ 與 $\pi_t$。lane $\ell$ 從兩個嵌入資料列擷取元素 $\ell, \ell+32, \dots$。每次 warp 存取都會讀取表格資料列中的連續 128 位元組區段；以資料列層級來看是*擷取*，但資料列內仍為合併存取。
2. 將 $s$ 保存在 `vals[32]` 中（32 × 32 = 1024 ≥ $D$）。迴圈具有編譯期上限，因此陣列會留在暫存器中。
3. **從暫存器分兩次計算統計量**：總和 → shuffle 歸約 → $\mu$；接著是 $\sum (s - \mu)^2$ → shuffle 歸約 → $\sigma^2$。先減去中心值再平方，可避免 $E[s^2] - \mu^2$ 的消去誤差；由於值已在暫存器中，第二次走訪不會增加記憶體流量。
4. 寫入 $\gamma_d (s - \mu)\,\text{rstd} + \beta_d$。

相加後的嵌入不會寫入記憶體：只需一個融合核心，而非 gather + add + LayerNorm。

## 成本分析

$$
Q \approx \underbrace{8BTD}_{\text{two gathered rows}} + \underbrace{4BTD}_{\text{output}} + \underbrace{8D}_{\gamma,\beta} \ \text{bytes}, \qquad W \approx 8BTD
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（位置表只有 $T$ 個不同資料列，第一個批次資料列之後都會命中 L2） |
| $W$ | 浮點運算次數（相加、兩次歸約、正規化、仿射轉換） |

效能測試：$B T D = 12.6$M 個元素，因此約有 100–150 MB 流量，耗時約 60 µs。核心受記憶體頻寬限制。

## 常見陷阱

- **位置 ID 由整個批次共用**：索引應為 `position_ids[t]`，而非 `[b, t]`。
- 若使用**無偏變異數**（除以 $D-1$），將無法符合容許誤差。
- **$D = 1024$ 時的暫存器用量**：每個 lane 使用 32 個浮點數再加上暫存值，仍在合理範圍。若 $D \gg 1024$，則應改為每列一個區塊。

## 驗證

所有 LeetGPU 測試案例都以 `1e-4` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $D < 32$ 與 $D = 1024$。

## 延伸閱讀

- [層正規化](../113-layer-normalization/)、[GPT-2 區塊](../074-gpt2-block/)、[ViT 圖塊嵌入](../118-vit-patch-embedding/)。
