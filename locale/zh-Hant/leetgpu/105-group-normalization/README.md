---
title: 群組正規化
platform: LeetGPU
upstream: medium/105_group_normalization
url: https://leetgpu.com/challenges/group-normalization
difficulty: medium
tags: [normalization, row-reduction, cnn]
status: solved
---

# 群組正規化

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/group-normalization)

## 問題

對 NCHW 張量執行群組正規化。將 $C$ 個通道切成 $G$ 個連續群組，並對每個（樣本、群組）的 $(C/G)\cdot H\cdot W$ 個元素進行正規化，再依通道縮放與平移（$N \le 32$、$C \le 1024$；容許誤差 `1e-4`）。GroupNorm 是 Stable Diffusion U-Net 與許多 ResNet 所使用的正規化。它不像 BatchNorm，結果不受批次大小影響。$G = 1$ 時等同 LayerNorm，$G = C$ 時等同 InstanceNorm。

## 公式

$$
\mathcal S_{n,g} = \Bigl\{(c, h, w) : g\tfrac{C}{G} \le c < (g+1)\tfrac{C}{G}\Bigr\}, \qquad
\mu_{n,g} = \frac{1}{\lvert\mathcal S\rvert}\sum_{\mathcal S_{n,g}} x_{n,c,h,w}, \qquad
\sigma^2_{n,g} = \frac{1}{\lvert\mathcal S\rvert}\sum_{\mathcal S_{n,g}}\bigl(x_{n,c,h,w} - \mu_{n,g}\bigr)^2
$$

$$
y_{n,c,h,w} = \gamma_c\,\frac{x_{n,c,h,w} - \mu_{n,g(c)}}{\sqrt{\sigma^2_{n,g(c)} + \varepsilon}} + \beta_c, \qquad g(c) = \Bigl\lfloor\frac{cG}{C}\Bigr\rfloor
$$

| 符號 | 意義 |
|---|---|
| $N,\ C,\ H,\ W$ | 批次、通道數、高度、寬度 |
| $G$ | 群組數（$C$ 可被 $G$ 整除） |
| $\mathcal S_{n,g}$ | 樣本 $n$ 中群組 $g$ 的元素；$\lvert\mathcal S\rvert = (C/G)HW$ |
| $\mu_{n,g},\ \sigma^2_{n,g}$ | 群組平均值與有偏變異數 |
| $g(c)$ | 通道 $c$ 所屬的群組 |
| $\gamma_c,\ \beta_c$ | 每通道仿射參數 |
| $\varepsilon$ | 穩定常數 |

**連續性。** 在 NCHW 配置中，樣本 $n$ 的通道 $g\frac CG \dots (g+1)\frac CG - 1$ 是一個由 $\lvert\mathcal S\rvert$ 個浮點數構成的連續區塊，起點為 $\bigl(nC + g\tfrac CG\bigr)HW$。每個群組都是一個平坦的連續資料列。

## 方法

**每個 (n, g) 使用一個含 256 個執行緒的區塊**（網格為 $N\cdot G$）：

1. 以網格跨距走訪群組中的連續元素，並以 **float64** 累加 $\sum x$ 與 $\sum x^2$。合併式區塊歸約會在一次走訪中處理兩個總和：先進行 warp shuffle，再透過共享記憶體傳遞。
2. 計算 $\mu = \sum x/\lvert\mathcal S\rvert$ 與 $\sigma^2 = \sum x^2/\lvert\mathcal S\rvert - \mu^2$；在 float64 下很安全，且結果會限制為不小於 0。接著計算 $\text{rstd} = 1/\sqrt{\sigma^2 + \varepsilon}$。
3. 第二次走訪：$y = (x - \mu)\cdot\text{rstd}\cdot\gamma_c + \beta_c$，其中 $c = g\frac CG + \lfloor i/(HW)\rfloor$。

## 成本分析

$$
Q = 12\,NCHW\ \text{bytes} \quad(\text{read twice, write once}), \qquad \text{parallelism} = N\cdot G\ \text{blocks}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（群組大小適中時，第二次讀取通常會命中 L2） |
| 平行度 | 獨立區塊數量 |

當群組少而 $HW$ 很大時（例如 $N = 1$、$G = 32$、特徵圖為 $64\times64$），只有 32 個區塊會執行。若將每個群組切給多個區塊，再使用兩層歸約（如[批次正規化](../040-batch-normalization/)），即可使用更多 SM。

## 常見問題

- 使用 **float32 計算 $E[x^2] - \mu^2$**時，若平均值很大會損失精度。以 float64 累加可避免此問題。
- 群組中一個元素的**通道**是 $\lfloor i/HW\rfloor$，而不是 $i \bmod (C/G)$。

## 驗證

所有 LeetGPU 測試案例都以 `1e-4` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $G = 1$ 與 $G = C$。

## 相關內容

- [批次正規化](../040-batch-normalization/)、[層正規化](../113-layer-normalization/)、[RMS 正規化](../050-rms-normalization/)、[DiT 區塊](../116-dit-block/)。
