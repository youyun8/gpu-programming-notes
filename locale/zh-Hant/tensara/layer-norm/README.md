---
title: 層正規化
platform: Tensara
upstream: layer-norm
url: https://tensara.org/problems/layer-norm
difficulty: medium
tags: [normalization, reduction, fp64-accumulation]
status: solved
---

# 層正規化

**平台：** Tensara · **難度：** medium · [題目說明](https://tensara.org/problems/layer-norm)

## 題意

對形狀為 $(B, F, D_1, D_2)$ 的 float32 張量，沿最後三個軸執行層
正規化；逐元素仿射參數 $\gamma, \beta$ 的形狀為 $(F, D_1, D_2)$，
且 $\epsilon = 10^{-5}$，結果須與
`F.layer_norm(x, x.shape[1:], gamma, beta, eps)` 一致。檢查條件為
`rtol = 2e-4`、`atol = 1e-4`。

## 圖解

![對 (F, D₁, D₂) 做 LayerNorm：每個樣本一組平均與變異數，γ、β 逐元素套用](figure.svg)

每一列是一個完整樣本，共 F·D₁·D₂ 個值。它的統計量用來正規化該列，而 γ、β 的形狀與樣本相同。

## 數學表述

$$
G = F D_1 D_2, \qquad
\mu_b = \frac{1}{G}\sum_{g=0}^{G-1} x_{b,g}, \qquad
\sigma_b^2 = \frac{1}{G}\sum_{g=0}^{G-1}\bigl(x_{b,g} - \mu_b\bigr)^2
$$

$$
y_{b,g} = \frac{x_{b,g} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}\,\gamma_g + \beta_g
$$

| 符號 | 意義 |
|---|---|
| $B$ | 批次大小：每個樣本各有一個正規化群組 |
| $G$ | 群組大小，即各正規化軸長度的乘積 |
| $x_{b,g}$ | 樣本 $b$ 的第 $g$ 個元素（展平 $F, D_1, D_2$）；偏移為 $bG + g$ |
| $\mu_b, \sigma_b^2$ | 樣本 $b$ 的平均值與偏差變異數 |
| $\gamma_g, \beta_g$ | 位置 $g$ 的縮放與平移參數（由整個批次共用） |
| $\epsilon$ | $10^{-5}$ |
| $y_{b,g}$ | 輸出 |

與 [批次正規化](../batch-norm/)不同：後者會跨批次依通道計算統計量；
此處則跨特徵依樣本計算。

## 解題思路

1. **每個樣本使用一個含 1024 個執行緒的區塊。** 每個群組都是由 $G$
   個 float 組成的連續範圍，因此執行緒跨步存取可以完全合併。
2. **第一趟**：以 `double` 加總並在區塊內歸約，得到 $\mu_b$。
3. **第二趟**：以 `double` 加總中心化後的平方值並在區塊內歸約，得到
   $r_b = 1/\sqrt{\sigma_b^2 + \epsilon}$。先中心化可避免
   $\mathbb{E}[x^2] - \mu^2$ 產生災難性消去。
4. **第三趟**：計算
   $y = (x - \mu_b)\,r_b\,\gamma_g + \beta_g$（合併讀取 $\gamma, \beta$，
   並透過 L2 由所有區塊共用）。

## 成本分析

$$
Q \approx 3\cdot 4BG + 2\cdot 4G + 4BG = 16BG + 8G\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta_{\text{mem}}}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取 $x$ 三次（除非可放入 L2）、讀取 $\gamma, \beta$ 一次、寫入一次 |
| $\beta_{\text{mem}}$ | DRAM 頻寬（特別命名以區別平移參數 $\beta$） |

當樣本很少（$B$ 很小）而群組很大時，只會執行 $B$ 個區塊；可將每個群組
拆分到數個區塊（先計算部分和，再使用第二個核心），以運用所有 SM；也可
使用單趟 Welford 更新取代第一、二趟。

## 常見陷阱

- 使用**偏差變異數**（$1/G$）。
- **仿射參數是每個位置各一組**，而非每個通道：$\gamma$ 有 $G$ 個元素。
- **fp64 累加**：群組可能包含數百萬個元素。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [批次正規化](../batch-norm/)、[RMS 正規化](../rms-norm/)、
  LeetGPU [層正規化](../../leetgpu/113-layer-normalization/)。
