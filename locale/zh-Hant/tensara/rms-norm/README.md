---
title: RMS 正規化
platform: Tensara
upstream: rms-norm
url: https://tensara.org/problems/rms-norm
difficulty: easy
tags: [normalization, reduction, row-per-block]
status: solved
---

# RMS 正規化

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/rms-norm)

## 問題

對 $B\times N$ float32 矩陣的每一列執行 RMS 正規化（無權重），其中 $\epsilon = 10^{-5}$；形狀從 $(1024, 1024)$ 到 $(512, 16384)$。檢查條件為 `rtol = 2e-4`、`atol = 1e-4`。

## 公式

$$
\operatorname{RMS}_b = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x_{bn}^2 + \epsilon}, \qquad
y_{bn} = \frac{x_{bn}}{\operatorname{RMS}_b}
$$

| 符號 | 意義 |
|---|---|
| $B$ | 列數（樣本數） |
| $N$ | 每列的特徵數 |
| $x_{bn}, y_{bn}$ | 輸入與輸出元素 |
| $\operatorname{RMS}_b$ | 第 $b$ 列的均方根，$\epsilon$ 位於平方根內 |
| $\epsilon$ | $10^{-5}$ |

相較於 [Layer Norm](../layer-norm/)，RMSNorm 省略了平均值相減：只重新縮放而不重新置中，因此少一次縮減，且在 Transformer（LLaMA、T5）中效果同樣良好。

## 方法

每列使用一個含 256 個執行緒的區塊。第一階段：每個執行緒對跨步分配的元素累加 $x^2$，再經由區塊縮減（warp shuffle 加上一次共享記憶體跳轉）取得 $\sum x^2$。每個執行緒計算 $r_b = 1/\sqrt{\sum x^2/N + \epsilon}$。第二階段寫入 $y = x\,r_b$，並從 L1/L2 重新讀取該列。

## 成本分析

$$
Q_{\text{DRAM}} \approx 8BN\ \text{bytes}, \qquad W = 3BN\ \text{flops}, \qquad T_{\min} = \frac{8BN}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q_{\text{DRAM}}$ | 一次讀取（重讀會命中快取）和一次寫入 |
| $W$ | 計算 $x^2$ 的一次 FMA，以及縮放用的一次乘法 |
| $\beta$ | DRAM 頻寬 |

當尺寸為 $(2048, 8192)$ 時：共 134 MB，在 2 TB/s 下約需 67 µs。

## 注意事項

- **$\epsilon$ 位於平方根內**（加到平方平均值），並非像 [L2 範數](../l2-norm/) 那樣加到 RMS。
- **除以 $N$**（平均值），不是 $N - 1$。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 相關內容

- [Layer Norm](../layer-norm/)、[L2 範數](../l2-norm/)、LeetGPU [RMS 正規化](../../leetgpu/050-rms-normalization/)、LeetGPU [融合殘差加法 + RMSNorm](../../leetgpu/083-fused-residual-add-rms-norm/)。
