---
title: 均方誤差損失
platform: Tensara
upstream: mse-loss
url: https://tensara.org/problems/mse-loss
difficulty: easy
tags: [loss, reduction, fp64-accumulation, grid-reduction]
status: solved
---

# 均方誤差損失

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/mse-loss)

## 題意

計算兩個任意形狀 float32 張量之間的均方誤差。尺寸從 $4096^2$、$8192^2$ 到 $512^3$（1.34 億個元素），並回傳一個純量。形狀以陣列 `shape` 傳入，其中包含 `ndim` 個 64 位元大小值。檢查條件為 `rtol = 5e-5`、`atol = 1e-4`。

## 圖解

![任意形狀張量的 MSE：把平方差歸約成一個純量](figure.svg)

形狀只決定元素總數 n。每一對先平方，再由樹狀歸約加總，最後一步除以 n。

## 數學表述

$$
n = \prod_{k=0}^{\text{ndim}-1} S_k, \qquad
\text{MSE} = \frac{1}{n}\sum_{i=0}^{n-1} \bigl(x_i - y_i\bigr)^2
$$

| 符號 | 意義 |
|---|---|
| $S_k$ | 維度 $k$ 的大小 |
| $n$ | 元素總數 |
| $x_i, y_i$ | 攤平後的預測值與目標值 |
| MSE | 純量輸出 |

總和採用兩層分割，做法與 [Frobenius 範數](../frobenius-norm/) 相同：

$$
\text{MSE} = \frac{1}{n}\sum_{b=0}^{G-1} P_b, \qquad P_b = \sum_{i \in \mathcal{K}_b} (x_i - y_i)^2
$$

| 符號 | 意義 |
|---|---|
| $G$ | 第一個核心的區塊數（$\le 1024$） |
| $\mathcal{K}_b$ | 區塊 $b$ 造訪的索引 |
| $P_b$ | 區塊部分和（fp64） |

## 解題思路

1. **在主機上計算元素數量**：`shape` 可能是主機或裝置指標，因此使用 `cudaMemcpyDefault` 複製（由統一虛擬定址選擇方向），再將各維度相乘。
2. **`squaredDiffs`**：使用網格跨步迴圈與每執行緒 float 累加器，在 `double` 中進行區塊縮減，每個區塊產生一個部分和。
3. **`finalize`**：使用一個區塊在 `double` 中加總所有部分和，並將 $\sum/n$ 以 float 寫入。

與對單一 float 使用 `atomicAdd` 不同，此結果具有確定性。

## 成本分析

$$
Q = 8n\ \text{bytes}, \qquad T_{\min} = \frac{8n}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：兩個張量各讀取一次，輸出可忽略不計 |
| $\beta$ | DRAM 頻寬 |

當 $n = 2^{27}$（$512^3$）時：共 1.07 GB，在 2 TB/s 下約需 0.54 ms。

## 常見陷阱

- **精度**：若以 fp32 加總 $1.3\times10^8$ 個平方殘差，會超出 `rtol = 5e-5` 的誤差範圍；fp64 部分和可避免此問題。
- **`shape` 的指標種類**：在主機上解參考裝置指標會當機；`cudaMemcpyDefault` 可處理兩者。
- **純量輸出**：只寫入一個 float。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [Huber 損失](../huber-loss/)、[Frobenius 範數](../frobenius-norm/)、LeetGPU [均方誤差](../../leetgpu/027-mean-squared-error/)。
