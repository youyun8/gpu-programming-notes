---
title: 鉸鏈損失
platform: Tensara
upstream: hinge-loss
url: https://tensara.org/problems/hinge-loss
difficulty: easy
tags: [loss, elementwise]
status: solved
---

# 鉸鏈損失

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/hinge-loss)

## 題意

對 $N$ 個實數預測值與 $\{-1, +1\}$ 中的目標值計算逐元素鉸鏈損失
（1 M … 67 M 個元素）。輸出是尚未取平均的各元素損失。檢查條件為
`rtol = atol = 1e-4`。

## 圖解

![Hinge 損失：預測在正確一側且邊界至少為 1 時，損失為 0](figure.svg)

損失只取決於邊界 x·y：邊界達到 1 起（紅點）損失為 0，低於 1 時線性增加。

## 數學表述

$$
\ell_i = \max\bigl(0,\ 1 - x_i\,y_i\bigr), \qquad \mathcal{L} = \frac{1}{N}\sum_{i=0}^{N-1} \ell_i\ \ (\text{not required here})
$$

| 符號 | 意義 |
|---|---|
| $x_i$ | 預測值（原始分數，不是機率） |
| $y_i$ | 目標標籤，$-1$ 或 $+1$ |
| $x_i y_i$ | 間隔：符號正確時為正值 |
| $\ell_i$ | 每個元素的鉸鏈損失，即輸出 |
| $\mathcal{L}$ | SVM 訓練使用的平均損失 |

當預測位於正確一側且間隔至少為 1 時，損失為 0；否則會線性增加。

## 解題思路

對兩個輸入執行網格跨步映射：每個執行緒載入 $x_i$ 與 $y_i$
（合併存取）、計算 `fmaxf(0, 1 - x*y)`，再儲存 $\ell_i$。

## 成本分析

$$
Q = 12N\ \text{bytes}, \qquad T_{\min} = \frac{12N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：每個元素有兩個輸入與一個輸出 |
| $\beta$ | DRAM 頻寬 |

在 $N = 2^{26}$ 時：共 805 MB，以 2 TB/s 計算約需 0.4 ms。使用
`float4` 載入可減少指令數，但不能減少位元組數。

## 常見陷阱

- **逐元素輸出**：題目說明雖然顯示平均值，但預期輸出是向量 $\ell$。
- `fmaf(-x, y, 1)` 與 `1 - x*y` 的捨入差異最多為 1 ulp。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [Huber 損失](../huber-loss/)、[MSE 損失](../mse-loss/)、
  [Triplet Margin 損失](../triplet-margin/)。
