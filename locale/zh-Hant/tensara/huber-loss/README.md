---
title: Huber 損失
platform: Tensara
upstream: huber-loss
url: https://tensara.org/problems/huber-loss
difficulty: easy
tags: [loss, elementwise]
status: solved
---

# Huber 損失

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/huber-loss)

## 問題

計算長度為 $N$ 的預測值與目標值之間的逐元素 Smooth L1 損失（$\delta = 1$
的 Huber 損失；1 M … 67 M 個元素），結果須與
`F.smooth_l1_loss(p, t, reduction='none', beta=1.0)` 一致。檢查條件為
`rtol = 2e-4`、`atol = 1e-4`。

## 公式

$$
d_i = x_i - y_i, \qquad
z_i = \begin{cases} \dfrac{d_i^2}{2\beta}, & \lvert d_i \rvert < \beta \\[4pt] \lvert d_i \rvert - \dfrac{\beta}{2}, & \text{otherwise} \end{cases}, \qquad \beta = 1
$$

| 符號 | 意義 |
|---|---|
| $x_i, y_i$ | 預測值與目標值 |
| $d_i$ | 殘差 |
| $\beta$ | 二次區間與線性區間的轉換點 |
| $z_i$ | 每個元素的損失，即輸出 |

兩段在 $\lvert d\rvert = \beta$ 時具有相同的值與斜率：

$$
\frac{\beta^2}{2\beta} = \beta - \frac{\beta}{2} = \frac{\beta}{2}, \qquad
\frac{d}{dd}\Bigl(\frac{d^2}{2\beta}\Bigr)\Big|_{d=\beta} = 1 = \frac{d}{dd}\bigl(d - \tfrac{\beta}{2}\bigr)
$$

| 符號 | 意義 |
|---|---|
| $\frac{d}{dd}$ | 對殘差微分 |

因此，誤差較小時損失為二次函數（類似 MSE），誤差較大時則為線性函數
（類似 L1，對離群值較穩健）。

## 方法

對兩個輸入執行網格跨步的逐元素映射：
`a = fabsf(d); out = a < 1 ? 0.5f*d*d : a - 0.5f`。分支會編譯成選擇
指令，因此不會產生分歧。

## 成本分析

$$
Q = 12N\ \text{bytes}, \qquad T_{\min} = \frac{12N}{\beta_{\text{mem}}}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：兩個輸入、一個輸出 |
| $\beta_{\text{mem}}$ | DRAM 頻寬（如此命名是為了避免與損失參數 $\beta$ 混淆） |

在 $N = 2^{26}$ 時：共 805 MB，以 2 TB/s 計算約需 0.4 ms。

## 注意事項

- 二次分支使用**嚴格不等式** $\lvert d\rvert < 1$（兩個分支在 1 時
  結果相同，因此這只影響完全精確性）。
- 輸出是**逐元素結果**，不是平均值。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 相關內容

- [MSE 損失](../mse-loss/)、[Hinge 損失](../hinge-loss/)、
  [KL 損失](../kl-loss/)。
