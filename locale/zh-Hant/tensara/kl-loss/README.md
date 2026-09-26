---
title: Kullback-Leibler 散度
platform: Tensara
upstream: kl-loss
url: https://tensara.org/problems/kl-loss
difficulty: medium
tags: [loss, elementwise, numerics]
status: solved
---

# Kullback-Leibler 散度

**平台：** Tensara · **難度：** medium · [題目說明](https://tensara.org/problems/kl-loss)

## 問題

計算目標分布 $P$ 與預測分布 $Q$ 之間的逐元素 Kullback–Leibler 散度
貢獻值，兩者都是長度為 $N$ 的 float32 向量。參考實作會在取對數前將
兩個輸入限制為至少 $\epsilon = 10^{-10}$，並將目標值不是正數的項目
設為零。檢查條件非常嚴格：`rtol = atol = 1e-5`。

## 公式

$$
D_{\mathrm{KL}}(P\,\Vert\,Q) = \sum_i p_i \log\frac{p_i}{q_i}
$$

| 符號 | 意義 |
|---|---|
| $P, Q$ | 目標分布與預測分布 |
| $p_i, q_i$ | 兩者的機率（`targets[i]`、`predictions[i]`） |
| $D_{\mathrm{KL}}$ | KL 散度（非負，且僅在 $P = Q$ 時為 0） |

所需輸出是逐元素項目，計算方式須與參考實作完全相同：

$$
\tilde{p}_i = \max(p_i, \epsilon), \quad \tilde{q}_i = \max(q_i, \epsilon), \qquad
\text{out}_i = \begin{cases} \tilde{p}_i\bigl(\ln\tilde{p}_i - \ln\tilde{q}_i\bigr), & p_i > 0 \\ 0, & p_i \le 0 \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $\epsilon$ | 下限值 $10^{-10}$ |
| $\tilde{p}_i, \tilde{q}_i$ | 限制範圍後的機率 |
| $\ln$ | 自然對數（`logf`） |
| $\text{out}_i$ | 逐元素貢獻值；可能為負 |

零值的處理方式來自極限 $\lim_{p\to0^+} p\ln p = 0$。

## 方法

使用網格跨步的逐元素映射：兩次載入、兩次 `logf`、一次減法、一次乘法、
一次選擇及一次儲存。使用兩個對數計算
$\ln\tilde{p} - \ln\tilde{q}$（與參考實作相同），而非計算
$\ln(\tilde{p}/\tilde{q})$，可讓捨入結果與 PyTorch 完全一致。

## 成本分析

$$
Q = 12N\ \text{bytes}, \qquad T_{\min} = \frac{12N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：兩個輸入、一個輸出 |
| $\beta$ | DRAM 頻寬 |

每個元素執行兩次 `logf` 約需 40 條指令，但仍會隱藏在 12 位元組的
記憶體流量之後。

## 注意事項

- **重現下限限制**：$\ln(0) = -\infty$，而
  $0\cdot(-\infty) =$ NaN。
- **條件應使用未限制範圍的目標值**（$p_i > 0$），與參考實作中的
  `torch.where(targets > 0, ...)` 相同。
- **`__logf`**（快速數學）有約 $2^{-21.4}$ 的絕對誤差；乘上較大的
  $p$ 後可能超過 `atol = 1e-5`，因此應使用 `logf`。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 相關內容

- [Huber 損失](../huber-loss/)、[Log Softmax](../log-softmax/)、
  LeetGPU [類別交叉熵](../../leetgpu/025-categorical-cross-entropy-loss/)。
