---
title: Softmax
platform: LeetGPU
upstream: medium/5_softmax
url: https://leetgpu.com/challenges/softmax
difficulty: medium
tags: [softmax, reduction, online-softmax, numerical-stability]
status: solved
---

# Softmax

**平台：** LeetGPU · **難度：** medium · [題目說明](https://leetgpu.com/challenges/softmax)

## 問題

計算長度為 $N$ 的 float32 向量之 softmax（$1 \le N \le 500\,000$，
基準 $N = 500\,000$），並使用減去最大值的技巧維持數值穩定性。
容許誤差為 `1e-5`。Softmax 將任意分數轉為機率分布，
是注意力與所有分類輸出層的核心。

## 公式

$$
\sigma(x)_i = \frac{e^{x_i - m}}{\displaystyle\sum_{j=0}^{N-1} e^{x_j - m}}, \qquad m = \max_{0 \le j < N} x_j
$$

| 符號 | 意義 |
|---|---|
| $N$ | 向量長度 |
| $x_i$ | 第 $i$ 個輸入分數（float32） |
| $m$ | 輸入最大值；減去它不會改變數學結果 |
| $\sigma(x)_i$ | 第 $i$ 個輸出機率；$\sum_i \sigma(x)_i = 1$ |

**為何減去 $m$？** 當引數大於約 88.7 時，`expf` 會溢位成
$+\infty$。平移後每個指數都 $\le 0$，所以每一項都在 $(0, 1]$，
分母位於 $[1, N]$。不可能發生溢位，且至少有一項等於 1。

### 線上（單遍）最大值與總和

樸素演算法需要三遍：最大值、指數總和、正規化。前兩遍可攜帶
$(m, s)$ 配對合併成一遍，其中 $m$ 是目前最大值，
$s = \sum e^{x_j - m}$ 是已看過元素的總和。兩個部分配對以下式合併：

$$
(m_1, s_1) \oplus (m_2, s_2) = \bigl(m,\ s_1 e^{m_1 - m} + s_2 e^{m_2 - m}\bigr), \qquad m = \max(m_1, m_2)
$$

| 符號 | 意義 |
|---|---|
| $(m_k, s_k)$ | 元素子集 $k$ 的部分結果：其最大值與 $e^{x - m_k}$ 的總和 |
| $\oplus$ | 合併運算子；具結合律與交換律，單位元素為 $(-\infty, 0)$ |
| $e^{m_k - m}$ | 將部分和改以新最大值表示的縮放因子 |

單一元素 $x$ 的配對是 $(x, 1)$。由於 $\oplus$ 具結合律，
可像[歸約](../004-reduction/)中的加總一樣，用任何歸約樹計算。

## 方法

使用三個核心：

1. **`blockMaxSum`。** 最多 512 個區塊，各自將網格步進切片歸併成
   一個配對：先由執行緒局部歸併，再用 warp shuffle，最後經共享記憶體
   傳遞。它將 $(m_b, s_b)$ 寫入裝置陣列。
2. **`globalMaxSum`。** 一個區塊將 $B$ 個配對合併成全域 $(M, S)$。
3. **`normalize`。** 網格步進的逐元素階段寫入
   $e^{x_i - M} \cdot (1/S)$。每個執行緒只計算一次倒數，
   因此每個元素只需一次乘法而不是除法。

單位元素以 $(-\text{FLT\_MAX}, 0)$ 表示，而非 $-\infty$。
當兩個輸入都是單位元素時，`combine` 會提早返回，避免計算
$e^{(-\infty) - (-\infty)} = e^{\text{NaN}}$。

## 成本分析

$$
Q = \underbrace{4N}_{\text{pass 1}} + \underbrace{4N + 4N}_{\text{pass 3}} = 12N \ \text{bytes}, \qquad
Q_{\text{naive}} = 16N \ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取 $x$（第一遍），再讀取 $x$ 並寫入 $\sigma$（第三遍） |
| $Q_{\text{naive}}$ | 三遍版本：為最大值讀取 $x$、為總和讀取 $x$、再讀取並於正規化時寫入 |

即使有 $2N$ 次指數運算，核心仍受記憶體限制（約每位元組 0.2 次
`expf`）。當 $N = 5\times10^5$，整個向量（2 MB）可放入 L2，
第二次讀取大多命中 L2。此時啟動延遲居主導，三個小核心約需 10–20 µs。

## 常見問題

- **省略最大值平移。** 輸入約為 100 時會得到
  $\infty/\infty = \text{NaN}$。
- **盲目使用 `__expf`。** 快速內建函式約有 2 ulp 誤差，
  對很大的負引數精度不佳；`expf` 可穩定維持在 `1e-5` 內。
- **與單位元素合併。** `combine` 必須處理空執行緒產生的 $s = 0$
  （當 $N <$ 執行緒數量），且不能產生 NaN。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上通過，
包括 $N = 1$（輸出恰為 1）及絕對值很大的輸入。

## 相關內容

- [Softmax 注意力](../006-softmax-attention/)：對每個查詢列套用相同的線上合併。
- Tensara [Softmax](../../tensara/softmax/)、[Log-Softmax](../../tensara/log-softmax/)。
- [教學 03－平行歸約](../../tutorials/03-parallel-reduction.md)。
