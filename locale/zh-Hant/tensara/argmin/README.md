---
title: 沿維度取最小值索引
platform: Tensara
upstream: argmin
url: https://tensara.org/problems/argmin
difficulty: easy
tags: [reduction, argmin, strided-reduction]
status: solved
---

# 沿維度取最小值索引

**平台：** Tensara · **難度：** easy · [題目敘述](https://tensara.org/problems/argmin)

## 問題

找出 $n$ 維 float32 張量沿 `dim` 維度的最小值索引（若有相同值，取第一次出現的位置），並在輸出中移除縮減的維度。這是 [Argmax](../argmax/) 的相反版本，使用相同的測試形狀。

## 公式

採用 [Argmax](../argmax/) 的 $(O, R, I)$ 表示法：

$$
\text{out}[oI + i] = \min\Bigl\{\ j^\star\ :\ x[o, j^\star, i] = \min_{0\le j<R} x[o, j, i]\ \Bigr\}, \qquad
(v_1, j_1)\oplus(v_2, j_2) = \begin{cases}(v_2, j_2), & v_2 < v_1 \lor (v_2 = v_1 \land j_2 < j_1)\\(v_1, j_1), & \text{otherwise}\end{cases}
$$

| 符號 | 意義 |
|---|---|
| $O,\ R,\ I$ | 外部大小、縮減長度、內部大小（縮減軸的跨距） |
| $x[o, j, i]$ | 元素 $\text{input}[(oR + j)I + i]$ |
| $\oplus$ | 取最小值索引，並以第一個索引處理相同值 |
| out | int32 索引，共 $O\cdot I$ 個 |

## 方法

與 [Argmax](../argmax/) 完全相同，但反轉比較方向，並將單位值設為 $+\text{FLT\_MAX}$：

- $I = 1$：每個輸出使用一個 warp，各通道以跨步方式合併讀取，並以 shuffle 縮減 $(v, j)$。
- $I > 1$：每個輸出使用一個執行緒，以跨距 $I$ 走訪 $R$，相鄰執行緒間形成合併存取。

## 成本分析

$$
Q = 4ORI + 4OI\ \text{bytes}, \qquad T_{\min} = \frac{4ORI}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| $\beta$ | DRAM 頻寬 |

這是受限於記憶體頻寬的單趟處理。

## 常見陷阱

- **相同值的處理方式**與 argmax 相同。
- **負零**：$-0.0 = +0.0$ 的比較結果為相等，因此和 PyTorch 一樣取第一個索引。

## 驗證

所有測試案例皆已在 [cuemu](../../tools/cuemu/README.md) 上通過，索引完全相同。

## 相關內容

- [Argmax](../argmax/)、[沿維度取最小值](../min-dim/)。
