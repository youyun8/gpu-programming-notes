---
title: 線性遞迴
platform: LeetGPU
upstream: medium/82_linear_recurrence
url: https://leetgpu.com/challenges/linear-recurrence
difficulty: medium
tags: [scan, linear-recurrence, ssm, affine-maps]
status: solved
---

# 線性遞迴

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/linear-recurrence)

## 題意

以 $h_0 = x_0$ 為起點，對 $B$ 個彼此獨立、長度為 $L$ 的序列計算一階線性
遞迴 $h_t = a_t h_{t-1} + x_t$（$B \le 256$、$L \le 65\,536$；基準測試
$B = 64$、$L = 16\,384$；容許誤差 `1e-5`）。這是**狀態空間模型**
（S4、Mamba、H3）與線性 RNN 的計算核心。它看似本質上必須循序執行，
其實可視為對仿射映射進行**掃描**。

## 圖解

![把線性遞迴 hₜ = aₜ hₜ₋₁ + xₜ 寫成仿射映射的掃描](figure.svg)

每個狀態等於前一個狀態乘上 aₜ 再加上新輸入 xₜ。把一步寫成數對 (aₜ, xₜ) 後，合成運算具有結合律，於是這條序列鏈可以用平行掃描計算。

## 數學表述

$$
h_0 = x_0, \qquad h_t = a_t\,h_{t-1} + x_t\quad (1 \le t < L)
$$

每一步都是仿射映射 $f_t(h) = a_t h + x_t$，可寫成配對 $(a_t, x_t)$
（而 $f_0 = (0, x_0)$，因為位置 0 沒有前一個狀態）。因此
$h_t = (f_t\circ\cdots\circ f_0)(0)$，而仿射映射的合成為

$$
(A_1, X_1)\ \text{then}\ (A_2, X_2) = \bigl(A_1A_2,\ \ A_2X_1 + X_2\bigr)
$$

| 符號 | 意義 |
|---|---|
| $B,\ L$ | 批次大小與序列長度 |
| $a_t$ | 步驟 $t$ 的衰減／轉移係數 |
| $x_t$ | 步驟 $t$ 的輸入 |
| $h_t$ | 步驟 $t$ 的狀態（輸出） |
| $f_t$ | 步驟 $t$ 套用的仿射映射 |
| $(A, X)$ | 仿射映射 $h \mapsto Ah + X$（一段步驟的合成） |

合成具有**結合律**（它等同於矩陣
$\begin{bmatrix}A & X\\0 & 1\end{bmatrix}$ 的乘法），單位元素為 $(1, 0)$。
對這些配對執行包含式掃描，再套用於 $h = 0$，即可在 $O(\log L)$ 深度內
得到每個 $h_t$。

## 解題思路

**每個序列使用一個含 1024 個執行緒的區塊**（網格 = $B$）：

1. **局部折疊。** 執行緒 $\tau$ 負責一個包含
   $\lceil L/1024\rceil$ 個連續步驟的區塊（基準測試為 16 個），並以
   float64 將其合成為一個映射 $(A_\tau, X_\tau)$。
2. 對 1024 個映射執行**區塊掃描**：先以 warp `__shfl_up_sync` 和
   `compose(prev, cur)` 掃描（順序很重要：較早的映射在左側），再由 warp 0
   掃描 32 個 warp 的總值。
3. **重播。** 執行緒 $\tau$ 的傳入狀態 $h_{\text{in}}$，是排他前綴中
   $X$ 的部分（也就是將所有較早區塊套用於 0）。它會再次循序走訪自己的
   區塊，並寫入 $h_t$。

這與 Mamba 的選擇性掃描得以在 GPU 上平行執行所用的「分塊掃描」概念相同。

## 成本分析

$$
W \approx 3BL\ (\text{fold}) + 2BL\ (\text{replay}) + O(B\cdot 1024\log 1024), \qquad Q = 12BL\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP 數：局部折疊與重播都是線性的；區塊掃描對每個區塊只有很小的常數成本 |
| $Q$ | DRAM 位元組數：每輪各讀取一次 $a$ 與 $x$（合計兩次，第二次來自 L2），並寫入 $h$ |

基準測試為 12.6 MB，也就是僅數微秒的流量。執行時間主要受限於只有 64 個
區塊的網格（在大型 GPU 上少於 SM 數量），以及循序的 16 步區塊迴圈。
若每個序列使用數個區塊，並讓它們透過解耦回看互相銜接，可在 $B$ 較小時
提高平行度。

## 常見陷阱

- **不可交換的合成。** 使用 `compose(prev, incl)`，較早的映射放在前面。
  反轉順序會悄悄讓長鏈的結果出錯。
- **$h_0 = x_0$。** 必須忽略位置 0 的係數 $a_0$（視為 0）。
- **精度。** 許多 $a_t$ 的乘積在長區塊中可能發生 float32 下溢或上溢；
  float64 掃描能提高穩健性。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-5`
通過，包括 $L = 1$、$L < 1024$（許多空區塊）及 $a_t$ 接近 1。

## 延伸閱讀

- [SSM 選擇性掃描](../094-ssm-selective-scan/)、[GAE 反向掃描](../110-gae-reverse-scan/)、
  [分段前綴和](../070-segmented-prefix-sum/)、[線性注意力](../056-linear-attention/)。
