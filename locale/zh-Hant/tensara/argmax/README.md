---
title: 沿維度取最大值索引
platform: Tensara
upstream: argmax
url: https://tensara.org/problems/argmax
difficulty: easy
tags: [reduction, argmax, strided-reduction]
status: solved
---

# 沿維度取最大值索引

**平台：** Tensara · **難度：** easy · [題目敘述](https://tensara.org/problems/argmax)

## 題意

找出 $n$ 維 float32 張量沿 `dim` 維度的最大值索引（列優先排列；若有相同值，取第一次出現的位置）。輸出為 int32，並移除該維度。測試形狀從 $(16, 128, 256)$ 到
$(64, 128, 128, 128)$，且會沿不同軸縮減。Tensara 上每個 `*-dim` 問題都使用同一套「沿任意軸縮減」機制。

## 圖解

![沿單一維度的 argmax：取最大值所在的索引 j，平手時取第一個](figure.svg)

第 i = 2 欄的值為 4、5、9、6，因此輸出索引 2。合併（值, 索引）數對時，平手取較小的索引，無論歸約順序為何都會得到第一個出現的位置。

## 數學表述

將張量視為三個軸：`dim` 之前的所有維度、要縮減的軸，以及其後的所有維度。

$$
O = \prod_{k<\text{dim}} S_k, \qquad R = S_{\text{dim}}, \qquad I = \prod_{k>\text{dim}} S_k, \qquad
x[o, j, i] = \text{input}[\,(oR + j)\,I + i\,]
$$

$$
\text{out}[oI + i] = \min\Bigl\{\ j^\star \ :\ x[o, j^\star, i] = \max_{0 \le j < R} x[o, j, i]\ \Bigr\}
$$

| 符號 | 意義 |
|---|---|
| $S_k$ | 第 $k$ 個維度的大小 |
| $O$ | 「外部」大小：`dim` 之前各維度的乘積 |
| $R$ | 縮減維度的長度 |
| $I$ | 「內部」大小：`dim` 之後各維度的乘積（縮減軸的記憶體跨距） |
| $x[o, j, i]$ | 外部索引 $o$、縮減索引 $j$、內部索引 $i$ 所指的元素 |
| out | $O\cdot I$ 個索引；最大值相同時取最小索引 |

縮減運算子作用於配對 $(v, j)$：

$$
(v_1, j_1) \oplus (v_2, j_2) = \begin{cases} (v_2, j_2), & v_2 > v_1\ \lor\ (v_2 = v_1 \land j_2 < j_1) \\ (v_1, j_1), & \text{otherwise}\end{cases}
$$

| 符號 | 意義 |
|---|---|
| $(v, j)$ | 候選值及其索引 |
| $\oplus$ | 可結合且可交換的「取最大值索引，並以第一個索引處理相同值」運算 |

## 解題思路

依 $I$ 選擇兩個核心函式之一：

- **$I = 1$（縮減連續的最後一軸）**：每個輸出使用一個 **warp**。
  各通道跨步讀取該列中連續的 $R$ 個浮點數（合併存取），以
  $\oplus$ 累積，再分別對 $v$ 和 $j$ 執行五次 `__shfl_down_sync`。
- **$I > 1$**：每個輸出 $(o, i)$ 使用一個**執行緒**，以跨距
  $I$ 逐一處理 $j$。相鄰執行緒具有相鄰的 $i$，因此每一步中，warp 都會讀取 32 個連續浮點數，形成跨執行緒的合併存取。

縮減邏輯封裝在小型 `Acc` 結構中（identity、make、combine、shuffle、finish）。相同的兩個核心函式只要具現化不同結構，就能解決 argmin，以及沿維度取 max/min/sum/mean/product 的問題。

`shape` 陣列可能以主機指標或裝置指標傳入，因此使用
`cudaMemcpyDefault` 複製，再由統一位址空間決定傳輸方向。

## 成本分析

$$
Q = 4\,ORI + 4\,OI\ \text{bytes}, \qquad T_{\min} = \frac{4ORI}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：完整讀取張量一次，再寫入索引 |
| $\beta$ | DRAM 頻寬 |

最大案例 $(64, 128, 128, 128)$ 為 537 MB，也就是在 2 TB/s 下約需 0.27 ms。當 $I = 1$ 且 $R$ 很小（例如 128）時，每列使用一個 warp 會浪費部分通道。讓每個 warp 處理多列可改善此問題。

## 常見陷阱

- **相同值**：`torch.argmax` 會回傳第一個最大值的索引。組合規則會在值相等時比較索引。
- **單位元素**：沒有資料的通道使用索引 `INT_MAX`。
- **64 位元大小**：$O\cdot R\cdot I$ 可能超過 $2^{31}$ 個元素，因此索引使用 `long long`。

## 驗證

所有測試案例（每種官方形狀／維度的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，索引完全相同。

## 延伸閱讀

- [Argmin](../argmin/)、[沿維度取最大值](../max-dim/)、[沿維度加總](../sum-dim/)、
  [平均值](../mean-dim/)、[乘積](../product-dim/)。LeetGPU [縮減](../../leetgpu/004-reduction/)。
