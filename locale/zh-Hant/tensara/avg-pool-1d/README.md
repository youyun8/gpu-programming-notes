---
title: 一維平均池化
platform: Tensara
upstream: avg-pool-1d
url: https://tensara.org/problems/avg-pool-1d
difficulty: easy
tags: [pooling, stencil, memory-bound]
status: solved
---

# 一維平均池化

**平台：** Tensara · **難度：** easy · [題目敘述](https://tensara.org/problems/avg-pool-1d)

## 問題

在一個很長的 float32 向量上，使用視窗 $k$、步幅 $S$ 和零填補 $P$ 執行一維平均池化（$H$ 最大為 $6.7\times10^7$；例如 $k = 7$、$S = 4$、$P = 3$）。其行為與使用預設值的 `F.avg_pool1d` 相同，也就是
`count_include_pad=True`：填補位置視為零並計入數量，除數一律為 $k$。

## 公式

$$
H_{\text{out}} = \left\lfloor\frac{H + 2P - k}{S}\right\rfloor + 1, \qquad
y_i = \frac1k\sum_{m=0}^{k-1}\tilde x_{Si + m - P}, \qquad
\tilde x_t = \begin{cases} x_t, & 0 \le t < H\\ 0, & \text{otherwise}\end{cases}
$$

| 符號 | 意義 |
|---|---|
| $H$ | 輸入長度 |
| $k$ | 視窗大小（`kernel_size`） |
| $S$ | 步幅 |
| $P$ | 每一側的填補量 |
| $H_{\text{out}}$ | 輸出長度 |
| $\tilde x$ | 以零填補的輸入 |
| $y_i$ | 輸出：使用固定除數 $k$ 的視窗平均值 |

## 方法

每個輸出 $i$ 使用一個執行緒（以網格跨步走訪 $H_{\text{out}}$）。視窗從 $t_0 = Si - P$ 開始。執行緒加總範圍內的元素並略過範圍外的位置（等同於加零），最後乘以 $1/k$。

相鄰執行緒的起點相差 $S$ 個元素，因此一個 warp 的視窗載入範圍涵蓋約 $32S + k$ 個連續元素：大多可合併存取；當 $S < k$ 時，重疊視窗由 L1 快取提供資料。

## 成本分析

$$
Q \approx 4H + 4H_{\text{out}}\ \text{bytes}, \qquad W = kH_{\text{out}}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（因快取作用，每個輸入約讀取一次；每個輸出寫入一次） |
| $W$ | 加法次數 |

最大案例（$H = 6.7\times10^7$、$S = 3$）約為 360 MB，也就是約 0.18 ms。此核心函式受限於記憶體頻寬。

## 常見陷阱

- **除數**：一律為 $k$（`count_include_pad=True`），即使視窗超出填補範圍也一樣。
- **輸出長度公式**使用向下取整除法。
- 對很大的 $H$ 使用 **64 位元索引**。

## 驗證

所有測試案例（官方形狀的縮小及奇數大小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過。

## 相關內容

- [二維平均池化](../avg-pool-2d/)、[三維平均池化](../avg-pool-3d/)、[一維最大池化](../max-pool-1d/)。
