---
title: 二維平均池化
platform: Tensara
upstream: avg-pool-2d
url: https://tensara.org/problems/avg-pool-2d
difficulty: medium
tags: [pooling, stencil, grid-stride]
status: solved
---

# 二維平均池化

**平台：** Tensara · **難度：** medium · [題目敘述](https://tensara.org/problems/avg-pool-2d)

## 問題

對 $H\times W$ 的 float32 矩陣，以 $k\times k$ 視窗、步幅 $S$ 和零填補 $P$ 執行二維平均池化。參考實作為使用預設值的
`torch.nn.functional.avg_pool2d`，也就是
`count_include_pad=True`：填補位置視為零，並且**會計入**除數。檢查誤差為 `rtol = 2e-4`、`atol = 2e-5`。

## 公式

$$
H_{\text{out}} = \left\lfloor \frac{H + 2P - k}{S} \right\rfloor + 1, \qquad
W_{\text{out}} = \left\lfloor \frac{W + 2P - k}{S} \right\rfloor + 1
$$

$$
\text{out}[i, j] = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \tilde{x}\bigl[S i + m - P,\ S j + n - P\bigr], \qquad
\tilde{x}[r, c] = \begin{cases} x[r, c], & 0 \le r < H,\ 0 \le c < W \\ 0, & \text{otherwise} \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $x$ | 輸入矩陣，$H\times W$，列優先排列 |
| $\tilde{x}$ | 將 $x$ 的邊界外延伸為零（即填補） |
| $k$ | 視窗邊長（`kernel_size`） |
| $S$ | 相鄰視窗間的步幅 |
| $P$ | 每一側的零填補量 |
| $H_{\text{out}}, W_{\text{out}}$ | 輸出高度和寬度 |
| $i, j$ | 輸出的列與欄 |
| $m, n$ | 視窗內的偏移量 |
| $k^2$ | 除數；即使在邊界也一律使用完整視窗大小 |

## 方法

1. **每個輸出使用一個執行緒**，以網格跨步迴圈處理
   $H_{\text{out}}W_{\text{out}}$ 個輸出。相鄰執行緒取得連續的
   $j$，因此一個 warp 會讀取由 $k$ 列組成的帶狀區域；當
   $S < k$ 時，各執行緒讀取的欄會重疊：相鄰執行緒會命中相同的快取列，而 L1/L2 會供應大部分的 $k^2$ 次載入。
2. **以邊界檢查處理填補**，不另外複製資料：略過 $[0, H)$ 和
   $[0, W)$ 以外的列與欄（等同於加 0）。
3. 最後**除以 $k^2$**，不受範圍內元素數量影響。

由於大小很小（$k \le$ 數個元素），使用共享記憶體圖塊只能省下快取命中。若 $k$ 很大，可採用 [方框模糊](../box-blur/) 的可分離技巧（先算列總和，再算欄總和）。

## 成本分析

$$
W_{\text{ops}} = k^2 H_{\text{out}} W_{\text{out}}, \qquad
Q \approx 4\,(HW + H_{\text{out}}W_{\text{out}})\ \text{bytes}, \qquad
T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{ops}}$ | 加法次數 |
| $Q$ | DRAM 位元組數，假設重疊視窗由快取供應 |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | 執行時間的頻寬下限 |

算術強度約為每位元組 $k^2/8$ 次運算（$S = 1$ 時），遠低於轉折點，因此核心函式受限於記憶體頻寬。

## 常見陷阱

- **`count_include_pad`**：除以*有效*元素數量是方框模糊所需的行為，不是本題的行為。
- **輸出大小**使用 $H + 2P - k$ 除以 $S$ 的整數除法；應以有號
  `int` 計算，而不是 `size_t`。
- 扁平化輸入偏移量使用 **64 位元索引**。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 相關內容

- [一維平均池化](../avg-pool-1d/)、[三維平均池化](../avg-pool-3d/)、
  [二維最大池化](../max-pool-2d/)、[方框模糊](../box-blur/)。
