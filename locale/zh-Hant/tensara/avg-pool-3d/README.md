---
title: 三維平均池化
platform: Tensara
upstream: avg-pool-3d
url: https://tensara.org/problems/avg-pool-3d
difficulty: hard
tags: [pooling, stencil, grid-stride]
status: solved
---

# 三維平均池化

**平台：** Tensara · **難度：** hard · [題目敘述](https://tensara.org/problems/avg-pool-3d)

## 題意

對 $H\times W\times D$ 的 float32 張量，以
$k\times k\times k$ 視窗、步幅 $S$ 和零填補 $P$ 執行三維平均池化，並與 `torch.nn.functional.avg_pool3d` 的行為一致（`count_include_pad=True`，因此除數一律為 $k^3$）。檢查誤差為 `rtol = 2e-4`、`atol = 1e-5`。

## 圖解

![三維平均池化：以步幅 S 移動的 k³ 方塊，除數固定為 k³](figure.svg)

在三個連續深度上取同一個 3 × 3 區域，這 27 個值的平均就是一個輸出體素。

## 數學表述

$$
X_{\text{out}} = \left\lfloor \frac{X + 2P - k}{S} \right\rfloor + 1 \quad \text{for } X \in \{H, W, D\}
$$

$$
\text{out}[a, b, c] = \frac{1}{k^3} \sum_{m=0}^{k-1}\sum_{n=0}^{k-1}\sum_{o=0}^{k-1}
\tilde{x}\bigl[S a + m - P,\ S b + n - P,\ S c + o - P\bigr]
$$

| 符號 | 意義 |
|---|---|
| $x$ | 輸入張量 $H\times W\times D$，列優先排列（$D$ 為連續維度） |
| $\tilde{x}$ | 將 $x$ 的邊界外延伸為零 |
| $k, S, P$ | 視窗邊長、步幅、填補量 |
| $X_{\text{out}}$ | 輸入範圍為 $X$ 的軸所對應的輸出範圍 |
| $a, b, c$ | 輸出在 $H, W, D$ 軸上的索引 |
| $m, n, o$ | 視窗內的偏移量 |
| $k^3$ | 除數，一律使用完整視窗體積 |

輸出索引由扁平索引 $t$ 解碼如下：

$$
c = t \bmod D_{\text{out}}, \qquad b = \left\lfloor t / D_{\text{out}} \right\rfloor \bmod W_{\text{out}}, \qquad
a = \left\lfloor t / (D_{\text{out}} W_{\text{out}}) \right\rfloor
$$

| 符號 | 意義 |
|---|---|
| $t$ | 由一個執行緒處理的扁平輸出索引 |

## 解題思路

每個輸出使用一個執行緒（網格跨步），透過三層巢狀視窗迴圈，提早略過超出邊界的平面、列和欄。相鄰執行緒具有連續的 $c$，也就是連續軸，因此最內層迴圈的讀取能在 warp 間形成合併存取，重疊視窗則會命中快取。最後將總和除以 $k^3$。

## 成本分析

$$
W_{\text{ops}} = k^3\,H_{\text{out}} W_{\text{out}} D_{\text{out}}, \qquad
Q \approx 4\,(HWD + H_{\text{out}}W_{\text{out}}D_{\text{out}})\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{ops}}$ | 加法次數 |
| $Q$ | 必要的 DRAM 流量（若快取能保留重疊資料，每個輸入只讀取一次） |

當 $S < k$ 時，視窗會沿三個軸重疊，而且一個執行緒區塊的工作集（由 $k$ 個平面、每個平面 $k$ 列組成）比二維情況更大。若 L2 發生快取顛簸，下一步可改用可分離版本（三趟一維處理，每個輸出執行 $3k$ 次運算，而非 $k^3$ 次）。

## 常見陷阱

- 即使視窗伸出張量外，**除數仍為 $k^3$**。
- **規格重複使用 $k$**（視窗大小和第三個輸出索引）；程式碼應使用不同名稱。
- 索引解碼順序：$D$ 是最內層。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 延伸閱讀

- [一維平均池化](../avg-pool-1d/)、[二維平均池化](../avg-pool-2d/)、
  [三維最大池化](../max-pool-3d/)、[三維方形卷積](../conv-square-3d/)。
