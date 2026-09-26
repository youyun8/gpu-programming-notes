---
title: 三維方形卷積
platform: Tensara
upstream: conv-square-3d
url: https://tensara.org/problems/conv-square-3d
difficulty: hard
tags: [convolution, 3d, shared-memory]
status: solved
---

# 三維方形卷積

**平台：** Tensara · **難度：** hard · [題目敘述](https://tensara.org/problems/conv-square-3d)

## 問題

對 $n\times n\times n$ 的 float32 體積資料，以
$K\times K\times K$ 核心（$K$ 為奇數，$3 \le K \le 11$）和
$K/2$ 零填補執行「same」三維互相關。體積大小從 $32^3$ 到
$512^3$。檢查誤差為 `rtol = 1e-3`、`atol = 1e-2`。

## 公式

$$
C[z, y, x] = \sum_{a=0}^{K-1}\sum_{b=0}^{K-1}\sum_{c=0}^{K-1}
\tilde{A}\bigl[z + a - p,\ y + b - p,\ x + c - p\bigr]\; B[a, b, c], \qquad p = \frac{K-1}{2}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入體積資料，$n^3$，列優先排列（$x$ 為連續維度）；$\tilde{A}$ 在邊界外為零 |
| $B$ | 立方核心，共 $K^3$ 個取樣點 |
| $p$ | 每一面的填補量 |
| $C$ | 輸出體積資料，$n^3$ |
| $z, y, x$ | 輸出座標（深度、列、欄） |
| $a, b, c$ | 核心偏移量 |

## 方法

1. 使用由 $32\times8$ 區塊組成的**網格**
   $(n/32,\ n/8,\ n)$：`blockIdx.z` 表示輸出平面，`threadIdx.x` 表示連續的 $x$ 軸。
2. **將核心放入共享記憶體**：最多 $11^3 = 1331$ 個取樣點（5.3 KB）。warp 的所有通道會同時讀取同一個取樣點，形成廣播。
3. **直接從全域記憶體讀取輸入**：相鄰通道讀取相鄰的 $x$，因此每列取樣點都是一次合併的 128 位元組存取；warp 所存取的
   $K^2$ 列會透過 L1/L2 供該區塊的其他 warp 及下一平面的區塊重複使用。
4. 使用 `continue` 略過超出範圍的平面和列，因此邊界不會產生額外成本。

## 成本分析

$$
W = 2n^3K^3, \qquad Q_{\min} = 8n^3\ \text{bytes}, \qquad I = \frac{W}{Q_{\min}} = \frac{K^3}{4}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數 |
| $Q_{\min}$ | 必要的 DRAM 位元組數（讀取 $A$ 一次、寫入 $C$ 一次） |
| $I$ | 最佳情況的算術強度，每位元組的 flops 數 |

當 $512^3$ 且 $K = 9$ 時：$W = 196$ GFLOP，$I = 182$ flop/B，因此受限於運算效能。由於輸入並未暫存在共享記憶體，每次 FMA 還需要一次 L1 載入。下一步可使用帶有輸入暈邊的共享記憶體圖塊（每個平面
$(8 + K - 1)\times(32 + K - 1)$），並沿 $z$ 軸重複使用暫存器。

## 常見陷阱

- **`size` 是邊長**，不是元素數量。
- **誤差容許範圍較寬**（`atol = 1e-2`），因為最多有 1331 個乘積，且加總順序與 cuDNN 不同。
- 網格的 $z$ 維度上限為 65535；對 $n \le 512$ 足夠。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 相關內容

- [二維卷積](../conv-2d/)、[三維平均池化](../avg-pool-3d/)、
  LeetGPU [三維卷積](../../leetgpu/011-3d-convolution/)。
