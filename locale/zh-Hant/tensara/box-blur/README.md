---
title: 方框模糊
platform: Tensara
upstream: box-blur
url: https://tensara.org/problems/box-blur
difficulty: easy
tags: [stencil, separable, image-processing]
status: solved
---

# 方框模糊

**平台：** Tensara · **難度：** easy · [題目敘述](https://tensara.org/problems/box-blur)

## 題意

對高度 $h$、寬度 $w$ 的單通道 float32 影像，以邊長為奇數的視窗
$K$ 執行方框模糊（測試使用 15、21 或 27，以及
$1920\times1080$ 和 $2048\times2048$ 影像）。在邊界上只會平均實際存在的像素，因此靠近邊緣時除數會變小。檢查誤差為
`rtol = atol = 1e-4`。

## 圖解

![方框模糊：取 K × K 視窗內實際存在的像素平均](figure.svg)

在角落，3 × 3 視窗只覆蓋四個真實像素（紅框），因此除以 4 而不是 9；不做任何填補。

## 數學表述

$$
r = \left\lfloor K/2 \right\rfloor, \qquad
\text{out}[i, j] = \frac{1}{N_{ij}} \sum_{u = i_0}^{i_1} \sum_{v = j_0}^{j_1} x[u, v]
$$

$$
i_0 = \max(i - r, 0),\quad i_1 = \min(i + r, h - 1),\quad
j_0 = \max(j - r, 0),\quad j_1 = \min(j + r, w - 1),\quad
N_{ij} = (i_1 - i_0 + 1)(j_1 - j_0 + 1)
$$

| 符號 | 意義 |
|---|---|
| $x$ | 輸入影像，$h\times w$，列優先排列 |
| $K$ | 視窗邊長（`kernel_size`，奇數） |
| $r$ | 視窗半徑 |
| $i, j$ | 像素的列與欄 |
| $i_0, i_1, j_0, j_1$ | 裁切至影像範圍內的視窗邊界 |
| $N_{ij}$ | 裁切後視窗中的有效像素數 |

由於裁切後的視窗是矩形，總和可分解為：

$$
\text{out}[i, j] = \frac{1}{N_{ij}} \sum_{u=i_0}^{i_1} R[u, j], \qquad R[u, j] = \sum_{v=j_0}^{j_1} x[u, v]
$$

| 符號 | 意義 |
|---|---|
| $R$ | 水平（列）總和，是一張中間結果的 $h\times w$ 影像 |

## 解題思路

1. **列處理** `rowSums`：每個像素使用一個執行緒，在同一列中計算最多 $K$ 個鄰近像素的 $R[u, j]$。warp 中的相鄰執行緒會共用幾乎所有載入，而這些資料由 L1 提供。
2. **欄處理** `colSums`：每個像素使用一個執行緒，沿該欄加總 $R$（最多載入 $K$ 次；因相鄰執行緒具有相鄰的 $j$，可在 warp 間合併存取），再除以 $N_{ij}$。
3. 每次呼叫都以 `cudaMalloc` 配置暫存的 $R$，並在同步後釋放。

每個像素只需 $2K$ 次加法，而非 $K^2$ 次（$K = 27$ 時為 54 次，而非 729 次）。使用累加和的滑動視窗版本可讓每個像素只需 $O(1)$ 次運算，但會讓每一列序列化，降低平行度。

## 成本分析

$$
W_{\text{ops}} = 2K\,hw, \qquad Q \approx 4 \cdot 4\,hw\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{ops}}$ | 兩趟處理中的加法次數 |
| $Q$ | DRAM 位元組數：讀取 $x$、寫入 $R$、讀取 $R$、寫入輸出 |
| $\beta$ | DRAM 頻寬 |

對 $2048^2$ 而言：$Q \approx 67$ MB，在 2 TB/s 下約需 34 µs。使用帶有共享記憶體圖塊（另加 $r$ 列暈邊）的單一核心函式融合兩趟處理，可省下 $R$ 的來回傳輸。

## 常見陷阱

- **除數**：有效像素數，而非 $K^2$（與[二維平均池化](../avg-pool-2d/)不同）。
- **捨入**：參考實作採用不同的加總順序（conv2d），因此結果的最後幾個位元不同；`1e-4` 的誤差容許範圍可涵蓋此差異。
- **暫存配置**會在計時區段中產生少量成本；實際的函式庫會保留工作空間。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 延伸閱讀

- [二維平均池化](../avg-pool-2d/)、[邊緣偵測](../edge-detect/)、
  [二維卷積](../conv-2d/)、LeetGPU [二維卷積](../../leetgpu/010-2d-convolution/)。
