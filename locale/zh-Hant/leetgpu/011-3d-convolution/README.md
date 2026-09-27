---
title: 三維卷積
platform: LeetGPU
upstream: medium/11_3d_convolution
url: https://leetgpu.com/challenges/3d-convolution
difficulty: medium
tags: [convolution, 3d, shared-memory]
status: solved
---

# 三維卷積

**平台：** LeetGPU · **難度：** medium · [題目說明](https://leetgpu.com/challenges/3d-convolution)

## 問題

以 $K_d \times K_r \times K_c$ 核心對 $D \times R \times C$ float32
體積執行「有效」三維互相關（$1 \le D, R, C \le 256$，
$1 \le K_d, K_r, K_c \le 5$；容許誤差 `1e-5`）。輸出形狀為
$(D-K_d+1) \times (R-K_r+1) \times (C-K_c+1)$。這是影片及體積
（醫學影像）CNN 的基本操作。核心最多只有 $5^3 = 125$ 個 tap，
輸入很容易放入快取階層。

## 公式

$$
Y_{z,r,c} = \sum_{a=0}^{K_d-1}\ \sum_{b=0}^{K_r-1}\ \sum_{e=0}^{K_c-1} X_{z+a,\ r+b,\ c+e}\ w_{a,b,e}
$$

$$
\text{offset}_X(z, r, c) = (zR + r)\,C + c, \qquad \text{offset}_Y(z, r, c) = \bigl(z(R-K_r+1) + r\bigr)(C-K_c+1) + c
$$

| 符號 | 意義 |
|---|---|
| $D,\ R,\ C$ | 輸入深度、列數、欄數 |
| $K_d,\ K_r,\ K_c$ | 核心深度、列數、欄數 |
| $X_{z,r,c}$ | 輸入體素（深度切片 $z$、第 $r$ 列、第 $c$ 欄） |
| $w_{a,b,e}$ | 核心 tap |
| $Y_{z,r,c}$ | 輸出體素 |
| $a,\ b,\ e$ | 沿深度、列、欄的核心偏移量 |
| $\text{offset}$ | 列優先（切片、列、欄）配置中的線性索引 |

## 方法

- **網格。** 使用 $\lceil C_o/32\rceil \times \lceil R_o/8\rceil \times D_o$
  個區塊，每區塊有 $32 \times 8$ 個執行緒，每個執行緒負責一個輸出體素。
  `threadIdx.x` 沿欄方向移動，`blockIdx.z` 是輸出深度切片。
- **核心放入共享記憶體。** 每區塊只複製一次不超過 125 個 tap。
  內部迴圈中，一個 warp 的所有執行緒讀取同一 tap，形成廣播。
- **透過 L1 讀取輸入。** 對固定 tap $(a, b, e)$，一個 warp 讀取
  輸入列中的 32 個連續浮點數，形成合併存取。走訪一列的 $K_c$ 個 tap
  時，warp 視窗每次只滑動一個元素，所以第一次以後幾乎都命中 L1。
  因核心很小，不需要像[二維卷積](../010-2d-convolution/)那樣明確使用
  共享記憶體邊暈平鋪。
- **在屏障後提早離開。** 超出輸出範圍的執行緒只能在核心暫存用的
  `__syncthreads()` *之後*返回。

## 成本分析

$$
W = 2K_dK_rK_c\ D_oR_oC_o, \qquad Q_{\min} = 4\,(DRC + D_oR_oC_o), \qquad I_{\max} = \frac{W}{Q_{\min}} \approx \frac{K_dK_rK_c}{4}
$$

| 符號 | 意義 |
|---|---|
| $D_o,\ R_o,\ C_o$ | 輸出維度 $D-K_d+1$、$R-K_r+1$、$C-K_c+1$ |
| $W$ | FLOP |
| $Q_{\min}$ | 必要 DRAM 流量：輸入讀一次、輸出寫一次 |
| $I_{\max}$ | 最佳情況的算術強度（快取提供所有重用時） |

對 $5^3$ 核心，$I_{\max} \approx 31$ FLOP/byte，因此核心受計算或 L1
限制，而非 DRAM。每次 FMA 會進行一次 L1 載入及一次共享載入。
沿欄方向做暫存器分塊（每執行緒計算多個連續 $c$，並滑動暫存器視窗）
最多可將每次 FMA 的 L1 載入減少 $K_c$ 倍。

## 常見問題

- **配置順序。** 攤平配置是（切片、列、欄）。交換 $D$ 與 $R$ 的意義
  在立方體上仍會通過，但在非立方體積上會失敗。
- **在屏障前返回。** 超出範圍的執行緒仍須協助暫存核心並到達
  `__syncthreads()`，才能離開。
- **Tap 數量。** 靜態共享陣列可容納 125 個 tap，正好是 $5^3$ 上限。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上通過，
包括與體積一樣大的核心及 1 × 1 × 1 核心。

## 相關內容

- [一維](../009-1d-convolution/)與[二維卷積](../010-2d-convolution/)。
- Tensara [三維卷積（方形）](../../tensara/conv-square-3d/)。
