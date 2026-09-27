---
title: 二維卷積
platform: LeetGPU
upstream: medium/10_2d_convolution
url: https://leetgpu.com/challenges/2d-convolution
difficulty: medium
tags: [convolution, shared-memory, tiling, dynamic-shared-memory]
status: solved
---

# 二維卷積

**平台：** LeetGPU · **難度：** medium · [題目說明](https://leetgpu.com/challenges/2d-convolution)

## 題意

以 $K_r \times K_c$ 核心對 $R \times C$ float32 影像執行「有效」
二維互相關（$1 \le R, C \le 3072$，$1 \le K_r, K_c \le 31$；
基準為 $3072 \times 3072$ 影像搭配 $15\times15$ 核心）。
輸出為 $(R - K_r + 1) \times (C - K_c + 1)$，容許誤差 `1e-5`。
這是[一維卷積](../009-1d-convolution/)邊暈平鋪概念的二維版本，
也是模糊、邊緣偵測器與 CNN 層的基本操作。

## 圖解

![有效二維卷積：每個輸出像素對應輸入中的一個 K × K 視窗](figure.svg)

紅框是產生右側標示輸出像素的 3 × 3 視窗。視窗每移動一格，九個輸入中有六個會被重用，因此區塊會把分塊連同 halo 一起載入共享記憶體。

## 數學表述

$$
Y_{ij} = \sum_{m=0}^{K_r-1}\sum_{n=0}^{K_c-1} X_{i+m,\ j+n}\ w_{mn},
\qquad 0 \le i < R - K_r + 1,\ \ 0 \le j < C - K_c + 1
$$

| 符號 | 意義 |
|---|---|
| $R,\ C$ | 輸入列數與欄數 |
| $K_r,\ K_c$ | 核心列數與欄數 |
| $X_{ab}$ | 第 $a$ 列、第 $b$ 欄的輸入像素（列優先，偏移量 $aC + b$） |
| $w_{mn}$ | 核心第 $m$ 列、第 $n$ 欄的權重 |
| $Y_{ij}$ | 輸出像素；輸出有 $C - K_c + 1$ 欄 |
| $i,\ j$ | 輸出列與欄 |
| $m,\ n$ | 核心列與欄偏移量 |

### 分塊與邊暈

一個區塊計算左上角為 $(i_0, j_0)$ 的 $32 \times 32$ 輸出分塊，
需要以下輸入視窗：

$$
X[\,i_0 : i_0 + 32 + K_r - 1,\ \ j_0 : j_0 + 32 + K_c - 1\,]
$$

| 符號 | 意義 |
|---|---|
| $(i_0, j_0)$ | 分塊原點：$(32\,\texttt{blockIdx.y},\ 32\,\texttt{blockIdx.x})$ |
| $32 + K - 1$ | 分塊加邊暈；$K = 31$ 時最大為 $62 \times 62$ |

## 解題思路

- **區塊形狀。** 以 32 × 8 個執行緒處理 32 × 32 分塊：
  每個執行緒計算同一欄中第 $t_y, t_y+8, t_y+16, t_y+24$ 列的 4 個輸出。
- **暫存。** 將核心（$K_rK_c$ 個浮點數）與視窗複製至動態共享記憶體：
  $(K_rK_c + (32+K_r-1)(32+K_c-1))\cdot 4$ 位元組，最大時為 19 KB。
  接著執行 `__syncthreads()`。
- **計算。** 走訪 $(m, n)$。權重以廣播方式讀取並用於 4 次 FMA。
  輸入存取 `s_input[(ty + 8r + m)·win_cols + tx + n]` 在 warp 的
  32 個執行緒間（變動 `tx`）是連續的，因此沒有 bank 衝突。
- **儲存**前依輸出形狀檢查邊界。

每個執行緒以暫存器保留 4 個累加器，所以 `s_kernel[m][n]`
的載入成本可由 4 個輸出分攤。

## 成本分析

$$
W = 2K_rK_c\,(R-K_r+1)(C-K_c+1), \qquad
\text{reuse} = \frac{32\cdot 32\cdot K_rK_c}{(32+K_r-1)(32+K_c-1)}
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP（每個輸出的每個 tap 各一次乘加） |
| 重用 | 每個暫存輸入值在共享記憶體中的平均使用次數 |

基準（$3072^2$ 影像、$15\times15$ 核心）的 $W \approx 4.1$ GFLOP。
重用因子為 $1024 \cdot 225 / 46^2 \approx 109$，因此 DRAM 流量
（輸入約 38 MB、輸出 36 MB）不是瓶頸。內部迴圈受共享載入與 FMA 限制。
下一步可沿列方向做暫存器平鋪，將滑動輸入視窗保留在暫存器中。

## 常見陷阱

- **視窗與輸出的邊界不同。** 視窗載入檢查*輸入*形狀；
  儲存檢查*輸出*形狀。
- **每個測試的核心大小不同。** 視窗間距 `win_cols` 是執行期值，
  因此共享陣列為動態配置並以手動索引。
- **大型核心與占用率。** 每區塊 19 KB 會限制每個 SM 同時駐留的區塊數，
  在此規模下仍沒有問題。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上通過，
包括 1 × 1 核心、與輸入一樣大的核心及非方形形狀。

## 延伸閱讀

- [一維卷積](../009-1d-convolution/)、[三維卷積](../011-3d-convolution/)、
  [高斯模糊](../028-gaussian-blur/)、[Jacobi 樣板](../069-jacobi-stencil-2d/)。
- Tensara [二維卷積](../../tensara/conv-2d/)、[方框模糊](../../tensara/box-blur/)。
