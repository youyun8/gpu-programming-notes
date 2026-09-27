---
title: 高斯模糊
platform: LeetGPU
upstream: medium/28_gaussian_blur
url: https://leetgpu.com/challenges/gaussian-blur
difficulty: medium
tags: [convolution, stencil, shared-memory, zero-padding]
status: solved
---

# 高斯模糊

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/gaussian-blur)

## 題意

使用正規化的 $K_r \times K_c$ 高斯核心，模糊一張
$R \times C$ 的 float32 影像（$1 \le R, C \le 4096$；
$K_r, K_c$ 是 $[3, 21]$ 中的奇數；基準測試為 $512 \times 512$、
核心大小 $7\times7$；容許誤差為 `1e-5`）。這是 **"same"** 卷積：
輸出與輸入大小相同，影像外的像素視為 0（零填補）。參考實作使用
`F.conv2d`，padding 為 $(K_r/2, K_c/2)$。

## 圖解

![零填補的「same」模糊：靠近邊緣的視窗會讀到填補的 0](figure.svg)

輸出與輸入大小相同。對輸出像素 (0, 0) 而言，3 × 3 視窗超出邊界，紅色取樣點落在填補區，貢獻為 0。

## 數學表述

$$
Y_{ij} = \sum_{m=0}^{K_r-1}\sum_{n=0}^{K_c-1} \tilde X_{\,i + m - h_r,\ j + n - h_c}\ w_{mn}, \qquad
\tilde X_{ab} = \begin{cases} X_{ab}, & 0 \le a < R,\ 0 \le b < C \\ 0, & \text{otherwise} \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $R,\ C$ | 影像列數與欄數（也是輸出大小） |
| $K_r,\ K_c$ | 核心高度與寬度（奇數） |
| $h_r,\ h_c$ | 半徑 $\lfloor K_r/2\rfloor$、$\lfloor K_c/2\rfloor$：核心中心的偏移量 |
| $w_{mn}$ | 核心權重，$w_{mn} \ge 0$、$\sum w_{mn} = 1$ |
| $X_{ab}$ | 輸入像素 |
| $\tilde X_{ab}$ | 零填補後的輸入 |
| $Y_{ij}$ | 輸出像素，$0 \le i < R$、$0 \le j < C$ |

真正的高斯核心可**分離**為 $w_{mn} = g_m g_n$。如此可改用兩次
一維處理，每個像素的 tap 數從 $K_rK_c$ 降為 $K_r + K_c$。
但題目傳入任意正規化核心，因此必須計算二維形式。

## 解題思路

這是 [二維卷積](../010-2d-convolution/)核心函式的變體，有兩項改動：

1. **光環偏移。** 暫存視窗從 $(i_0 - h_r,\ j_0 - h_c)$ 開始，
   也就是在 $32\times32$ 輸出分塊前各多出半個核心。
2. **暫存時零填補。** 將
   $(32 + K_r - 1)\times(32 + K_c - 1)$ 視窗複製到共享記憶體時，
   影像外的位置寫入 0。如此一來，內層 FMA 迴圈完全**不需要邊界檢查**。
   每個執行緒都執行相同的指令流，邊緣分塊與內部分塊的成本相同。

每個 32 × 8 區塊中，每個執行緒計算 4 個輸出。核心權重透過
共享記憶體廣播，而同一 warp 的輸入讀取位址連續。

## 成本分析

$$
W = 2K_rK_c\,RC, \qquad Q \approx 4RC\left(\frac{(32+K_r-1)(32+K_c-1)}{32\cdot32} + 1\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP 數 |
| $Q$ | DRAM 位元組數：每個分塊載入含光環的視窗，每個輸出寫入一次 |

基準測試（$512^2$、$7\times7$）：$W \approx 25.7$ MFLOP，
$Q \approx 2.4$ MB。這是微秒等級的核心函式，主要成本來自啟動開銷。
在 $4096^2$、$21\times21$ 下，同一份程式碼會從共享記憶體執行
14.8 GFLOP，重複使用率約為
$1024\cdot441/52^2 \approx 167$。

## 常見陷阱

- **Valid 與 same。** 與[二維卷積](../010-2d-convolution/)不同，
  輸出大小與輸入相同，且分塊起點要減去半個核心。
- **負座標。** 視窗可能從列/欄 $-h$ 開始，因此邊界檢查必須包含
  `r >= 0 && c >= 0`。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
以 `1e-5` 容許誤差通過，包括單像素影像，以及比影像更大的
$21 \times 21$ 核心。

## 延伸閱讀

- [二維卷積](../010-2d-convolution/)、
  [Jacobi 樣板](../069-jacobi-stencil-2d/)。
- Tensara [方框模糊](../../tensara/box-blur/)、
  [邊緣偵測](../../tensara/edge-detect/)。
