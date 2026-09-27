---
title: 邊緣偵測
platform: Tensara
upstream: edge-detect
url: https://tensara.org/problems/edge-detect
difficulty: easy
tags: [stencil, reduction, atomics, image-processing]
status: solved
---

# 邊緣偵測

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/edge-detect)

## 題意

對 $h\times w$ 的 float32 影像（$1024\times768$ … $4096\times4096$）
執行梯度幅度邊緣偵測：計算 $x$ 與 $y$ 方向的中央差分及其幅度，將邊界
像素設為 0，再重新縮放整張影像，使其最大值變成 255。檢查條件為
`rtol = atol = 1e-3`。

## 圖解

![邊緣偵測：中央差分、梯度大小，再縮放使最大值為 255](figure.svg)

四個藍色鄰居提供標示像素的水平與垂直差分。先求出全域最大值，再把每個梯度大小縮放到 0 … 255。

## 數學表述

$$
G_x[i, j] = \frac{I[i, j+1] - I[i, j-1]}{2}, \qquad
G_y[i, j] = \frac{I[i+1, j] - I[i-1, j]}{2}, \qquad
M[i, j] = \sqrt{G_x[i, j]^2 + G_y[i, j]^2}
$$

以上適用於內部像素 $1 \le i \le h-2$、$1 \le j \le w-2$，1 像素寬的
邊界則令 $M = 0$。接著：

$$
\text{out}[i, j] = \begin{cases} 255\,\dfrac{M[i, j]}{M_{\max}}, & M_{\max} > 0 \\ M[i, j], & \text{otherwise} \end{cases}, \qquad
M_{\max} = \max_{i, j} M[i, j]
$$

| 符號 | 意義 |
|---|---|
| $I$ | 輸入影像，$h\times w$，以列為主 |
| $G_x, G_y$ | 水平與垂直中央差分 |
| $M$ | 梯度幅度，$\ge 0$ |
| $M_{\max}$ | $M$ 的全域最大值 |
| out | 縮放至 $[0, 255]$ 的輸出 |

全域最大值是以原始浮點位元上的整數 `atomicMax` 求得。這是有效的，因為
對非負 IEEE-754 數值而言：

$$
0 \le a < b \iff \operatorname{bits}(a) < \operatorname{bits}(b)
$$

| 符號 | 意義 |
|---|---|
| $\operatorname{bits}(a)$ | 將 float $a$ 當作無號整數讀取時的 32 位元模式（`__float_as_uint`） |

## 解題思路

1. **`resetMax`** 將裝置端全域變數 `g_max_bits` 設為 0（只啟動一個
   執行緒，因此每次呼叫都會重新開始）。
2. **`magnitude`**：每個像素由一個執行緒處理（網格跨步）。四個鄰居距離
   該像素分別為 $\pm1$ 與 $\pm w$；一個 warp 會存取三段連續的列，全部
   由 L1 提供。每個執行緒追蹤自己的最大值，warp 用
   `__shfl_xor_sync` 歸約，最後每個 warp 由一條 lane 對位元執行
   `atomicMax`。
3. **`normalize`** 讀取一次 `g_max_bits`，並原地重新縮放（影像為常數時
   跳過）。

## 成本分析

$$
Q \approx 4hw\ (\text{read } I) + 4hw\ (\text{write } M) + 8hw\ (\text{rescale}) = 16hw\ \text{bytes}, \qquad
\#\text{atomics} = \frac{\#\text{threads}}{32}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| #atomics | 每個 warp 一次（此處最多 32 K 次） |

在 $4096^2$ 時：共 268 MB，以 2 TB/s 計算約需 0.13 ms。第二趟占一半
流量；除非事先知道最大值，否則無法避免（例如第二趟重新計算 $M$，不儲存
它，雖可省下 $M$ 的往返，卻必須重新讀取 $I$，資料量相同）。

## 常見陷阱

- **邊界像素為 0**，且必須納入最大值計算。
- **對浮點位元執行 `atomicMax`** 只因所有值皆為 $\ge 0$ 才能成立；
  負浮點數的排序相反。
- **常數影像**：若 $M_{\max} = 0$，請勿執行除法。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [方框模糊](../box-blur/)、[灰階轉換](../grayscale/)、[二維卷積](../conv-2d/)、
  LeetGPU [二維 Jacobi 模板運算](../../leetgpu/069-jacobi-stencil-2d/)。
