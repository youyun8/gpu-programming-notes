---
title: 灰階轉換
platform: Tensara
upstream: grayscale
url: https://tensara.org/problems/grayscale
difficulty: easy
tags: [elementwise, image-processing, strided-access]
status: solved
---

# 灰階轉換

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/grayscale)

## 題意

使用 ITU-R BT.601 亮度權重，將高度 $h$、寬度 $w$（$512^2$ …
$3840\times2160$）的交錯 RGB float32 影像（HWC 配置，值位於
$[0, 255]$）轉為灰階。檢查條件為 `rtol = atol = 1e-5`。

## 圖解

![HWC 影像的灰階轉換：Y = 0.299 R + 0.587 G + 0.114 B](figure.svg)

純紅、純綠與純藍像素清楚呈現權重的差異：綠色對感知亮度的貢獻最大。

## 數學表述

$$
Y[i, j] = 0.299\,R[i, j] + 0.587\,G[i, j] + 0.114\,B[i, j]
$$

$$
R[i, j] = x[\,3(iw + j)\,], \quad G[i, j] = x[\,3(iw + j) + 1\,], \quad B[i, j] = x[\,3(iw + j) + 2\,]
$$

| 符號 | 意義 |
|---|---|
| $x$ | 輸入緩衝區，包含 $h\cdot w\cdot 3$ 個 float，通道交錯排列 |
| $R, G, B$ | 像素 $(i, j)$ 的紅、綠、藍通道 |
| $Y$ | 輸出亮度影像，$h\times w$ |
| $0.299, 0.587, 0.114$ | BT.601 權重（總和為 1）；綠色對感知亮度影響最大 |

## 解題思路

每個像素由一個執行緒處理（網格跨步）。執行緒 $t$ 讀取位於
$3t, 3t+1, 3t+2$ 的三個 float。對每條載入指令而言，warp 的位址跨距為
12 位元組，因此一條指令會存取 3 條快取線而非 1 條；但三條指令合計
恰好讀取該 warp 之 32 個像素所占的 384 個連續位元組。第一條指令會將
資料載入 L1，因此另外兩條都會命中，DRAM 流量仍維持最低值。輸出儲存
則完全合併。

`channels` 引數用作像素跨距，因此核心也適用於 RGBA（4 通道）配置。

## 成本分析

$$
Q = 12hw + 4hw = 16hw\ \text{bytes}, \qquad W = 5hw\ \text{flops}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：每個像素包含三個輸入 float 與一個輸出 float |
| $W$ | 每個像素執行兩次 FMA 與一次乘法 |
| $\beta$ | DRAM 頻寬 |

在 $3840\times2160$ 時：$Q = 133$ MB，以 2 TB/s 計算約需 66 µs。
完全向量化的版本可讓每個執行緒載入 3 個 `float4`（4 個像素）；這會減少
指令數，但不會減少位元組數。

## 常見陷阱

- **捨入順序**：參考實作會以 fp32 分別執行乘法與加法來計算
  $0.299R + 0.587G + 0.114B$，而 `fmaf` 縮約只會捨入一次。當值最大為
  255 時，兩者的絕對差異約為 $10^{-5}$，正好位於容許誤差邊緣；目前
  可以通過，但若日後無法通過，應先檢查此處。
- **配置**：使用 HWC（交錯）而非 CHW（平面）配置。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [閾值](../threshold/)、[邊緣偵測](../edge-detect/)、
  LeetGPU [RGB 轉灰階](../../leetgpu/066-rgb-to-grayscale/)。
