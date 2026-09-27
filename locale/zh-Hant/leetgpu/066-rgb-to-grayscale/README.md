---
title: RGB 轉灰階
platform: LeetGPU
upstream: easy/66_rgb_to_grayscale
url: https://leetgpu.com/challenges/rgb-to-grayscale
difficulty: easy
tags: [image, elementwise, strided-access]
status: solved
---

# RGB 轉灰階

**平台：** LeetGPU · **難度：** 簡單 · [題目說明](https://leetgpu.com/challenges/rgb-to-grayscale)

## 問題

使用 ITU-R BT.601 亮度權重，將 $H\times W$ RGB 影像
（float32，每個像素的 R、G、B 交錯排列，值域為 $[0, 255]$）轉為灰階
（$WH \le 4.2$M；基準測試為 $2048 \times 2048$；容許誤差 `1e-5`）。

## 公式

$$
Y_p = 0.299\,R_p + 0.587\,G_p + 0.114\,B_p, \qquad (R_p, G_p, B_p) = (x_{3p},\ x_{3p+1},\ x_{3p+2})
$$

| 符號 | 意義 |
|---|---|
| $p$ | 像素索引 $yW + x$，$0 \le p < WH$ |
| $x_k$ | 輸入陣列（長度為 $3WH$） |
| $R_p,\ G_p,\ B_p$ | 像素 $p$ 的紅、綠、藍值 |
| $Y_p$ | 輸出亮度 |
| 0.299, 0.587, 0.114 | BT.601 權重（總和為 1；由於人眼對綠色最敏感，所以綠色權重最高） |

## 方法

每個像素使用一個執行緒，從 `input + 3p` 讀取 3 個連續 float，再寫入一個 float。

### 跨步讀取是否浪費？

對每條指令而言，一個 warp 以 12 位元組步幅讀取 32 個 float：涵蓋
384 位元組的範圍，卻只使用三分之一。但三條指令（R、G、B）合計恰好涵蓋
同一個 384 位元組範圍。此範圍只會從 DRAM 擷取一次，第二與第三次載入都會命中 L1。
因此 DRAM 流量等於輸入大小。效率損失只在 L1 交易數量，而非頻寬。

替代方法包括：以合併的 `float4` 載入將像素暫存至共享記憶體，
或讓每個執行緒處理 4 個像素 = 3 次 `float4` 載入。這些方法可減少指令數，
但不會減少 DRAM 位元組數。

## 成本分析

$$
Q = 12WH + 4WH = 16WH\ \text{bytes}, \qquad W_{\text{flop}} = 5WH
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：每個像素讀取 3 個 float、寫入 1 個 |
| $W_{\text{flop}}$ | 每個像素 3 次乘法 + 2 次加法 |

基準測試（$2048^2$）為 67 MB，在 2 TB/s 下約需 34 µs。

## 注意事項

- **運算順序。** PyTorch 以 float32 從左至右計算
  `0.299*R + 0.587*G + 0.114*B`。編譯器可能融合成 FMA，造成最後一個位元不同，
  但對不超過 255 的值而言，仍遠低於 `1e-5`。
- **Double 常值。** 寫成 `0.299` 而不是 `0.299f`，會在未察覺下提升為
  float64 運算。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆以
`1e-5` 通過。

## 相關內容

- [色彩反轉](../007-color-inversion/)、Tensara [灰階](../../tensara/grayscale/)。
