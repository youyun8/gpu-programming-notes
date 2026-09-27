---
title: 影像閾值處理
platform: Tensara
upstream: threshold
url: https://tensara.org/problems/threshold
difficulty: easy
tags: [elementwise, image-processing, float4]
status: solved
---

# 影像閾值處理

**平台：** Tensara · **難度：** 簡單 · [題目敘述](https://tensara.org/problems/threshold)

## 題意

對高度為 $h$、寬度為 $w$（$1024\times768$ 到 $3840\times2160$）的
float32 灰階影像，以執行期閾值 $\theta$（64、128 或 192）進行二值化：
嚴格大於 $\theta$ 的像素變成 255，其餘變成 0。檢查器會精確比對。

## 圖解

![二值化：像素嚴格大於 θ 時輸出 255，否則輸出 0](figure.svg)

輸出在 θ = 128 之後從 0 跳到 255；紅點表示像素恰好等於 θ 時仍輸出 0。

## 數學表述

$$
\text{out}[i, j] = \begin{cases} 255, & I[i, j] > \theta \\ 0, & \text{otherwise} \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $I$ | 輸入影像，$h\times w$ float32，值介於 $[0, 255]$ |
| $\theta$ | 閾值（`threshold_value`） |
| out | 二值輸出影像，值屬於 $\{0, 255\}$ |

## 解題思路

所有 Tensara 逐元素問題都使用同一種核心結構：

1. **`float4` 網格步進迴圈。** 將緩衝區視為 $\lfloor n/4 \rfloor$
   個 16 位元組向量；每次迭代載入一個 `float4`，對四個 lane 套用純量函式，
   再儲存一個 `float4`。`cudaMalloc` 傳回以 256 位元組對齊的指標，
   因此重新解讀型別是安全的。
2. **純量尾端處理**最後的 $n \bmod 4$ 個元素。
3. **啟動** 256 執行緒的區塊，最多 4096 個區塊；網格步進迴圈可涵蓋任意
   大小，而 4096 × 256 個執行緒足以用滿 DRAM。
4. 此函式是 `__forceinline__` 裝置函式，因此除函式本身的選擇操作外，
   迴圈主體沒有分支。

選擇式 `x > threshold ? 255.0f : 0.0f` 完全沒有算術運算；
這是純粹的頻寬測試。

## 成本分析

$$
n = hw, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 元素數量 |
| $Q$ | 必要的 DRAM 流量：讀取一次輸入並寫入一次輸出 |
| $\beta$ | DRAM 頻寬（目前資料中心 GPU 約為 2–3 TB/s） |
| $T_{\min}$ | 核心執行時間受頻寬限制的下限 |

對 $3840\times2160$ 而言，$Q = 66$ MB；在 2 TB/s 時約為 33 µs。
在此尺寸下，數 µs 的啟動額外負擔已清楚可見。

## 常見陷阱

- **嚴格不等式**：等於 $\theta$ 的像素會映射至 0。由於輸出必須完全相符，
  對整數值輸入使用 `>=` 會失敗。
- **引數順序**：`(input_image, threshold_value, output_image, height, width)`。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [灰階](../grayscale/)、[邊緣偵測](../edge-detect/)、[直方圖](../histogram/)。
