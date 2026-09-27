---
title: 色彩反轉
platform: LeetGPU
upstream: easy/7_color_inversion
url: https://leetgpu.com/challenges/color-inversion
difficulty: easy
tags: [elementwise, vectorized, uint8, image]
status: solved
---

# 色彩反轉

**平台：** LeetGPU · **難度：** easy · [題目說明](https://leetgpu.com/challenges/color-inversion)

## 題意

**原地**反轉 RGBA 影像的色彩。`image` 以列優先順序存放
$H \times W$ 個像素，每個像素包含 4 個無號位元組（R、G、B、A）
（$1 \le W, H \le 4096$，$WH \le 8\,388\,608$；基準 $H = 5120$、
$W = 4096$）。R、G、B 各自替換為 $255 - v$，alpha 保持不變。
算術上非常簡單；重點在於**存取粒度**：一次搬移一個位元組會浪費
記憶體系統的大部分能力。

## 圖解

![色彩反轉：R、G、B 變成 255 − v，alpha 保持不變](figure.svg)

第 0–3 個位元組是像素 0，第 4–7 個是像素 1；第 3 與第 7 個位元組是 alpha，原樣保留。把整個像素當成一個 32 位元字組載入，一次 XOR 就能同時反轉 R、G、B。

## 數學表述

$$
\text{img}'_{p,c} =
\begin{cases}
255 - \text{img}_{p,c}, & c \in \{0, 1, 2\} \ (\text{R, G, B}) \\
\text{img}_{p,c}, & c = 3 \ (\text{A})
\end{cases}
\qquad \text{offset}(p, c) = 4p + c, \quad p = yW + x
$$

| 符號 | 意義 |
|---|---|
| $W,\ H$ | 影像的像素寬度與高度 |
| $x,\ y$ | 像素欄與列 |
| $p$ | 線性像素索引，$0 \le p < WH$ |
| $c$ | 色彩通道索引：0 = R、1 = G、2 = B、3 = A |
| $\text{img}_{p,c}$ | 反轉前的位元組值，$0..255$ |
| $\text{img}'_{p,c}$ | 反轉後的位元組值 |

對 8 位元值而言，$255 - v$ 等於位元補數 $\lnot v$，所以也可寫成
`v ^ 0xFF`。

## 解題思路

每個**像素**由一個執行緒處理。將緩衝區重新解讀成 `uchar4*`，
讓執行緒只做一次 4 位元組載入與一次 4 位元組儲存：

1. `p = blockIdx.x * 256 + threadIdx.x`，並以 `p < W*H` 防護。
2. `uchar4 v = pixels[p]`，接著執行 `v.x = 255 - v.x`
   （`.y`、`.z` 同理）；`.w`（alpha）不變。
3. `pixels[p] = v`。

如此一個 warp 會存取 $32 \times 4 = 128$ 個連續位元組，
形成一筆完全利用的交易。若改成每個*位元組*一個執行緒，
同樣資料會需要 4 倍的指令與載入／儲存操作，還要對每個位元組分支以跳過 alpha。

也能用 `uint4`（16 位元組 = 每執行緒 4 個像素），對每個 32 位元字組
套用 `v ^ 0x00FFFFFF`，再提升 4 倍粒度。此大小下，每像素一執行緒的
版本已受頻寬限制，因此保留較清楚的寫法。

## 成本分析

$$
Q = 2 \cdot 4\,WH \ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 流量：每個像素讀寫各一次（每個方向 4 位元組） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | 頻寬造成的時間下限 |

基準大小下，$WH = 2.1\times10^7$ 個像素、$Q = 168$ MB，
所以在 2 TB/s 時 $T_{\min} \approx 84\ \mu s$。算術成本可忽略。

## 常見陷阱

- **修改 alpha。** 參考實作會保持通道 3 不變。
- **對齊。** `uchar4` 載入需對齊 4 位元組。`cudaMalloc` 緩衝區
  對齊 256 位元組，符合要求。
- **原地更新。** 每個執行緒只讀寫自己的像素，因此執行緒間沒有先讀後寫危險。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 案例
（包括 1 × 1 影像）都達到精確相等（整數資料）。

## 延伸閱讀

- [RGB 轉灰階](../066-rgb-to-grayscale/)、[向量加法](../001-vector-add/)。
- Tensara [灰階](../../tensara/grayscale/)。
