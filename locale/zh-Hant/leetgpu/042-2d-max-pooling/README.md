---
title: 2D 最大池化
platform: LeetGPU
upstream: medium/42_2d_max_pooling
url: https://leetgpu.com/challenges/2d-max-pooling
difficulty: medium
tags: [pooling, cnn, stencil]
status: solved
---

# 2D 最大池化

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/2d-max-pooling)

## 題意

對 $N \times C \times H \times W$ float32 張量（NCHW）執行 2D 最大池化，
使用大小為 $k$ 的正方形視窗、步幅 $s$ 與寬度為 $p$ 的零填補
（$N \le 100$、$C \le 512$、$H, W \le 1024$、$k, s, p \le 16$；
效能評測使用 $N = 4$、$k = 3$、$s = 2$；容許誤差為 `1e-5`）。
結果應與 `F.max_pool2d(…, kernel_size=k, stride=s, padding=p)` 相符。

## 圖解

![二維最大池化：k × k 視窗依步幅移動，填補值永遠不會勝出](figure.svg)

紅框是輸出 (1, 2) 的 3 × 3 視窗，每個輸出移動兩格。灰色格是填補，視為 −∞，因此不可能成為最大值。

## 數學表述

$$
H_o = \left\lfloor\frac{H + 2p - k}{s}\right\rfloor + 1, \qquad W_o = \left\lfloor\frac{W + 2p - k}{s}\right\rfloor + 1
$$

$$
Y_{n,c,y,x} = \max_{\substack{0 \le a, b < k \\ 0 \le ys - p + a < H \\ 0 \le xs - p + b < W}} X_{n,\,c,\ ys - p + a,\ xs - p + b}
$$

| 符號 | 意義 |
|---|---|
| $N,\ C,\ H,\ W$ | 批次、通道、輸入高度與寬度 |
| $k$ | 視窗大小（`kernel_size`） |
| $s$ | 步幅 |
| $p$ | 每一側的填補大小（填補儲存格視為 $-\infty$，永遠不會勝出） |
| $H_o,\ W_o$ | 輸出高度與寬度 |
| $X_{n,c,h,w}$ | 輸入元素，偏移為 $((nC + c)H + h)W + w$ |
| $Y_{n,c,y,x}$ | 輸出元素，偏移為 $((nC + c)H_o + y)W_o + x$ |
| $a,\ b$ | 視窗內的偏移 |

PyTorch 要求 $p \le k/2$，因此每個視窗都至少包含一個真正的輸入儲存格，
最大值一定是有限值。

## 解題思路

- 將 $N \cdot C \cdot H_o \cdot W_o$ 攤平後，每個輸出元素使用一個
  執行緒，並採用網格跨步迴圈（網格上限為 65 535 個區塊）。
- 從平坦索引解出 $(\text{plane}, y, x)$，並讓 $x$ 變化最快。
  如此一個 warp 會寫入連續的輸出，而輸入讀取會在相同列內以 $s$ 為間距。
- 走訪 $k \times k$ 視窗。影像外的列或欄會直接 `continue`，
  這與使用 $-\infty$ 填補完全相同。

當 $k \le 16$ 且 $s < k$ 時，相鄰輸出的視窗會重疊
（例如 $3\times3$、步幅為 2）。這些重複讀取由 L1/L2 快取供應，
對如此小的視窗而言，明確使用共享記憶體分塊幫助不大。

## 成本分析

$$
W_{\text{cmp}} = k^2 N C H_o W_o, \qquad Q_{\min} = 4NC\,(HW + H_oW_o)
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{cmp}}$ | 比較次數（`fmaxf`） |
| $Q_{\min}$ | 必要的位元組數：讀取輸入一次、寫入輸出一次 |

池化受記憶體限制：最差情況下，每個輸入位元組最多進行 $k^2/4$ 次比較，
通常會少得多。

## 常見陷阱

- **輸出大小公式。** 應如 PyTorch 的 `ceil_mode=False` 使用向下取整除法。
- **填補值。** 若以 0 而非 $-\infty$ 填補，邊界上全為負值的視窗
  會得到錯誤結果。
- **大型張量。** $N C H W$ 可能超過 $2^{31}$
  （100 × 512 × 1024² 為 $5\times10^{10}$），因此所有平坦索引均使用
  `size_t`。

## 驗證

所有 LeetGPU 測試案例均在 [cuemu](../../tools/cuemu/README.md) 通過，
包括 $k = 1$、$s > k$（視窗之間有空隙）與 $p = k/2$。

## 延伸閱讀

- Tensara [1D 最大池化](../../tensara/max-pool-1d/)、
  [2D](../../tensara/max-pool-2d/)、[3D](../../tensara/max-pool-3d/)、
  [2D 平均池化](../../tensara/avg-pool-2d/)。
