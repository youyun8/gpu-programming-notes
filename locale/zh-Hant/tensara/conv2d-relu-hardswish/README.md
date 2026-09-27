---
title: 搭配 ReLU 與 HardSwish 的二維卷積
platform: Tensara
upstream: conv2d-relu-hardswish
url: https://tensara.org/problems/conv2d-relu-hardswish
difficulty: medium
tags: [convolution, fusion, activation]
status: solved
---

# 搭配 ReLU 與 HardSwish 的二維卷積

**平台：** Tensara · **難度：** medium · [題目敘述](https://tensara.org/problems/conv2d-relu-hardswish)

## 問題

對 $H\times W$ 影像，以奇數大小的 $K_h\times K_w$ 核心（零填補）執行「same」二維卷積，再依序套用 ReLU 和 HardSwish，全部在一次呼叫中完成。大小最大為使用 $13\times13$ 核心的 $2048^2$ 影像。檢查誤差為 `rtol = 9e-5`、`atol = 2e-4`。

## 公式

$$
C[i, j] = \sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1} \tilde{I}\bigl[i + u - p_h,\ j + v - p_w\bigr]\,\kappa[u, v]
$$

$$
R = \max(0, C), \qquad
\operatorname{ReLU6}(t) = \min\bigl(6, \max(0, t)\bigr), \qquad
O = R \cdot \frac{\operatorname{ReLU6}(R + 3)}{6}
$$

| 符號 | 意義 |
|---|---|
| $\tilde{I}$ | 在 $H\times W$ 邊界外以零填補的輸入影像 |
| $\kappa$ | 核心 $K_h\times K_w$ |
| $p_h, p_w$ | $(K_h-1)/2$ 和 $(K_w-1)/2$ |
| $C$ | 卷積結果 |
| $R$ | 套用 ReLU 後的結果 |
| $O$ | 套用 HardSwish 後的輸出 |

由於 $R \ge 0$，組合後可簡化為：

$$
O = \begin{cases} 0, & C \le 0 \\ C\,(C + 3)/6, & 0 < C < 3 \\ C, & C \ge 3 \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $C$ | 單一像素的卷積值 |

## 方法

以 `ReluHardSwish` 尾聲函式物件具現化[二維卷積](../conv-2d/)中使用分帶共享記憶體的卷積（32 × 32 輸出圖塊、每個執行緒處理 4 列、每次暫存 8 個核心列）。啟用函數會直接套用在暫存器中的 fp32 累加器，並緊接在唯一一次儲存前執行。中間影像 $C$ 和 $R$ 從不進入記憶體；這正是融合的目的。

## 成本分析

$$
W = 2HWK_hK_w + 5HW, \qquad Q_{\text{fused}} \approx 8HW, \qquad Q_{\text{unfused}} \approx 8HW + 2\cdot 8HW
$$

| 符號 | 意義 |
|---|---|
| $W$ | Flops：卷積加上每個像素約 5 次啟用函數運算 |
| $Q_{\text{fused}}$ | 必要的位元組數：讀取影像、寫入輸出 |
| $Q_{\text{unfused}}$ | 若 ReLU 和 HardSwish 使用個別核心函式，兩者都會讀寫一張 $H\times W$ 影像 |

對小型核心（$512^2$ 影像上的 $3\times3$ 核心）而言，卷積每個像素只有 18 flops，未融合的管線會受限於記憶體頻寬，因此融合在此情況下大約可將速度提高三倍。

## 常見陷阱

- **運算順序**：先套用 ReLU，再套用 HardSwish；若只對負輸入套用 HardSwish，當 $-3 < C < 0$ 時結果並非 0。
- **除以 6**：參考實作使用除法；乘以 $1/6$ 會相差一個 ulp，雖然仍在誤差容許範圍內，但使用除法可完全一致。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 相關內容

- [二維卷積](../conv-2d/)、[ReLU](../relu/)、[GEMM + ReLU](../gemm-relu/)。
