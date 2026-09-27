---
title: Leaky ReLU
platform: Tensara
upstream: leaky-relu
url: https://tensara.org/problems/leaky-relu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Leaky ReLU

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/leaky-relu)

## 題意

對 $M\times N$ 的 float32 矩陣（$4096\times4096$ 與
$6144\times4096$）逐元素套用執行時斜率 $\alpha$（測試中為 0.01 … 0.2）的
Leaky ReLU，結果須與 `F.leaky_relu(x, alpha)` 一致。檢查條件為
`rtol = 1e-4`、`atol = 1e-6`。

## 圖解

![斜率 α 於執行期給定的 Leaky ReLU（圖示 α = 0.2，即測試中的最大值）](figure.svg)

M × N 矩陣的每個元素各自獨立映射，因此 kernel 只是一連串 float4 的載入與儲存。α = 0.2 時，x = −4 對應到 −0.8（紅點）。

## 數學表述

$$
C_{ij} = \begin{cases} x, & x > 0 \\ \alpha x, & x \le 0 \end{cases} = \max(x, 0) + \alpha\,\min(x, 0), \qquad x = A_{ij}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，列優先 |
| $C$ | 輸出矩陣，形狀相同 |
| $x$ | 一個輸入元素 $A_{ij}$ |
| $\alpha$ | 負值區段的斜率，$0 < \alpha < 1$ |

## 解題思路

所有 Tensara 的逐元素問題都共用同一種核心形態：

1. **`float4` 網格跨步迴圈。** 將緩衝區視為 $\lfloor n/4 \rfloor$
   個 16 位元組向量；每次迭代載入一個 `float4`，對四個分量套用純量函式，
   再儲存一個 `float4`。`cudaMalloc` 傳回按 256 位元組對齊的指標，
   因此重新解讀型別是安全的。
2. **純量尾端處理**最後 $n \bmod 4$ 個元素。
3. **啟動** 256 執行緒的區塊，最多 4096 個區塊；網格跨步迴圈可涵蓋任何大小，
   而 4096 × 256 個執行緒足以讓 DRAM 飽和。
4. 函式是 `__forceinline__` 裝置函式，因此除了函式本身的選擇運算外，
   迴圈主體沒有分支。

選擇運算 `x > 0.0f ? x : alpha * x` 會編譯成一次乘法與一次述詞化搬移，不會發生分歧。

## 成本分析

$$
n = MN, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 元素數量 |
| $Q$ | 必要的 DRAM 傳輸量：輸入各讀取一次，輸出寫入一次 |
| $\beta$ | DRAM 頻寬（目前資料中心 GPU 約為 2–3 TB/s） |
| $T_{\min}$ | 核心執行時間的頻寬下限 |

對 $6144\times4096$ 而言：$Q = 201$ MB，在 2 TB/s 下約為 0.1 ms。

## 常見陷阱

- **引數順序**：簽章為 `(input, alpha, output, n, m)`，$\alpha$ 位於兩個指標之間。
- **嚴格的 `atol = 1e-6`**：結果必須正好是 fp32 的 $\alpha x$，因此不要用
  `x * (x > 0 ? 1 : alpha)` 搭配經過捨入的常數來計算。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [ReLU](../relu/)、[ELU](../elu/)、LeetGPU [Leaky ReLU](../../leetgpu/023-leaky-relu/)。
