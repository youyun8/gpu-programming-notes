---
title: GELU
platform: Tensara
upstream: gelu
url: https://tensara.org/problems/gelu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# GELU

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/gelu)

## 題意

以 **tanh 近似式**將 GELU 逐元素套用至 $M\times N$ float32 矩陣
（`F.gelu(x, approximate="tanh")`）。測試矩陣從 $4096\times4096$
至 $8192\times8192$（最多 67 M 個元素）。檢查條件為
`rtol = 1e-4`、`atol = 2e-5`。

## 圖解

![採用 tanh 近似的 GELU，幾乎與精確的 erf 曲線重疊](figure.svg)

M × N 矩陣的每個元素各自獨立映射，因此 kernel 只是一連串 float4 的載入與儲存。虛線為精確曲線，在此範圍內與近似值的差距小於 0.001。

## 數學表述

精確的 GELU 會以標準常態變數小於 $x$ 的機率對 $x$ 加權：

$$
\operatorname{GELU}(x) = x\,\Phi(x) = \frac{x}{2}\left(1 + \operatorname{erf}\frac{x}{\sqrt{2}}\right)
$$

| 符號 | 意義 |
|---|---|
| $x$ | 一個輸入元素 |
| $\Phi(x)$ | 標準常態累積分布函數 |
| $\operatorname{erf}$ | Gauss 誤差函數 |

本題要求 tanh 近似式：

$$
C_{ij} = \frac{x}{2}\Bigl(1 + \tanh\bigl(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\bigr)\Bigr), \qquad x = A_{ij}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，以列為主 |
| $C$ | 輸出矩陣，形狀相同 |
| $x$ | 一個輸入元素 $A_{ij}$ |
| $\sqrt{2/\pi}$ | $\approx 0.7978845608$（`kSqrt2OverPi`） |
| $0.044715$ | 為使 tanh 形式符合 $\Phi$ 而擬合的三次項係數（`kCubic`） |

## 解題思路

所有 Tensara 逐元素問題都採用同一種核心形狀：

1. **`float4` 網格跨步迴圈。** 將緩衝區視為 $\lfloor n/4 \rfloor$ 個
   16 位元組向量；每次迭代載入一個 `float4`、將純量函式套用至四條
   lane，再儲存一個 `float4`。`cudaMalloc` 回傳以 256 位元組對齊的
   指標，因此重新解讀型別是安全的。
2. **純量尾端**處理最後 $n \bmod 4$ 個元素。
3. **啟動** 256 執行緒的區塊，最多 4096 個區塊；網格跨步迴圈可處理
   任何大小，4096 × 256 個執行緒也足以讓 DRAM 達到飽和。
4. 函式是 `__forceinline__` 裝置函式，因此除了函式本身的選擇指令之外，
   迴圈主體沒有分支。

程式以 `x + kCubic * x * x * x` 計算多項式，並呼叫 `tanhf`（完整
精確度；`__tanhf` 或 `tanh.approx` 雖然較快，但在接近 0 時精確度低於
容許範圍）。

## 成本分析

$$
n = MN, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 元素數量 |
| $Q$ | 必要的 DRAM 流量：輸入讀取一次，輸出寫入一次 |
| $\beta$ | DRAM 頻寬（目前資料中心 GPU 約為 2–3 TB/s） |
| $T_{\min}$ | 核心執行時間的頻寬下限 |

在 $8192\times8192$ 時：$Q = 537$ MB，以 2 TB/s 計算約需 0.27 ms。

## 常見陷阱

- **使用哪一種 GELU**：erf 形式與 tanh 形式最多相差約 $10^{-3}$，
  大於容許誤差。本題使用 tanh 形式（題目要求精確 GELU 時則使用 erf
  形式）。
- **快速數學 `tanh`**（`-use_fast_math`）對較小引數會失去相對精確度。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [ELU](../elu/)、[Swish](../swish/)、[Sigmoid](../sigmoid/)、
  LeetGPU [GEGLU](../../leetgpu/065-geglu/)。
