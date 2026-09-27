---
title: Softplus
platform: Tensara
upstream: soft-plus
url: https://tensara.org/problems/soft-plus
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Softplus

**平台：** Tensara · **難度：** 簡單 · [題目敘述](https://tensara.org/problems/soft-plus)

## 題意

對 $M\times N$ 的 float32 矩陣逐元素套用 Softplus，結果須符合
`F.softplus`（$\beta = 1$，且使用 PyTorch 的閾值 20）。測試矩陣從
$4096\times4096$ 到 $8192\times8192$（最多 6,700 萬個元素）。
檢查條件為 `rtol = 1e-4`、`atol = 9e-5`。

## 圖解

![Softplus：平滑版的 ReLU，ln(1 + eˣ)，超過門檻 20 後直接取 x](figure.svg)

M × N 矩陣的每個元素各自獨立映射，因此 kernel 只是一連串 float4 的載入與儲存。虛線是 ReLU；x = 0 時 softplus 等於 ln 2（紅點）。

## 數學表述

$$
C_{ij} = \operatorname{softplus}(A_{ij}), \qquad
\operatorname{softplus}(x) = \begin{cases} \ln\bigl(1 + e^{x}\bigr), & x \le \tau \\ x, & x > \tau \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，列優先 |
| $C$ | 相同形狀的輸出矩陣 |
| $x$ | 一個輸入元素 $A_{ij}$ |
| $\ln(1 + e^x)$ | $\max(x, 0)$ 的平滑近似，以 `log1pf(expf(x))` 計算 |
| $\tau$ | 閾值 20：超過此值時，在 float 精度下 $\ln(1 + e^x) = x$ |

需要此閾值是因為 $e^x$ 會溢位（當 $x > 88.7$），而且在 $x = 20$ 時已經有

$$
\ln(1 + e^{x}) - x = \ln(1 + e^{-x}) \approx e^{-20} \approx 2\times10^{-9} \ll 2^{-24}\cdot 20
$$

| 符號 | 意義 |
|---|---|
| $2^{-24}\cdot 20$ | $x$ 的半個 ulp（在 $x = 20$ 時），約為 $1.2\times10^{-6}$ |

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

當 $e^x$ 很小（$x$ 為很大的負值）時，`log1pf` 可維持精度；
`logf(1 + e^x)` 則會把 $1 + e^x$ 取整為 1，回傳 0 而非 $\approx e^x$。

## 成本分析

$$
n = MN, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 元素數量 |
| $Q$ | 必要的 DRAM 流量：讀取一次輸入並寫入一次輸出 |
| $\beta$ | DRAM 頻寬（目前資料中心 GPU 約為 2–3 TB/s） |
| $T_{\min}$ | 核心執行時間受頻寬限制的下限 |

對 $8192\times8192$ 而言，$Q = 537$ MB；在 2 TB/s 時約為 0.27 ms。

## 常見陷阱

- **溢位**：若沒有閾值，$x > 88.7$ 會得到 $\ln(\infty) = \infty$。
- **下溢端**：`log1pf(expf(x))` 會回傳 $\approx e^x$（對很大的負 $x$）；
  `logf(1.0f + expf(x))` 則回傳 0。兩者都能通過 `atol`，但前者才正確。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [ReLU](../relu/)、[Sigmoid](../sigmoid/)（softplus 的導數）、[ELU](../elu/)。
