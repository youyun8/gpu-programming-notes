---
title: Sigmoid
platform: Tensara
upstream: sigmoid
url: https://tensara.org/problems/sigmoid
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Sigmoid

**平台：** Tensara · **難度：** 簡單 · [題目敘述](https://tensara.org/problems/sigmoid)

## 題意

對 $M\times N$ 的 float32 矩陣逐元素套用 logistic sigmoid，結果須符合
`torch.sigmoid`。測試矩陣從 $4096\times4096$ 到 $8192\times8192$
（最多 6,700 萬個元素）。檢查條件為 `rtol = 1e-4`、`atol = 6e-5`。

## 圖解

![矩陣上的 sigmoid：σ(x) = 1 / (1 + e⁻ˣ)](figure.svg)

M × N 矩陣的每個元素各自獨立映射，因此 kernel 只是一連串 float4 的載入與儲存。紅點為 σ(0) = 0.5。

## 數學表述

$$
C_{ij} = \sigma(A_{ij}), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，列優先 |
| $C$ | 相同形狀的輸出矩陣 |
| $x$ | 一個輸入元素 $A_{ij}$ |
| $\sigma$ | Logistic sigmoid，值域為 $(0, 1)$ |

此公式在 fp32 的兩端都很安全：

$$
x \to +\infty:\ e^{-x} \to 0,\ \sigma \to 1; \qquad
x \to -\infty:\ e^{-x} \to +\infty,\ \sigma = 1/\infty = 0
$$

| 符號 | 意義 |
|---|---|
| $e^{-x}$ | 會溢位成 $+\infty$（當 $x < -88.7$），仍會得到正確的極限 0 |

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

`1.0f / (1.0f + expf(-x))`：一次 `expf`（範圍化簡，加上
`ex2.approx` 與修正）以及一次除法。

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

- 不要用分支「修正」**溢位**：$1/(1+\infty) = 0$ 是精確值。
- 當 $x$ 為很大的負值時，$\sigma$ 的相對誤差主要來自 `expf`；
  使用 `-use_fast_math`（`__expf`）時，絕對誤差仍很小，而檢查條件衡量的
  正是絕對誤差。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [Swish](../swish/)、[Hard Sigmoid](../hard-sigmoid/)、[Tanh](../tanh/)、
  LeetGPU [Sigmoid](../../leetgpu/068-sigmoid/)。
