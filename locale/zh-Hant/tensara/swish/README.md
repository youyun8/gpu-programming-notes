---
title: Swish
platform: Tensara
upstream: swish
url: https://tensara.org/problems/swish
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Swish

**平台：** Tensara · **難度：** 簡單 · [題目敘述](https://tensara.org/problems/swish)

## 題意

對 $M\times N$ 的 float32 矩陣逐元素套用 Swish（SiLU）；參考公式為
`x * torch.sigmoid(x)`。測試矩陣從 $4096\times4096$ 到
$8192\times8192$（最多 6,700 萬個元素）。檢查條件為
`rtol = 1e-4`、`atol = 4e-5`。

## 圖解

![矩陣上的 Swish（SiLU）：x · σ(x)](figure.svg)

M × N 矩陣的每個元素各自獨立映射，因此 kernel 只是一連串 float4 的載入與儲存。紅點標示最小值 ≈ −0.2785。

## 數學表述

$$
C_{ij} = x\,\sigma(x) = \frac{x}{1 + e^{-x}}, \qquad x = A_{ij}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，列優先 |
| $C$ | 相同形狀的輸出矩陣 |
| $x$ | 一個輸入元素 $A_{ij}$ |
| $\sigma$ | Logistic sigmoid |

Swish 有一小段負值凹陷，其最小值為

$$
\min_x x\,\sigma(x) \approx -0.2785 \quad \text{at}\ x \approx -1.2785
$$

| 符號 | 意義 |
|---|---|
| $x$ | Swish 達到最小值時的輸入 |

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

程式寫入 `x / (1.0f + expf(-x))`：一次指數運算與一次除法，成本與
sigmoid 相同。

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

- **很大的負 $x$**：$e^{-x} = \infty$ 會得到 $x/\infty = -0$，
  這是正確的極限。
- **參考計算順序**：PyTorch 會先計算 $\sigma(x)$ 再相乘；直接相除的
  結果最多相差 1–2 ulp。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [Sigmoid](../sigmoid/)、[矩陣乘法 + Swish](../matmul-swish/)、
  LeetGPU [SiLU](../../leetgpu/052-silu/)。
