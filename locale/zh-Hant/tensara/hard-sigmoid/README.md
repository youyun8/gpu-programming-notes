---
title: 硬式 Sigmoid
platform: Tensara
upstream: hard-sigmoid
url: https://tensara.org/problems/hard-sigmoid
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# 硬式 Sigmoid

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/hard-sigmoid)

## 題意

將分段線性的「硬式」sigmoid 逐元素套用至 $M\times N$ float32 矩陣，
結果須與 `F.hardsigmoid` 一致。測試矩陣從 $4096\times4096$ 至
$8192\times8192$（最多 67 M 個元素）。檢查條件為
`rtol = 1e-4`、`atol = 6e-5`。

## 圖解

![Hard sigmoid：從 (−3, 0) 到 (3, 1) 的直線斜坡，區間外截斷](figure.svg)

M × N 矩陣的每個元素各自獨立映射，因此 kernel 只是一連串 float4 的載入與儲存。紅點標示兩個轉折處，虛線是平滑的 sigmoid。

## 數學表述

$$
C_{ij} = \operatorname{hsig}(A_{ij}), \qquad
\operatorname{hsig}(x) = \begin{cases} 0, & x \le -3 \\ \dfrac{x}{6} + \dfrac{1}{2}, & -3 < x < 3 \\ 1, & x \ge 3 \end{cases}
= \min\Bigl(1,\ \max\bigl(0,\ \tfrac{x}{6} + \tfrac{1}{2}\bigr)\Bigr)
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，以列為主 |
| $C$ | 輸出矩陣，形狀相同 |
| $x$ | 一個輸入元素 $A_{ij}$ |
| $\operatorname{hsig}$ | 硬式 sigmoid：從 $(-3, 0)$ 到 $(3, 1)$ 的線性斜坡 |

它是 logistic sigmoid 飽和點間的線性插值，在沒有快速指數函式的硬體
（行動 NPU）上成本較低。

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

程式會計算 `fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f)`：一次除法
（若允許，編譯器會改為乘上倒數）、一次加法，以及兩次 min/max 指令。

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

- **斜率 1/6 與偏移 1/2**：有些框架（Keras）使用 $0.2x + 0.5$，
  轉折點為 $\pm2.5$；PyTorch 則使用 $x/6 + 1/2$。
- `x / 6.0f` 與 `x * (1.0f / 6.0f)` 最多可能相差 1 ulp；兩者都在
  容許誤差內。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [Sigmoid](../sigmoid/)、[Swish](../swish/)、[ReLU](../relu/)。
