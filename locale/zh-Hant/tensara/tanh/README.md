---
title: Tanh
platform: Tensara
upstream: tanh
url: https://tensara.org/problems/tanh
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# Tanh

**平台：** Tensara · **難度：** 簡單 · [題目敘述](https://tensara.org/problems/tanh)

## 問題

對 $M\times N$ 的 float32 矩陣逐元素套用雙曲正切，結果須符合
`torch.tanh`。測試矩陣從 $4096\times4096$ 到 $8192\times8192$
（最多 6,700 萬個元素）。檢查條件為 `rtol = 1e-4`、`atol = 6e-5`。

## 公式

$$
C_{ij} = \tanh(A_{ij}), \qquad \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = 2\sigma(2x) - 1
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，列優先 |
| $C$ | 相同形狀的輸出矩陣 |
| $x$ | 一個輸入元素 $A_{ij}$ |
| $\tanh$ | 雙曲正切，值域為 $(-1, 1)$ |
| $\sigma$ | Logistic sigmoid |

## 方法

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

程式呼叫 CUDA 的 `tanhf`。它會以多項式處理較小的 $|x|$
（避免 $e^x - e^{-x}$ 的消去誤差），並飽和至 $\pm1$（當 $|x|$ 很大時）。
硬體 `tanh.approx.f32`（sm_75+）較快，但相對誤差約為 $2^{-11}$；
在接近 0 時，對 `rtol = 1e-4` 而言太粗略。

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

## 注意事項

- **直觀公式**：$(e^x - e^{-x})/(e^x + e^{-x})$ 會成為
  $\infty/\infty =$ NaN（當 $|x| > 88$）；接近 0 時也不準確。
- **快速數學運算**會以近似指令取代 `tanhf`。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [Sigmoid](../sigmoid/)、[GELU](../gelu/)、[Hard Sigmoid](../hard-sigmoid/)。
