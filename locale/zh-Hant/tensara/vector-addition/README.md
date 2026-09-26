---
title: 向量加法
platform: Tensara
upstream: vector-addition
url: https://tensara.org/problems/vector-addition
difficulty: easy
tags: [elementwise, float4]
status: solved
---

# 向量加法

**平台：** Tensara · **難度：** 簡單 · [題目敘述](https://tensara.org/problems/vector-addition)

## 問題

將兩個長度為 $n$ 的 float32 向量相加，大小從 $2^{20}$ 到 $2^{30}$
個元素（最大時每個向量為 4 GB）。檢查條件為 `rtol = 2e-4`、
`atol = 1e-4`。這是 CUDA 的「Hello World」，但在 $2^{30}$ 個元素時，
64 位元索引與合理的網格大小非常重要。

## 公式

$$
c_i = a_i + b_i, \qquad 0 \le i < n
$$

| 符號 | 意義 |
|---|---|
| $a, b$ | 輸入向量（`d_input1`、`d_input2`） |
| $c$ | 輸出向量（`d_output`） |
| $n$ | 向量長度 |

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

當 $n = 2^{30}$ 時，每個元素使用一個執行緒會需要 $2^{22}$ 個 256
執行緒區塊；雖然合法（$x$ 網格限制為 $2^{31} - 1$），但很浪費。
限制區塊數的網格步進迴圈只使用 4096 個區塊，而且每個索引都是 `size_t`，
因為 $4n$ 位元組 $= 2^{32}$ 會使 32 位元算術溢位。

## 成本分析

$$
n = MN, \qquad Q = 12\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 元素數量 |
| $Q$ | 必要的 DRAM 流量：讀取一次輸入並寫入一次輸出 |
| $\beta$ | DRAM 頻寬（目前資料中心 GPU 約為 2–3 TB/s） |
| $T_{\min}$ | 核心執行時間受頻寬限制的下限 |

當 $n = 2^{30}$ 時，$Q = 12.9$ GB；在 2 TB/s 下約為 6.4 ms。
運算強度為每位元組 $1/12$ flop。

## 注意事項

- **64 位元索引**：`int i = blockIdx.x * blockDim.x + threadIdx.x`
  會在 $2^{31}$ 時溢位。
- **三條資料流**，所以 $Q = 12n$，而不是 $8n$。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [矩陣純量運算](../matrix-scalar/)、[ReLU](../relu/)、
  LeetGPU [向量加法](../../leetgpu/001-vector-add/)。
