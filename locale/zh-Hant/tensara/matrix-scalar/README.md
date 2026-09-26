---
title: 矩陣純量乘法
platform: Tensara
upstream: matrix-scalar
url: https://tensara.org/problems/matrix-scalar
difficulty: easy
tags: [elementwise, float4]
status: solved
---

# 矩陣純量乘法

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/matrix-scalar)

## 問題

將 $n\times n$ 的 float32 矩陣乘以純量 $s$（$n$ = 8192 或 9216，
$s \in \{0.1, 0.2, -0.3, 0.4, -0.5\}$）。
檢查條件為 `rtol = 1e-4`、`atol = 7e-6`。

## 公式

$$
C_{ij} = s\,A_{ij}, \qquad 0 \le i, j < n
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$n\times n$ float32 |
| $s$ | 純量乘數 |
| $C$ | 輸出矩陣 |

## 方法

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

每個元素執行一次 `FMUL`；結果正是正確捨入的乘積，與 PyTorch 完全相同。

## 成本分析

$$
n = n^2, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 元素數量 |
| $Q$ | 必要的 DRAM 傳輸量：輸入各讀取一次，輸出寫入一次 |
| $\beta$ | DRAM 頻寬（目前資料中心 GPU 約為 2–3 TB/s） |
| $T_{\min}$ | 核心執行時間的頻寬下限 |

對 $n = 9216$ 而言：$Q = 680$ MB，在 2 TB/s 下約為 0.34 ms。

## 注意事項

- **簽章**：矩陣是方陣，因此只傳入一個大小 `n`；元素數為 $n^2$
  （使用 `size_t` 計算：$9216^2 = 85$ M 放得下，但更大的尺寸會讓
  以 `int` 表示的位元組索引溢位）。
- 純量以值傳遞，不是裝置指標。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [對角矩陣乘法](../diagonal-matmul/)、[向量加法](../vector-addition/)。
