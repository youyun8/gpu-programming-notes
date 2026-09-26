---
title: 點積
platform: LeetGPU
upstream: medium/17_dot_product
url: https://leetgpu.com/challenges/dot-product
difficulty: medium
tags: [reduction, two-pass, fma, deterministic]
status: solved
---

# 點積

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/dot-product)

## 問題

計算兩個長度為 $N$ 的 float32 向量之點積（$1 \le N \le 10^8$），
並將結果寫入 `result[0]`，容許誤差為 `1e-5`。這是一種歸約，
但逐元素運算不是加法，而是乘加。[歸約](../004-reduction/)的整體設計
都可沿用，只需多處理一個輸入資料流。

## 公式

$$
s = \mathbf a \cdot \mathbf b = \sum_{i=0}^{N-1} a_i\, b_i
$$

| 符號 | 意義 |
|---|---|
| $N$ | 向量長度 |
| $a_i,\ b_i$ | 輸入向量 `A`、`B` 的元素（float32） |
| $s$ | 純量結果，儲存在 `result[0]` |

每個執行緒使用**融合乘加**累加乘積：

$$
\text{acc} \leftarrow \operatorname{fma}(a_i, b_i, \text{acc}) = \operatorname{round}(a_i b_i + \text{acc})
$$

| 符號 | 意義 |
|---|---|
| acc | 該執行緒持續累加的 float32 部分和 |
| $\operatorname{fma}$ | 融合乘加：乘法與加法合併，只進行一次捨入 |
| Round | 捨入至最接近的 float32 |

相較於分開執行乘法與加法，FMA 更快（一條指令），也更準確
（只捨入一次，而不是兩次）。

## 方法

與[歸約](../004-reduction/)相同，使用兩個核心函式：

1. **`partialDots`。** 最多 1024 個區塊 × 256 個執行緒。以網格跨步迴圈
   走訪成對的 `float4`，每次迭代執行 4 次 `fmaf`，再用純量尾端處理
   $N \bmod 4$。每個區塊以 float64 歸約其中的執行緒（warp shuffle 加上
   一次共享記憶體傳遞），並寫出一個部分結果。
2. **`finalSum`。** 一個區塊以 float64 加總所有部分結果，最後只捨入一次。

區塊層級與最終層級都使用 float64，以提高準確度並確保結果可重現。
基準測試使用 $N = 5$，因此全部成本都來自啟動開銷。同一份程式碼也能以
完整頻寬處理多達 $10^8$ 個元素。

## 成本分析

$$
Q = 8N \ \text{bytes}, \qquad W = 2N, \qquad I = \frac{2N}{8N} = \frac14 \ \text{FLOP/byte}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（兩個向量各讀取一次） |
| $W$ | FLOP 數（每個元素一次乘法與一次加法） |
| $I$ | 算術強度 |
| $\beta$ | DRAM 頻寬 |

在任何 GPU 上，這個核心函式都受限於記憶體。

## 常見問題

- **單一核心函式使用原子操作。** 對 `result` 使用 `atomicAdd` 雖然可行，
  但會使低位元結果無法重現。
- **至少要有一個區塊。** 當 $N < 4$ 時，向量化區塊數會算成 0，
  因此必須限制下限為 1，再由尾端迴圈完成所有工作。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md) 通過，
包括 $N = 1..7$（只走尾端路徑）。

## 相關內容

- [歸約](../004-reduction/)、[FP16 點積](../058-fp16-dot-product/)、
  [稀疏矩陣向量乘法](../018-sparse-matrix-vector-multiplication/)。
- Tensara [餘弦相似度](../../tensara/cosine-similarity/)。
