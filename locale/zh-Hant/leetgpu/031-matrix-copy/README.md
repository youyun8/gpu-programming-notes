---
title: 矩陣複製
platform: LeetGPU
upstream: easy/31_matrix_copy
url: https://leetgpu.com/challenges/matrix-copy
difficulty: easy
tags: [memory-bound, vectorized, bandwidth]
status: solved
---

# 矩陣複製

**平台：** LeetGPU · **難度：** easy · [題目敘述](https://leetgpu.com/challenges/matrix-copy)

## 問題

將 $N \times N$ 的 float32 矩陣 `A` 複製到 `B`
（$1 \le N \le 4096$；基準測試為 $N = 4096$）。這裡沒有任何計算。
此問題衡量核心函式能多接近 GPU 的**峰值複製頻寬**；
這也是本站所有受限於記憶體之問題的上限。

## 公式

$$
B_k = A_k, \qquad 0 \le k < N^2
$$

| 符號 | 意義 |
|---|---|
| $N$ | 矩陣邊長 |
| $k$ | 攤平後的列優先索引 |
| $A_k,\ B_k$ | 來源與目的元素（float32） |

## 方法

- 將連續矩陣視為攤平的一維資料。
- 執行緒 $t < \lfloor N^2/4\rfloor$ 複製一個 `float4`
  （載入 16 位元組並儲存 16 位元組）；執行緒
  $t < N^2 \bmod 4$ 則複製純量尾端。
- 在此規模下，每個執行緒處理一組元素就足夠：
  $4096^2/4 = 4.2$M 個執行緒能讓每個 SM 都充滿記憶體請求。

`cudaMemcpy(…, cudaMemcpyDeviceToDevice)` 會使用複製引擎或內部核心函式
完成相同工作，但本練習要求自行撰寫核心函式。

### 複製操作的限制因素

可達頻寬取決於**同時傳輸中的位元組數**。依據 Little 定律：

$$
\text{bytes in flight} = \beta \times \lambda
$$

| 符號 | 意義 |
|---|---|
| $\beta$ | 目標頻寬（位元組/秒） |
| $\lambda$ | DRAM 延遲（約 500–800 ns） |

當 $\beta = 2$ TB/s、$\lambda \approx 600$ ns 時，必須隨時有約
1.2 MB 尚未完成。`float4` 存取讓每條指令的位元組數提高四倍，
一般占用率便足以輕易達到此需求。

## 成本分析

$$
Q = 2 \cdot 4N^2 \ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 搬移的位元組數：讀取 $N^2$ 個 float，再寫入 $N^2$ 個 float |
| $T_{\min}$ | 頻寬下限 |

當 $N = 4096$ 時，$Q = 134$ MB，因此在 2 TB/s 下
$T_{\min} \approx 67\ \mu s$。可用這個數字與
[矩陣轉置](../003-matrix-transpose/)比較。

## 常見問題

- **別名。** 此處來源與目的不會重疊。一般的 `memmove`
  則必須處理重疊情況。
- **尾端執行緒。** $N$ 為奇數時，$N^2$ 也是奇數，需要尾端執行緒。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
通過完全相等檢查。

## 相關內容

- [矩陣轉置](../003-matrix-transpose/)、
  [向量加法](../001-vector-add/)。
- [教學 02－記憶體階層](../../tutorials/02-memory-hierarchy.md)。
