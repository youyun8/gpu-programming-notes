---
title: 批次矩陣乘法
platform: LeetGPU
upstream: medium/30_batched_matrix_multiplication
url: https://leetgpu.com/challenges/batched-matrix-multiplication
difficulty: medium
tags: [gemm, batched, register-blocking, shared-memory]
status: solved
---

# 批次矩陣乘法

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/batched-matrix-multiplication)

## 問題

批次 fp32 GEMM：對 $B$ 個批次項目，分別計算 $C_b = A_b B_b$，
其中 $A_b$ 的形狀為 $M \times K$，$B_b$ 的形狀為 $K \times N$；
全部採列優先格式並連續儲存（$1 \le B \le 128$、
$1 \le M, N, K \le 1024$；基準測試為 $M = N = K = 256$；
容許誤差為 `1e-5`）。批次 GEMM 常見於注意力
（每個頭一次矩陣乘法）與分組卷積。批次只是網格的第三個維度。

## 公式

$$
C_{b,r,c} = \sum_{k=0}^{K-1} A_{b,r,k}\, B_{b,k,c}, \qquad 0 \le b < B,\ 0 \le r < M,\ 0 \le c < N
$$

$$
\text{offset}(A_{b,r,k}) = bMK + rK + k, \qquad \text{offset}(B_{b,k,c}) = bKN + kN + c, \qquad \text{offset}(C_{b,r,c}) = bMN + rN + c
$$

| 符號 | 意義 |
|---|---|
| $B$ | 批次大小（`BATCH`） |
| $M,\ N,\ K$ | $A_b$/$C_b$ 的列數、$B_b$/$C_b$ 的欄數，以及內部維度 |
| $b$ | 批次索引 |
| $r,\ c,\ k$ | 列、欄、內部索引 |
| $A_{b,r,k}$ 等 | 元素，使用上述連續三維偏移量 |

## 方法

直接重複使用[矩陣乘法](../002-matrix-multiplication/)中以暫存器分塊的
SGEMM（64 × 64 區塊分塊、16 寬 K 切片、每個執行緒計算 4 × 4
個跨距輸出，並將 $A$ 轉置存放於共享記憶體）。只需加入：

- **網格** = $\lceil N/64\rceil \times \lceil M/64\rceil \times B$，
  其中 `blockIdx.z` 是批次索引；
- 進入核心函式時，將三個指標分別加上 $bMK$、$bKN$、$bMN$
  的偏移量（以 `size_t` 計算）。

各批次項目互相獨立，因此區塊之間不需通訊。當 $M = N = 256$ 時，
每個矩陣只有 16 個區塊。真正填滿 GPU 的是批次維度：
總共有 $16 \cdot B$ 個區塊。

## 成本分析

$$
W = 2BMNK, \qquad Q \approx 4B\left(MK\frac{N}{64} + KN\frac{M}{64} + MN\right), \qquad I \approx 16 \ \text{FLOP/byte}
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP 數 |
| $Q$ | 使用 64 × 64 分塊時的 DRAM 位元組數（輸入依分塊列/欄重讀一次，輸出寫入一次） |
| $I$ | $M, N$ 很大時的算術強度 |

當 $B = 128$、矩陣大小為 $256^3$ 時，$W \approx 4.3$ GFLOP。
此核心函式受限於 fp32 FMA 計算。對小型矩陣而言，「重讀」項會命中 L2。

## 常見問題

- **批次偏移量溢位。** 當 $B = 128$ 且矩陣為 $1024^2$ 時，
  $bMK$ 可能接近 $2^{31}$（$1.3\times10^8$，仍可容納但已很接近）。
  使用 `size_t` 算術即可消除疑慮。
- **引數順序。** 核心函式的 `(rows, inner, cols)` 對應 $(M, K, N)$。
  將 $K$ 與 $N$ 搞混，只會在非方形測試中失敗。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
以 `1e-5` 容許誤差通過，包括 $B = 1$ 與維度為 1 的情況。

## 相關內容

- [矩陣乘法](../002-matrix-multiplication/)、
  [FP16 批次矩陣乘法](../057-fp16-batched-matmul/)。
- Tensara [三維矩陣乘法](../../tensara/matmul-3d/)、
  [四維矩陣乘法](../../tensara/matmul-4d/)。
