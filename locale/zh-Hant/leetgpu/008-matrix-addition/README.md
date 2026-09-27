---
title: 矩陣加法
platform: LeetGPU
upstream: easy/8_matrix_addition
url: https://leetgpu.com/challenges/matrix-addition
difficulty: easy
tags: [elementwise, vectorized, memory-bound]
status: solved
---

# 矩陣加法

**平台：** LeetGPU · **難度：** easy · [題目說明](https://leetgpu.com/challenges/matrix-addition)

## 題意

將兩個 $N \times N$ float32 矩陣逐元素相加並寫入 `C`
（$1 \le N \le 4096$；基準 $N = 4096$）。矩陣以列優先方式連續儲存，
所以二維結構與計算無關。本題示範如何利用此特性進行
**向量化 128 位元存取**。

## 圖解

![在攤平的陣列上做矩陣加法：以 float4 分組，再處理剩餘的純量](figure.svg)

矩陣在記憶體中是連續的，因此二維形狀並不重要。每個執行緒處理一組 float4（深淺藍色），最後剩下的 N² mod 4 個元素（紅色）逐一處理。

## 數學表述

$$
C_{rc} = A_{rc} + B_{rc}, \qquad 0 \le r, c < N
\quad\Longleftrightarrow\quad
C_k = A_k + B_k, \qquad k = rN + c,\ \ 0 \le k < N^2
$$

| 符號 | 意義 |
|---|---|
| $N$ | 矩陣邊長 |
| $r,\ c$ | 列與欄索引 |
| $k$ | 攤平後的列優先索引 |
| $A,\ B$ | 輸入矩陣（float32） |
| $C$ | 輸出矩陣（float32） |

### 向量化拆分

$$
N^2 = 4V + R, \qquad V = \left\lfloor \frac{N^2}{4} \right\rfloor, \quad R = N^2 \bmod 4
$$

| 符號 | 意義 |
|---|---|
| $V$ | 完整 `float4` 群組數（4 個連續浮點數） |
| $R$ | 尾端剩餘的純量元素，$0 \le R \le 3$ |

## 解題思路

- 全域索引為 $t$ 且 $t < V$ 的執行緒載入 `A` 與 `B` 的第 $t$ 個
  `float4`，將四個 lane 相加，再把一個 `float4` 儲存到 `C`。
  4 個元素只需 3 條記憶體指令，而不是 12 條。
- $t < R$ 的執行緒另外處理尾端純量元素 $4V + t$。
- 網格使用 $\lceil (V + 1)/256 \rceil$ 個區塊、每區塊 256 個執行緒，
  同時涵蓋向量部分與尾端執行緒。

向量載入可減少載入／儲存指令與同時在途的記憶體請求，
讓較少的 warp 就能使記憶體系統達到峰值頻寬。DRAM 流量與純量核心相同。

## 成本分析

$$
Q = 3 \cdot 4N^2 = 12N^2 \ \text{bytes}, \qquad W = N^2, \qquad I = \frac{1}{12}, \qquad T_{\min} = \frac{12N^2}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 流量（讀取 $A$、$B$；寫入 $C$） |
| $W$ | 加法次數 |
| $I$ | 算術強度（FLOP/byte） |
| $\beta$ | DRAM 頻寬 |

$N = 4096$ 時 $Q = 201$ MB，因此在 2 TB/s 時
$T_{\min} \approx 100\ \mu s$。

## 常見陷阱

- **視為二維問題。** 使用二維網格與 `C[r][c]` 索引也正確，
  但會增加整數運算，並讓向量化變得麻煩。攤平方式更簡單、更快。
- **尾端處理。** $N$ 為偶數時，$N^2$ 必為 4 的倍數；$N$ 為奇數時，
  最後 1–3 個元素需走純量路徑。
- **對齊。** `float4` 需要 16 位元組對齊，而 `cudaMalloc` 保證符合。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上以
`1e-5` 通過，包括會走尾端路徑的奇數 $N$。

## 延伸閱讀

- [向量加法](../001-vector-add/)、[矩陣複製](../031-matrix-copy/)。
- Tensara [向量加法](../../tensara/vector-addition/)、[矩陣－純量](../../tensara/matrix-scalar/)。
