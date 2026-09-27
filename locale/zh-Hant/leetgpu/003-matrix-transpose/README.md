---
title: 矩陣轉置
platform: LeetGPU
upstream: easy/3_matrix_transpose
url: https://leetgpu.com/challenges/matrix-transpose
difficulty: easy
tags: [shared-memory, coalescing, bank-conflicts, memory-bound]
status: solved
---

# 矩陣轉置

**平台：** LeetGPU · **難度：** easy · [題目說明](https://leetgpu.com/challenges/matrix-transpose)

## 題意

轉置一個以列優先儲存的 float32 矩陣：`input` 是 $R \times C$
（「列數 × 欄數」），`output` 必須是 $C \times R$ 的矩陣
$A^{\mathsf T}$（$1 \le R, C \le 8192$；基準為 $R = 7000$、$C = 6000$）。
完全不涉及算術運算。整個問題都在於如何有效搬移記憶體，因此它是
**合併存取**與**共享記憶體 bank 衝突**的經典範例。

## 圖解

![透過共享記憶體分塊轉置：讀取與寫入都是合併存取](figure.svg)

追蹤第 1 列（深藍）：它以連續的一列讀入、存進共享分塊，再作為輸出的第 1 列（深綠）寫出。多出來的灰色一欄是填補（padding），用來消除 bank 衝突。

## 數學表述

$$
\text{out}_{c r} = \text{in}_{r c}, \qquad 0 \le r < R,\ \ 0 \le c < C
$$

$$
\text{addr}_{\text{in}}(r, c) = rC + c, \qquad \text{addr}_{\text{out}}(c, r) = cR + r
$$

| 符號 | 意義 |
|---|---|
| $R$ | 輸入的列數（`rows`） |
| $C$ | 輸入的欄數（`cols`） |
| $r,\ c$ | 輸入中的列與欄索引 |
| $\text{in}_{rc}$ | 輸入中第 $r$ 列、第 $c$ 欄的元素 |
| $\text{out}_{cr}$ | 輸出中第 $c$ 列、第 $r$ 欄的元素 |
| $\text{addr}(\cdot)$ | 以列優先儲存時的線性（元素）偏移量 |

從位址公式即可看出難點。若連續執行緒處理連續的 $c$，讀取會連續，
但寫入位置相隔 $R$ 個元素（即 $4R$ 位元組）。其中一側必然是跨距存取。

## 解題思路

### 透過共享記憶體平鋪轉置

一個 32 × 8 的執行緒區塊處理一個 32 × 32 分塊。區塊原點為
$(r_0, c_0) = (32\,\texttt{blockIdx.y},\ 32\,\texttt{blockIdx.x})$。

1. **合併讀取。** 執行緒 $(t_x, t_y)$ 讀取
   $\text{in}_{r_0 + t_y + 8j,\ c_0 + t_x}$（$j = 0..3$）至
   `tile[t_y + 8j][t_x]`。一個 warp（固定 $t_y$，
   $t_x = 0..31$）會讀取連續 128 位元組。
2. `__syncthreads()`。
3. **合併寫入。** 交換區塊座標的角色。執行緒 $(t_x, t_y)$ 將
   `tile[t_x][t_y + 8j]` 寫入
   $\text{out}_{c_0 + t_y + 8j,\ r_0 + t_x}$。此時一個 warp
   會寫入同一輸出列中的 32 個連續元素。

轉置是在共享記憶體中完成（列索引 ↔ 欄索引），此處跨距存取成本很低；
DRAM 中從不進行轉置存取。

### 以填補避免 bank 衝突

共享記憶體有 32 個 bank，每個 4 位元組。字組 $w$ 位於
$w \bmod 32$ 號 bank。步驟 3 中，一個 warp 讀取分塊的某一*欄*，
即對 $t_x = 0..31$ 讀取 $t_x \cdot P + \text{const}$，
其中 $P$ 是列間距：

$$
\text{bank}(t_x) = (t_x \cdot P + q) \bmod 32
$$

| 符號 | 意義 |
|---|---|
| $P$ | 共享分塊以 4 位元組字組計算的列間距（未填補為 32，填補後為 33） |
| $q$ | 列內固定欄偏移量 $t_y + 8j$ |
| $\text{bank}(t_x)$ | lane $t_x$ 所存取的 bank |

當 $P = 32$，所有 lane 都命中同一個 bank，形成完全序列化的
**32 路衝突**。當 $P = 33$（`tile[32][33]`），bank 為
$(t_x + q) \bmod 32$，彼此都不同，因此讀取無衝突。

## 成本分析

$$
Q = 2 \cdot 4RC \ \text{bytes}, \qquad W = 0, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 必要的 DRAM 流量：每個元素讀一次、寫一次 |
| $W$ | 算術工作量（無） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | 執行時間下限 |

在基準大小下，$Q = 2 \cdot 4 \cdot 7000 \cdot 6000 = 336$ MB。
良好的轉置效能接近裝置間複製頻寬，因此可將計時結果與
[矩陣複製](../031-matrix-copy/)比較。樸素轉置在跨距存取的一側
會浪費每個 32 位元組區段的大部分空間，頻寬通常低 3–5 倍。

## 常見陷阱

- **兩種不同的邊界檢查。** 讀取階段以輸入座標檢查
  $r < R,\ c < C$；寫入階段則依*輸出*形狀（$C \times R$）檢查。
  混用會破壞非方形矩陣的邊緣。
- **忘記填補。** 結果仍然正確，但速度慢得多，因此測試不會發現。
  只有分析工具能看出 bank 衝突
  （`l1tex__data_bank_conflicts_pipe_lsu_mem_shared`）。
- **網格方向。** 網格是 $\lceil C/32\rceil \times \lceil R/32\rceil$（x ↔ 欄）。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上通過，
包含 $1 \times 1$、$1 \times C$、$R \times 1$ 矩陣及非 32 倍數的大小。

## 延伸閱讀

- [矩陣複製](../031-matrix-copy/)：此問題的頻寬上限。
- [教學 02－記憶體階層與合併存取](../../tutorials/02-memory-hierarchy.md)。
