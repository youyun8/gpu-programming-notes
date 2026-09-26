---
title: 通用矩陣乘法（GEMM）
platform: LeetGPU
upstream: medium/22_gemm
url: https://leetgpu.com/challenges/general-matrix-multiplication-gemm
difficulty: medium
tags: [gemm, fp16, tensor-cores, wmma]
status: solved
---

# 通用矩陣乘法（GEMM）

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/general-matrix-multiplication-gemm)

## 問題

帶縮放的半精度 GEMM：$C \leftarrow \alpha AB + \beta C$，其中 $A$ 為
$M \times K$、$B$ 為 $K \times N$、$C$ 為 $M \times N$，全都採 fp16
列優先格式；$\alpha, \beta$ 則是 float32 純量
（$16 \le M, N, K \le 4096$，不一定是 16 的倍數；基準測試為
$M = N = K = 1024$）。累加必須使用 float32，容許誤差為 `0.05`。
這道題引入了**張量核心**：每條 warp 指令可執行一次小型矩陣乘加的
專用矩陣單元，吞吐量是 fp32 FMA 的數倍。

## 公式

$$
C_{rc} \leftarrow \operatorname{fp16}\!\left(\alpha \sum_{k=0}^{K-1} \operatorname{fp32}(A_{rk})\operatorname{fp32}(B_{kc}) \;+\; \beta\,\operatorname{fp32}(C^{\text{old}}_{rc})\right)
$$

| 符號 | 意義 |
|---|---|
| $M,\ N,\ K$ | $A$/$C$ 的列數、$B$/$C$ 的欄數，以及內部維度 |
| $A_{rk},\ B_{kc}$ | fp16 輸入 |
| $C^{\text{old}}_{rc}$ | $C$ 的初始內容（fp16） |
| $\alpha,\ \beta$ | float32 純量 |
| fp32(·), fp16(·) | 轉換為 float32，以及以最接近值捨入後轉回 fp16 |

### 張量核心片段（WMMA 16 × 16 × 16）

每個 warp 呼叫一次 `wmma::mma_sync` 可計算

$$
D = A_f B_f + C_f, \qquad A_f \in \mathrm{fp16}^{16\times16},\ B_f \in \mathrm{fp16}^{16\times16},\ C_f, D \in \mathrm{fp32}^{16\times16}
$$

| 符號 | 意義 |
|---|---|
| $A_f,\ B_f$ | 使用 `load_matrix_sync` 從共享記憶體載入的矩陣片段 |
| $C_f,\ D$ | 累加器片段（float32），在整個 $K$ 迴圈中保留於暫存器 |

每條 warp 指令可執行 4096 次乘加。片段的暫存器配置不透明
（依架構而定），因此 API 只允許對它執行載入、儲存、填值與 MMA。
（[第 05 章](../../tutorials/05-amd-cdna3-mfma.md)會與配置已有文件說明的
AMD MFMA 比較。）

## 方法

### 分塊階層

| 層級 | $C$ 的分塊 | 說明 |
|---|---|---|
| 區塊（4 個 warp） | 64 × 64 | 網格為 $\lceil N/64\rceil \times \lceil M/64\rceil$ |
| Warp | 32 × 32 | 2 × 2 個累加器片段 |
| MMA | 16 × 16 × 16 | 一次 `mma_sync` |

### 以 32 為步長走訪 $K$ 的主迴圈

1. 將 `a_s[64][32+8]` 與 `b_s[32][64+8]`（fp16）從全域記憶體暫存，
   超出範圍的元素則**填零**。如此 $M$、$N$、$K$ 不必是 16 的倍數，
   而補零不會改變點積。
2. `__syncthreads()`。
3. 對 `kk = 0, 16`，每個 warp 載入 2 個 A 片段與 2 個 B 片段，
   並發出 4 次 `mma_sync`。每個 A 片段供 2 次 MMA 重複使用，
   每個 B 片段也是如此。
4. `__syncthreads()`。

### 收尾階段

累加器先儲存至共享 float32 分塊 `c_s[64][64+4]`。接著每個執行緒
處理分塊中的元素：讀取 $C^{\text{old}}$，以 float32 計算
$\alpha\cdot\text{acc} + \beta C^{\text{old}}$，轉換為 fp16，
並在邊界檢查後儲存。由於片段無法以可攜方式逐元素索引，必須經過
共享記憶體，才能套用逐元素的 $\beta$ 項與邊界檢查。

### 對齊規則（為何間距看似奇怪）

WMMA 要求指標以 32 位元組對齊，前導維度 `ldm` 則必須是
16 位元組的倍數，也就是 8 個 half 或 4 個 float。40 與 72 個 half、
以及 68 個 float 的間距都符合要求。這也會讓連續列相對於
128 位元組的 bank 週期偏移 16 位元組，使片段載入分散到各 bank。
所有共享陣列都宣告為 `__align__(32)`。

## 成本分析

$$
W = 2MNK, \qquad
Q \approx 2\left(MK\,\frac{N}{64} + KN\,\frac{M}{64}\right) + 4MN, \qquad
I \approx \frac{2MNK}{2\cdot 2MNK/64} = 32\ \text{FLOP/byte}
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP 數 |
| $Q$ | DRAM 位元組數：fp16 $A$ 每個欄分塊重讀一次，$B$ 每個列分塊重讀一次，$C$ 則各讀寫一次（2 + 2 位元組） |
| $I$ | 忽略 $C$ 項後的算術強度 |

當規模為 $1024^3$ 時，$W \approx 2.1$ GFLOP。在 A100
（fp16 張量運算 312 TFLOP/s）上，計算下限約為 7 µs。
採用同步共享記憶體暫存的 64 × 64 區塊分塊只能達到其中一小部分。
正式環境的核心函式會使用 128 × 128 以上的分塊、
`cp.async`/TMA 多階段管線，以及以 `ldmatrix` 載入片段
（CUTLASS、cuBLAS）。

## 常見問題

- **WMMA 指標未對齊**會在真實硬體上造成錯誤或讀到無效資料。
  [cuemu](../../tools/cuemu/README.md) 模擬器會檢查兩項對齊規則，
  並在開發期間抓出了一次違規。
- **就地更新 $C$。** $\beta$ 必須使用*原始* $C$。每個執行緒會在覆寫
  同一元素前讀取 `c[idx]`，且沒有其他執行緒依賴它。
- **fp16 溢位。** 最終轉換時，$\lvert x\rvert > 65504$ 會變成
  $\infty$。使用 fp32 累加可避免中間值溢位。

## 驗證

所有 LeetGPU 測試案例都已透過 [cuemu](../../tools/cuemu/README.md)
的 WMMA 模擬，包括 $M, N, K$ 不是 16 倍數的情況。核心函式也能使用
`nvcc -arch=sm_80` 編譯。

## 相關內容

- [矩陣乘法（fp32）](../002-matrix-multiplication/)、
  [FP16 批次矩陣乘法](../057-fp16-batched-matmul/)、
  [INT8 矩陣乘法](../032-int8-quantized-matmul/)、
  [INT4 矩陣乘法](../081-int4-matmul/)。
- AMD 的對應技術 MFMA：[教學 05](../../tutorials/05-amd-cdna3-mfma.md)。
