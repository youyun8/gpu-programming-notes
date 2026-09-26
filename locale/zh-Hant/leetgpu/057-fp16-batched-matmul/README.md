---
title: FP16 批次矩陣乘法
platform: LeetGPU
upstream: medium/57_fp16_batched_matmul
url: https://leetgpu.com/challenges/fp16-batched-matrix-multiplication
difficulty: medium
tags: [gemm, fp16, tensor-cores, wmma, batched]
status: solved
---

# FP16 批次矩陣乘法

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/fp16-batched-matrix-multiplication)

## 問題

批次半精度 GEMM：計算 $C_b = A_b B_b$，其中 $b = 0..B-1$，且
$A_b \in \mathrm{fp16}^{M\times K}$、$B_b \in \mathrm{fp16}^{K\times N}$，
以 fp32 累加並輸出 fp16 結果（$B \le 128$、$M, N, K \le 1024$；
基準測試為 $256^3$；容許誤差 `0.05`）。它結合了
[GEMM（fp16）](../022-gemm/)的 Tensor Core 核心與
[批次矩陣乘法](../030-batched-matrix-multiplication/)的批次處理。

## 公式

$$
C_{b,r,c} = \operatorname{fp16}\!\Bigl(\sum_{k=0}^{K-1} \operatorname{fp32}(A_{b,r,k})\,\operatorname{fp32}(B_{b,k,c})\Bigr)
$$

| 符號 | 意義 |
|---|---|
| $B$ | 批次大小（`BATCH`） |
| $M,\ N,\ K$ | 輸出列數、輸出欄數、內部維度 |
| $A_{b,r,k}$ | 位移 $bMK + rK + k$ 的 fp16 元素 |
| $B_{b,k,c}$ | 位移 $bKN + kN + c$ 的 fp16 元素 |
| $C_{b,r,c}$ | 位移 $bMN + rN + c$ 的 fp16 結果 |
| fp32(·), fp16(·) | 擴寬轉換與四捨五入至最近值的縮窄轉換 |

## 方法

- **網格**為 $\lceil N/64\rceil \times \lceil M/64\rceil \times B$。
  每個區塊依批次索引移動三個指標（使用 `size_t`）。
- **區塊**：4 個 warp，負責一個 64 × 64 輸出圖塊。每個 warp 負責
  32 × 32 = 2 × 2 個 WMMA 16 × 16 × 16 累加器（fp32）。
- **K 迴圈**以 32 為切片：暫存補零後的 fp16 圖塊
  （`a_s[64][40]`、`b_s[32][72]`），接著執行兩個 `kk` 步驟，
  每個 warp 各做 4 次 `mma_sync`。
- **結尾階段**：將累加器放入共享的 fp32 圖塊，再進行邊界檢查後的 fp16 寫入。

共享記憶體 pitch 符合 WMMA 的 `ldm` 規則（8 個 half 的倍數 = 16 位元組），
且每個 fragment 指標都對齊 32 位元組（請參閱 [GEMM](../022-gemm/)）。

## 成本分析

$$
W = 2BMNK, \qquad Q \approx 2B\left(MK\frac{N}{64} + KN\frac{M}{64}\right) + 2BMN
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP 數（在 Tensor Core 上執行） |
| $Q$ | 位元組數：每個圖塊列／欄重新讀取一次 fp16 運算元，輸出寫入一次（每元素 2 位元組） |

當 $B = 128$ 且矩陣為 $256^3$ 時：$W = 4.3$ GFLOP，在 A100
fp16 稠密效能 312 TFLOP/s 下約需 14 µs。每個矩陣只有 16 個區塊，
因此要靠大型批次才能讓 SM 飽和。

## 注意事項

- **未對齊的 fragment 指標**（請參閱 [GEMM](../022-gemm/)）。
- **結果轉換。** 最後才從 fp32 四捨五入一次，才能與參考實作一致。
  使用 fp16 累加不會得到相同結果。

## 驗證

在 [cuemu](../../tools/cuemu/README.md)（WMMA 模擬）上，所有 LeetGPU
測試案例皆以 `0.05` 通過，包括不是 16 倍數的維度。

## 相關內容

- [GEMM（fp16）](../022-gemm/)、[批次矩陣乘法（fp32）](../030-batched-matrix-multiplication/)。
