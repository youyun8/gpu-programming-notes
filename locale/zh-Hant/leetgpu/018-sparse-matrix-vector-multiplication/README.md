---
title: 稀疏矩陣向量乘法
platform: LeetGPU
upstream: medium/18_sparse_matrix_vector_multiplication
url: https://leetgpu.com/challenges/sparse-matrix-vector-multiplication
difficulty: medium
tags: [gemv, warp-per-row, memory-bound, sparse]
status: solved
---

# 稀疏矩陣向量乘法

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication)

## 問題

計算 $\mathbf y = A\mathbf x$，其中 $A$ 是約有 60–70% 零值的
$M \times N$ 矩陣，但仍以列優先方式**密集儲存**
（$1 \le M, N \le 10^4$；基準測試為 $M = 1000$、$N = 10\,000$；
容許誤差為 `1e-3`）。題目雖提供 `nnz`，卻沒有 CSR/COO 等索引結構。
這裡的重點是先找出真正限制核心函式的因素再最佳化：瓶頸是位元組數，
不是 FLOP 數。

## 公式

$$
y_r = \sum_{c=0}^{N-1} A_{rc}\, x_c, \qquad 0 \le r < M
$$

| 符號 | 意義 |
|---|---|
| $M,\ N$ | $A$ 的列數與欄數 |
| $A_{rc}$ | 位於偏移量 $rN + c$ 的矩陣元素（通常為零） |
| $x_c$ | 長度為 $N$ 的密集輸入向量 |
| $y_r$ | 長度為 $M$ 的輸出向量 |
| nnz | $A$ 的非零元素數（核心函式未使用） |

## 方法

### 為何「稀疏」在此沒有幫助

無論 $A$ 的元素是否為零，都必須先讀取才能得知，因此不論稀疏程度為何，
傳輸量都是 $4MN$ 位元組。略過零值只能省下 FMA，而 FMA 並非瓶頸。
若先在 GPU 上轉換為 CSR，除了完整走訪一次 $A$，還要執行一次掃描；
只有同一矩陣會重複相乘許多次時才划算。

### 每列一個 Warp

- 每個區塊有 8 個 warp。Warp $w$ 處理第 $r$ 列。
- Lane $\ell$ 使用 `fmaf` 累加
  $\sum_{c \equiv \ell \pmod{32}} A_{rc} x_c$。每一步中，32 個 lane
  讀取第 $r$ 列連續的 32 個 float：共 128 位元組，能完全合併存取。
- $\mathbf x$ 最多只有 40 KB，透過 `__ldg`（唯讀快取）讀取，
  並在處理各列時持續留在 L1/L2 中。
- 經過 5 步 `__shfl_down_sync` 歸約後，由 lane 0 取得 $y_r$。

另一種選擇是每列使用一個*執行緒*。但相鄰執行緒會讀取相隔 $N$ 個
float 的位址，每次載入都會接觸不同的 sector，造成 32 倍的頻寬浪費。

## 成本分析

$$
Q \approx 4MN + 4N + 4M, \qquad W = 2MN, \qquad I \approx \frac{2MN}{4MN} = \frac12, \qquad T_{\min} \approx \frac{4MN}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 位元組數：整個矩陣一次、$\mathbf x$ 一次（已快取）、$\mathbf y$ 一次 |
| $W$ | FLOP 數 |
| $I$ | 算術強度 |
| $\beta$ | DRAM 頻寬 |

基準測試：$4MN = 40$ MB，因此在 2 TB/s 下
$T_{\min} \approx 20\ \mu s$。只有 1000 列，也就是 1000 個 warp，
占用率不高。在大型 GPU 上，將長列分給多個 warp
（再透過共享記憶體歸約）可能有所幫助。

## 常見問題

- **在 warp 內提早返回。** `if (row >= m) return;` 是安全的，因為整個
  warp 共用相同的 `row`，所以使用完整遮罩的 shuffle 絕不會在缺少
  lane 時執行。
- **容許誤差。** `1e-3` 反映了含 10 000 項的 float32 點積；
  shuffle 樹可讓誤差遠低於此值。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md) 通過，
包括 $M = 1$ 與 $N < 32$、大多數 lane 閒置的情況。

## 相關內容

- [點積](../017-dot-product/)、[稀疏 × 密集矩陣乘法](../075-sparse-matrix-dense-matrix-multiplication/)。
- Tensara [矩陣向量乘法](../../tensara/matrix-vector/)、[NVFP4 GEMV](../../tensara/nvfp4-gemv/)。
