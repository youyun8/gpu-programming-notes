---
title: 稀疏矩陣－稠密矩陣乘法
platform: LeetGPU
upstream: medium/75_sparse_matrix_dense_matrix_multiplication
url: https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication
difficulty: medium
tags: [gemm, sparsity, register-blocking]
status: solved
---

# 稀疏矩陣－稠密矩陣乘法

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication)

## 題意

計算 $C = AB$，其中 $A$（$M\times N$）有 60–70% 的零，但以稠密格式儲存；
$B$（$N \times K$）則是稠密矩陣，兩者皆為 float32 列優先格式
（$M, N, K \le 8192$；基準測試 $M = 4096$、$N = 2048$、$K = 512$；
容許誤差 `1e-3`）。利用稀疏性值得嗎？在這個密度下，**不值得**。以下用
數字說明原因。

## 圖解

![密度 35% 的稀疏 × 稠密乘法：稠密分塊 GEMM 仍然較快](figure.svg)

A 中的灰色格為零。在這種密度下略過零只能省下約三分之二的運算量，卻會帶來不規則的讀取，所以一般的分塊 GEMM 反而更快。

## 數學表述

$$
C_{ij} = \sum_{k=0}^{N-1} A_{ik} B_{kj} = \sum_{k\,:\,A_{ik}\neq 0} A_{ik}B_{kj}
$$

| 符號 | 意義 |
|---|---|
| $M,\ N,\ K$ | $A$ 的列數、內部維度、$B$ 的欄數 |
| $A_{ik}$ | 稀疏矩陣元素（大多為零） |
| $B_{kj}$ | 稠密矩陣元素 |
| $C_{ij}$ | 輸出 |
| nnz | $A$ 的非零值數量，約為 $0.35MN$ |

### 稠密與稀疏實作的損益平衡點

$$
W_{\text{dense}} = 2MNK, \qquad W_{\text{sparse}} = 2\,\text{nnz}\cdot K = 2\rho MNK
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{dense}}$ | 稠密 GEMM 的 FLOP 數 |
| $W_{\text{sparse}}$ | 僅處理非零值時的有效 FLOP 數 |
| $\rho$ | 密度 nnz / $(MN)$，此處約為 0.3–0.4 |

稀疏核心最多可省下 $1/\rho \approx 3\times$ 的 FLOP，但有以下問題：

- 它必須先轉換為 CSR：掃過 $A$ 一次再執行掃描（參見
  [串流壓縮](../072-stream-compaction/)）。
- 每個非零值都會觸發對 $B$ 第 $k$ 列的*收集*。這種存取不規則，也失去
  讓稠密 GEMM 能以很高尖峰效能比例執行的暫存器與共享記憶體重用。
- 稠密分塊 GEMM 可達尖峰效能的 50–90%，而此密度的 CSR SpMM 通常只能
  達到 5–20%。

GPU 上的交叉點通常在 $\rho \approx 1$–$5\%$。當密度為 35% 時，稠密核心
較快，這也符合 cuSPARSE 本身的建議。

## 解題思路

使用[矩陣乘法](../002-matrix-multiplication/)的 64 × 64 暫存器分塊 SGEMM：
256 個執行緒，將 $A$（以轉置方式存放）與 $B$ 沿 $K$ 的 16 寬切片放入
共享記憶體，每個執行緒負責 4 × 4 個交錯輸出，邊緣 tile 以零補齊。
直接執行與零相乘；相較於替代方案，這項成本可忽略。

## 成本分析

$$
W = 2MNK, \qquad Q \approx 4\left(MN\frac{K}{64} + NK\frac{M}{64} + MK\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 稠密乘法的 FLOP 數 |
| $Q$ | 使用 64 × 64 分塊時的 DRAM 位元組數 |

基準測試的 $W = 8.6$ GFLOP；在一張約 20 TFLOP/s 的 fp32 GPU 上，以實際
可達效率計算，約需 0.5 ms。

## 常見陷阱

- 名稱中的 **「稀疏」**。不要先假設稀疏格式一定有幫助，應實際量測。
- **維度命名。** 此處 $N$ 是內部維度。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-3`
通過。

## 延伸閱讀

- [稀疏矩陣－向量乘法](../018-sparse-matrix-vector-multiplication/)（對 GEMV 也適用相同論點）、
  [矩陣乘法](../002-matrix-multiplication/)。
