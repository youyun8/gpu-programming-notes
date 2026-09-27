---
title: 對角矩陣乘法
platform: Tensara
upstream: diagonal-matmul
url: https://tensara.org/problems/diagonal-matmul
difficulty: easy
tags: [elementwise, matmul, bandwidth-bound]
status: solved
---

# 對角矩陣乘法

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/diagonal-matmul)

## 題意

計算 $C = \operatorname{diag}(\mathbf{a})\,B$，其中 $\mathbf{a}$ 是長度為
$N$ 的向量，而 $B$ 是 $N\times M$ 的 float32 矩陣（最大為
$8192\times4096$）。參考實作真的會建立 `torch.diag(A) @ B`，這是需要
$N^3$ 次浮點運算的 GEMM；但其結果其實只是將 $B$ 的每一列分別乘上一個
數值。檢查條件為 `rtol = 1e-4`、`atol = 3e-5`。

## 圖解

![diag(a) · B：不必建立 N × N 的對角矩陣，只要把 B 的第 i 列乘上 aᵢ](figure.svg)

B 的每一列乘上 a 中的一個數，就得到 C 中對應的列。整題就是一個串流式的逐元素 kernel。

## 數學表述

$$
\operatorname{diag}(\mathbf{a})_{ik} = \begin{cases} a_i, & i = k \\ 0, & i \ne k \end{cases}
\quad\Longrightarrow\quad
C_{ij} = \sum_{k=0}^{N-1} \operatorname{diag}(\mathbf{a})_{ik} B_{kj} = a_i\,B_{ij}
$$

| 符號 | 意義 |
|---|---|
| $\mathbf{a}$ | 對角線元素，長度為 $N$ |
| $\operatorname{diag}(\mathbf{a})$ | $N\times N$ 對角矩陣（不會實際建立） |
| $B$ | 輸入矩陣 $N\times M$，以列為主 |
| $C$ | 輸出矩陣 $N\times M$ |
| $i, j, k$ | 列、欄與加總索引 |

## 解題思路

使用二維網格：`blockIdx.x` 涵蓋 256 欄，`blockIdx.y`（以網格跨步方式，
最多 65535）涵蓋各列。每個執行緒載入 $a_i$（整個區塊都讀取同一個位址，
由快取提供廣播）、$B$ 的一個元素，並儲存 $C$ 的一個元素。同一列中的
載入與儲存都是合併存取。

## 成本分析

$$
Q = 8NM + 4N\ \text{bytes}, \qquad W = NM\ \text{mults}, \qquad T_{\min} = \frac{8NM}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取 $B$、寫入 $C$、讀取 $\mathbf{a}$ |
| $W$ | 乘法次數 |
| $\beta$ | DRAM 頻寬 |

在 $8192\times4096$ 時：$Q = 268$ MB，以 2 TB/s 計算約需 0.13 ms。
GEMM 方法則會執行 $2N^2M = 5.5\times10^{11}$ 次浮點運算，工作量多出
數百倍。向量化的 `float4` 存取是僅剩的微幅最佳化空間。

## 常見陷阱

- **辨識運算結構**：在 $N = 8192$ 時，$N\times N$ 暫存矩陣會白白占用
  256 MB。
- **縮放列而非欄**：$\operatorname{diag}(\mathbf{a})B$ 會縮放各列；
  $B\operatorname{diag}(\mathbf{a})$ 才會縮放各欄。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [矩陣純量運算](../matrix-scalar/)、[下三角矩陣乘法](../lower-trig-matmul/)、
  [對稱矩陣乘法](../symmetric-matmul/)。
