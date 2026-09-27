---
title: 矩陣冪
platform: LeetGPU
upstream: medium/37_matrix_power
url: https://leetgpu.com/challenges/matrix-power
difficulty: medium
tags: [gemm, binary-exponentiation, linear-algebra]
status: solved
---

# 矩陣冪

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/matrix-power)

## 問題

對一個 $N \times N$ float32 矩陣計算 $A^P$
（$1 \le N \le 1024$、$1 \le P \le 20$、$\lvert A_{ij}\rvert \le 10$；
效能評測使用 $N = 512$；容許誤差為 `1e-4`）。直觀的方法需要執行
$P - 1$ 次矩陣乘法。**平方求冪法**只需 $O(\log P)$ 次。乘法順序也會
影響 float32 的捨入結果，因此本方法選用與 `torch.linalg.matrix_power`
相同的順序。

## 公式

將 $P$ 寫成二進位：$P = \sum_{j} b_j 2^j$，其中 $b_j \in \{0, 1\}$。則

$$
A^P = \prod_{j\,:\,b_j = 1} Z_j, \qquad Z_0 = A,\quad Z_{j+1} = Z_j^2
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$N \times N$ |
| $P$ | 指數，$\ge 1$ |
| $b_j$ | $P$ 的第 $j$ 個位元 |
| $Z_j$ | 透過重複平方得到的 $A^{2^j}$ |

乘法次數為

$$
\#\text{GEMM} = \underbrace{\lfloor\log_2 P\rfloor}_{\text{squarings}} + \underbrace{\operatorname{popcount}(P) - 1}_{\text{multiplies into the result}} \;\le\; 2\lfloor\log_2 P\rfloor
$$

| 符號 | 意義 |
|---|---|
| Popcount$(P)$ | $P$ 中設為 1 的位元數 |

當 $P = 20 = 10100_2$ 時：4 次平方加 1 次乘法，共 **5 次 GEMM**，
而非 19 次。

### 配合 PyTorch 的捨入結果

矩陣乘法在數學上具有結合律，但浮點運算沒有。元素最大為 10 時，
$A^{20}$ 的動態範圍非常大，採用不同的結合順序會讓相對差異超過 `1e-4`。
此解法完全仿照 `matrix_power`：

| $P$ | 乘法順序 |
|---|---|
| 1 | $A$（複製） |
| 2 | $A \cdot A$ |
| 3 | $(A \cdot A)\cdot A$ |
| ≥ 4 | 從最低有效位元開始走訪。第一個位元之後執行 $Z \leftarrow Z^2$。遇到設為 1 的位元時，執行 $\text{res} \leftarrow \text{res}\cdot Z$（第一個設為 1 的位元直接複製 $Z$） |

## 方法

- 每次乘法都使用[矩陣乘法](../002-matrix-multiplication/)中的
  64 × 64 暫存器分塊 SGEMM；網格大小為 $\lceil N/64\rceil^2$，
  每個區塊有 256 個執行緒。
- 四個暫存緩衝區（$Z$、$Z_{\text{next}}$、result、
  result$_{\text{next}}$）會**交替使用**，確保 GEMM 不會寫入自己的輸入
  （就地 GEMM 會產生競爭條件）。
- 所有核心函式都在同一個 stream 中啟動，因此每次 GEMM 都能直接看到
  前一次的結果，不需明確同步。

## 成本分析

$$
W = 2N^3 \cdot \#\text{GEMM}, \qquad Q \approx \#\text{GEMM}\cdot 4\left(2N^2\frac{N}{64} + N^2\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 所有乘法的 FLOP 數 |
| $Q$ | DRAM 位元組數（每次 GEMM 都會在每個資料塊列／欄重新讀取輸入） |

當 $N = 512$、$P = 20$ 時，$W = 5 \cdot 2.7\times10^8 = 1.3$ GFLOP；
在現代 GPU 上執行 fp32 FMA 約需 0.1–0.2 ms。完整工作集（4 MB）
會留在 L2 快取中。

## 常見陷阱

- **別名。** `matmul(z, z, z)` 會在其他區塊仍讀取 $Z$ 時覆寫它。
- **乘法順序。** 在此處的數學中，
  $\text{res}\cdot Z = Z \cdot \text{res}$（$A$ 的冪彼此可交換），
  但 float32 結果不同。請維持參考實作的順序。

## 驗證

所有 LeetGPU 測試案例均以 `1e-4` 的容許誤差在
[cuemu](../../tools/cuemu/README.md) 通過，涵蓋 $P = 1..20$，
包括 $P \le 3$ 的特殊情況。

## 相關內容

- [矩陣乘法](../002-matrix-multiplication/)、Tensara [矩陣冪](../../tensara/matrix-power/)。
