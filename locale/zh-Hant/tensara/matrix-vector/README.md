---
title: 矩陣向量乘法
platform: Tensara
upstream: matrix-vector
url: https://tensara.org/problems/matrix-vector
difficulty: easy
tags: [gemv, warp-per-row, float4, bandwidth-bound]
status: solved
---

# 矩陣向量乘法

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/matrix-vector)

## 題意

對大小為 $M\times K$ 的 $A$ 計算矩陣向量乘積
$\mathbf{c} = A\mathbf{b}$（$M$ = 4096 … 9216、$K = 4096$）。
檢查條件為 `rtol = 2e-4`、`atol = 3e-3`。與 GEMM 不同，$A$ 的每個元素
只使用一次，因此重點是以完整頻寬串流讀取 $A$。

## 圖解

![矩陣乘向量：一個 warp 以 float4 載入串流讀取 A 的一列](figure.svg)

warp 2 讀取 A 中標示的列：各 lane 交錯地取 float4 分組，與 b 中對應的分組相乘，再以 shuffle 合併部分和得到 c₂。

## 數學表述

$$
c_i = \sum_{k=0}^{K-1} A_{ik}\,b_k, \qquad 0 \le i < M
$$

| 符號 | 意義 |
|---|---|
| $A$ | 矩陣，$M\times K$，列優先 |
| $\mathbf{b}$ | 輸入向量，長度為 $K$ |
| $\mathbf{c}$ | 輸出向量，長度為 $M$ |

對每一列，warp 的 lane $\ell$ 計算跨步的部分總和，再由 warp 歸約：

$$
p_\ell = \sum_{q\,:\,q \equiv \ell \pmod{32}} \mathbf{a}^{(4)}_{iq}\cdot\mathbf{b}^{(4)}_q, \qquad c_i = \sum_{\ell=0}^{31} p_\ell
$$

| 符號 | 意義 |
|---|---|
| $\ell$ | Lane 索引 |
| $\mathbf{a}^{(4)}_{iq}, \mathbf{b}^{(4)}_q$ | 第 $i$ 列與 $\mathbf{b}$ 中的第 $q$ 個 `float4` |
| $p_\ell$ | Lane 的部分總和 |

## 解題思路

1. **每列使用一個 warp**，每個區塊 8 個 warp。每個 lane 以步距 32 讀取
   `float4`，因此一個 warp 指令會讀取 $A$ 中連續的 512 位元組。
2. **$\mathbf{b}$ 保留在快取中**：大小為 16 KB，每個 warp 都會讀取，
   第一次存取後由 L1/L2 提供。
3. **Shuffle 歸約**（`__shfl_down_sync`，5 步），再由 lane 0 寫入 $c_i$。
4. 當 $K \bmod 4 \ne 0$、各列未按 16 位元組對齊時，使用純量備援路徑。

## 成本分析

$$
Q = 4MK + 4K + 4M\ \text{bytes}, \qquad W = 2MK, \qquad I = \frac{W}{Q} \approx \frac{1}{2}\ \tfrac{\text{flop}}{\text{byte}}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數，主要來自讀取一次 $A$ |
| $W$ | 浮點運算數 |
| $I$ | 運算強度：遠低於效能轉折點，因此受頻寬限制 |

在 $9216\times4096$ 時為 151 MB，以 2 TB/s 計約 75 µs。

## 常見陷阱

- **不要使用 $N = 1$ 的 SGEMM**：其圖塊會有 98% 是填補內容。
- **對齊**：`float4` 路徑要求 $K \bmod 4 = 0$（並使用按 16 位元組對齊的緩衝區，
  `cudaMalloc` 可保證這一點）。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [NVFP4 GEMV](../nvfp4-gemv/)、[矩陣乘法](../matrix-multiplication/)、
  LeetGPU [點積](../../leetgpu/017-dot-product/)。
