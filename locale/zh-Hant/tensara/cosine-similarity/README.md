---
title: 餘弦相似度
platform: Tensara
upstream: cosine-similarity
url: https://tensara.org/problems/cosine-similarity
difficulty: easy
tags: [loss, reduction, row-per-block]
status: solved
---

# 餘弦相似度

**平台：** Tensara · **難度：** easy · [題目敘述](https://tensara.org/problems/cosine-similarity)

## 問題

對 $N$ 組長度為 $D$ 的 float32 向量配對（`predictions` 的第 $i$ 列與 `targets` 的第 $i$ 列），輸出餘弦*距離*
$1 - \cos(\mathbf{p}_i, \mathbf{t}_i)$。參考實作為
`1 - F.cosine_similarity(p, t, dim=1)`，其中 $\epsilon = 10^{-8}$。檢查誤差為 `rtol = atol = 1e-4`。

## 公式

$$
\text{out}_i = 1 - \frac{\mathbf{p}_i\cdot\mathbf{t}_i}{\sqrt{\max\bigl(\lVert\mathbf{p}_i\rVert^2\,\lVert\mathbf{t}_i\rVert^2,\ \epsilon^2\bigr)}}, \qquad
\mathbf{p}_i\cdot\mathbf{t}_i = \sum_{j=0}^{D-1} p_{ij}t_{ij}, \qquad
\lVert\mathbf{p}_i\rVert^2 = \sum_{j=0}^{D-1} p_{ij}^2
$$

| 符號 | 意義 |
|---|---|
| $N$ | 向量配對數（列數） |
| $D$ | 向量長度（欄數） |
| $\mathbf{p}_i, \mathbf{t}_i$ | `predictions` 與 `targets` 的第 $i$ 列 |
| $p_{ij}, t_{ij}$ | 兩者的元素 |
| $\lVert\cdot\rVert$ | 歐幾里得（L2）範數 |
| $\epsilon$ | $10^{-8}$；避免除以零 |
| $\text{out}_i$ | 第 $i$ 組配對的損失，範圍為 $[0, 2]$ |

題目敘述將分母寫成
$\max(\epsilon, \lVert\mathbf{p}\rVert)\cdot\max(\epsilon, \lVert\mathbf{t}\rVert)$；
目前的 PyTorch 則如上式，限制平方範數乘積的下限。兩者只會在某個範數小於 $10^{-8}$ 時出現差異，而隨機測試資料不會發生這種情況。

## 方法

1. **每列使用一個執行緒區塊**（256 個執行緒）。每個執行緒跨步走訪該列，並在一趟處理中累加三個總和
   $\sum pt$、$\sum p^2$、$\sum t^2$，因此每個輸入元素只載入一次。
2. 執行**三次區塊縮減**（warp `__shfl_xor_sync` 蝶式運算，再使用一個 32 項的共享陣列）。輔助函式最後會執行 `__syncthreads()`，使下一次呼叫能重複使用共享暫存空間。
3. 執行緒 0 寫入
   $1 - \text{dot}/\sqrt{\max(pp\cdot tt, 10^{-16})}$。

## 成本分析

$$
Q = 8ND + 4N\ \text{bytes}, \qquad W = 6ND\ \text{flops}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：兩個矩陣各讀取一次，每列輸出一個浮點數 |
| $W$ | 每組元素配對執行三次 FMA |
| $\beta$ | DRAM 頻寬 |

算術強度低於 1 flop/byte：完全受限於記憶體頻寬。若 $N$ 很小（列數少於 SM 數量的約 2 倍）且 $D$ 很大，就需要讓每列使用多個執行緒區塊，才能充分利用 GPU。

## 常見陷阱

- **輸出為 $1 - \cos$**，不是 $\cos$，也不是 $-\cos$。
- **只處理一趟**：用不同迴圈分別計算範數和內積，會讓流量增為三倍。
- 使用**乘積的 `sqrt`**，而非兩個 `sqrt` 的乘積，可避免一次捨入，並更貼近 PyTorch。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 相關內容

- [L2 範數](../l2-norm/)、[三元組邊界損失](../triplet-margin/)、
  LeetGPU [內積](../../leetgpu/017-dot-product/)。
