---
title: Frobenius 正規化
platform: Tensara
upstream: frobenius-norm
url: https://tensara.org/problems/frobenius-norm
difficulty: easy
tags: [normalization, reduction, fp64-accumulation, grid-reduction]
status: solved
---

# Frobenius 正規化

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/frobenius-norm)

## 問題

將任意形狀的 float32 張量中每個元素除以該張量的 Frobenius 範數。只會
傳入元素總數 $n$；測試包含 4 M 至 33 M 個元素（例如
$(4, 16, 32, 128, 128)$）。檢查條件為 `rtol = 3e-3`、`atol = 1e-6`。

## 公式

$$
\lVert X \rVert_F = \sqrt{\sum_{k=0}^{n-1} x_k^2}, \qquad y_k = \frac{x_k}{\lVert X \rVert_F}
$$

| 符號 | 意義 |
|---|---|
| $X$ | 輸入張量，視為含有 $n$ 個元素的一維向量 |
| $x_k, y_k$ | 第 $k$ 個輸入與輸出元素 |
| $\lVert X\rVert_F$ | Frobenius 範數：展平後張量的 L2 範數 |

全域總和會拆成各區塊的部分和：

$$
S = \sum_{b=0}^{G-1} S_b, \qquad S_b = \sum_{k \in \mathcal{K}_b} x_k^2, \qquad r = \frac{1}{\sqrt{S}}
$$

| 符號 | 意義 |
|---|---|
| $G$ | 第一個核心中的區塊數量（$\le 1024$） |
| $\mathcal{K}_b$ | 區塊 $b$ 的網格跨步迴圈所走訪的索引 |
| $S_b$ | 區塊部分和（以 `double` 儲存） |
| $r$ | 範數倒數，`g_inv_norm` |

## 方法

1. **`sumSquares`**：以網格跨步迴圈走訪 `float4` 向量；每個執行緒使用
   float 累加最多數百個元素的 $x^2$，接著區塊以 `double` 歸約每個
   執行緒的總和（warp shuffle 加上共享記憶體），並將 $S_b$ 寫入裝置
   陣列。
2. **`finishNorm`**：一個區塊以 `double` 加總 $G$ 個部分和，並將
   $r = 1/\sqrt{S}$ 以 float 儲存。
3. **`scale`**：逐元素計算 $y_k = x_k\,r$。

分開啟動核心即可免費取得整個裝置的同步屏障；不需要原子操作或
cooperative groups，且結果具有確定性（每次執行的分割方式都相同）。

## 成本分析

$$
Q = 4n\ (\text{pass 1}) + 8n\ (\text{pass 3}) = 12n\ \text{bytes}, \qquad T_{\min} = \frac{12n}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數 |
| $\beta$ | DRAM 頻寬 |

在 $n = 33.5$ M 時：共 403 MB，以 2 TB/s 計算約需 0.2 ms。兩趟處理
都無法避免：在完整總和確定之前不能寫入任何輸出（除非張量能放入晶片內
記憶體，但此處無法做到）。

## 注意事項

- **精確度**：使用單一 float 累加器加總 $3\times10^7$ 個平方值會失去
  數個有效位；fp64 部分和可讓範數保持 fp32 的精確度。
- **乘以 $r$** 而非除以範數：最多相差 1 ulp，遠在
  `rtol = 3e-3` 的範圍內。
- **任意形狀**：只有 $n$ 會影響運算，因此核心不依賴形狀。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 相關內容

- [L2 範數](../l2-norm/)（逐列）、[MSE 損失](../mse-loss/)（相同的
  兩層歸約）、LeetGPU [歸約](../../leetgpu/004-reduction/)。
