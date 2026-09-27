---
title: 類別交叉熵損失
platform: LeetGPU
upstream: medium/25_categorical_cross_entropy_loss
url: https://leetgpu.com/challenges/categorical-cross-entropy-loss
difficulty: medium
tags: [reduction, logsumexp, warp-per-row, loss]
status: solved
---

# 類別交叉熵損失

**平台：** LeetGPU · **難度：** medium · [題目敘述](https://leetgpu.com/challenges/categorical-cross-entropy-loss)

## 題意

計算一個批次的平均類別交叉熵：`logits` 是 $N \times C$ 的 float32
矩陣，`true_labels` 則存放 $N$ 個類別索引
（$1 \le N \le 10^4$、$2 \le C \le 1000$、$\lvert z\rvert \le 10$；
容許誤差為 `1e-5`）。輸出是單一 float。這是標準的分類損失。
數值運算的重點是 **log-sum-exp**，絕不能直接以
`log(sum(exp(z)))` 這種方式計算。

## 圖解

![交叉熵：每一列取 logits 的 log-sum-exp，再減去正確類別的 logit](figure.svg)

每一列是一個樣本，紅色格是其正確類別。該列的損失等於 log-sum-exp 減去紅格的 logit；整批的損失是右側三個數值的平均。

## 數學表述

$$
\mathcal L = \frac{1}{N}\sum_{j=0}^{N-1} \ell_j, \qquad
\ell_j = -\log \frac{e^{z_{j,y_j}}}{\sum_{k=0}^{C-1} e^{z_{jk}}} = \operatorname{LSE}(\mathbf z_j) - z_{j, y_j}
$$

$$
\operatorname{LSE}(\mathbf z_j) = \log\sum_{k=0}^{C-1} e^{z_{jk}} = m_j + \log\sum_{k=0}^{C-1} e^{z_{jk} - m_j}, \qquad m_j = \max_k z_{jk}
$$

| 符號 | 意義 |
|---|---|
| $N$ | 批次大小（樣本數） |
| $C$ | 類別數 |
| $z_{jk}$ | 樣本 $j$ 對類別 $k$ 的 logit（列優先，偏移量為 $jC + k$） |
| $\mathbf z_j$ | logits 的第 $j$ 列 |
| $y_j$ | 樣本 $j$ 的真實類別，$0 \le y_j < C$ |
| $\ell_j$ | 樣本 $j$ 的損失（真實類別的負對數概似） |
| $\operatorname{LSE}$ | log-sum-exp；以最大值 $m_j$ 平移，可讓每個指數都 $\le 0$ |
| $\mathcal L$ | 批次平均損失，寫入 `loss[0]` |

LSE 使用 [Softmax](../005-softmax/) 的線上 $(m, s)$ 配對合併，
只需走訪一次：
$(m_1,s_1)\oplus(m_2,s_2) = (m, s_1e^{m_1-m} + s_2e^{m_2-m})$，
其中 $\operatorname{LSE} = m + \log s$。

## 解題思路

1. **`sampleLosses`**（≤ 1024 個區塊 × 8 個 warp）。每個樣本使用
   **一個 warp**（以網格跨步方式走訪各列）：
   - Lane $\ell$ 依序讀取該列的 logits
     $\ell, \ell+32, \dots$（合併存取），並將每個值折疊進自己的
     $(m, s)$ 配對。
   - 5 步 `__shfl_xor_sync` 蝶形操作合併 32 個配對。XOR 蝶形完成後，
     *每個* lane 都持有完整結果。
   - Lane 0 計算 $\ell_j = m + \log s - z_{j,y_j}$，
     並加到 float64 累計值。
   - 區塊中的各 warp 累計值會透過共享記憶體合併，
     每個區塊產生一個 float64 部分結果。
2. **`finalMean`**（1 個區塊）。以 float64 加總各部分結果，再除以 $N$。

每列一個 warp 很適合 $C \le 1000$：一列最多只需每個 lane 群組執行
32 次合併存取，也不需要共享記憶體來進行列歸約。

## 成本分析

$$
Q \approx 4NC + 4N, \qquad W \approx NC\,(\text{1 exp} + 3\ \text{flops}), \qquad T_{\min} \approx \frac{4NC}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 位元組數：每個 logit 一次，另加標籤 |
| $W$ | 工作量，主要是每個 logit 一次 `expf`（線上合併也會重新縮放 $s$） |
| $\beta$ | DRAM 頻寬 |

當 $N = 10^4$、$C = 1000$ 時，logits 共 40 MB，所需頻寬時間約
20 µs。每個 logit 的 `expf`（各約 20 條指令）使其接近平衡點，
但在大型 GPU 上仍大致受限於記憶體。

## 常見陷阱

- **直接計算 log-sum-exp。** $e^{10}$ 在此沒有問題，但相同程式碼在
  logits > 88 時會溢位。一律要先減去最大值。
- **平均值的精度。** 以 float32 加總 $10^4$ 個量級約為 7 的損失，
  會損失約 4 位數。使用 float64 的區塊部分和與最終總和可避免此問題。
- **標籤索引。** $z_{j,y_j}$ 每列只由 lane 0 讀取一次。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md)
以 `1e-5` 容許誤差通過，包括 $C = 2$ 與 $C = 1000$。

## 延伸閱讀

- [Softmax](../005-softmax/)、Tensara
  [Log-Softmax](../../tensara/log-softmax/)、
  [KL 散度](../../tensara/kl-loss/)、[DPO 損失](../108-dpo-sequence-loss/)。
