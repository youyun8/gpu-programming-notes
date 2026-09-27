---
title: Softmax 注意力反向傳播
platform: LeetGPU
upstream: medium/111_softmax_attention_backward
url: https://leetgpu.com/challenges/softmax-attention-backward
difficulty: medium
tags: [attention, backward, flash-attention, autograd]
status: solved
---

# Softmax 注意力反向傳播

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/softmax-attention-backward)

## 題意

單頭 softmax 注意力的**反向傳播**。給定 $Q$（$M\times d$）、$K, V$（$N\times d$）與上游梯度 $dO$（$M\times d$），計算 $dQ$、$dK$ 與 $dV$（$M, N \le 10^5$、$d \le 128$；效能測試為 $M = 8192$、$N = 4096$、$d = 128$；容許誤差 `1e-4`）。訓練 Transformer 時，這個核心和前向傳播一樣重要。它也必須像前向傳播一樣，避免儲存 $M\times N$ 機率矩陣：在效能測試大小下需 128 MB，在上限則需 40 GB。

## 圖解

![不儲存 P 的注意力反向傳播：由 Q、K 與 LSE 逐分塊重新計算](figure.svg)

方框顯示各梯度之間的相依關係：先由 Q、K 與保存的 log-sum-exp 重建 P，再逐分塊依序算出 dV、dP、dS、dQ 與 dK。

## 數學表述

前向傳播：$S = QK^{\mathsf T}/\sqrt d$、$P = \operatorname{softmax}_{\text{row}}(S)$、$O = PV$。反向傳播：

$$
dV = P^{\mathsf T}dO, \qquad dP = dO\,V^{\mathsf T}, \qquad
dS_{ij} = P_{ij}\bigl(dP_{ij} - D_i\bigr),\ \ D_i = \sum_k P_{ik}\,dP_{ik}, \qquad
dQ = \frac{dS\,K}{\sqrt d}, \qquad dK = \frac{dS^{\mathsf T}Q}{\sqrt d}
$$

$$
D_i = \sum_k P_{ik}\,(dO_i\cdot V_k) = dO_i\cdot\Bigl(\sum_k P_{ik}V_k\Bigr) = dO_i\cdot O_i, \qquad
P_{ij} = e^{S_{ij} - L_i},\ \ L_i = \log\sum_j e^{S_{ij}}
$$

| 符號 | 意義 |
|---|---|
| $M,\ N,\ d$ | 查詢數、鍵數、頭維度 |
| $S,\ P$ | 縮放後的分數與注意力機率（$M\times N$，不會儲存） |
| $O$ | 前向傳播輸出 |
| $dO$ | 損失相對於 $O$ 的上游梯度 |
| $dP$ | 相對於 $P$ 的梯度：$dP_{ij} = dO_i\cdot V_j$ |
| $dS$ | 相對於分數的梯度（逐列套用 softmax Jacobian） |
| $D_i$ | 資料列修正項；等於 $dO_i\cdot O_i$ |
| $L_i$ | 資料列 log-sum-exp；只需 $S_{ij}$ 即可重算 $P_{ij}$ |
| $dQ,\ dK,\ dV$ | 輸出 |

**Softmax Jacobian。** 對 $\mathbf p = \operatorname{softmax}(\mathbf s)$，有 $\partial p_j/\partial s_k = p_j(\delta_{jk} - p_k)$。因此 $ds_k = \sum_j dp_j\,p_j(\delta_{jk} - p_k) = p_k(dp_k - \sum_j p_j dp_j)$，也就是上述 $dS$ 公式。

## 解題思路

採用 FlashAttention-2 的反向傳播策略：每個查詢資料列只儲存兩個純量，並即時重算 $P$。

1. **`rowStats`**（查詢平行）：使用線上 softmax 重新執行前向傳播（每個查詢資料列一個 warp，每次處理 32 個鍵的分塊）。儲存 $L_i = m + \log\ell$ 與 $D_i = dO_i\cdot O_i$，其中 $O_i$ 取自累加器。
2. **`gradQ`**（查詢平行）：對每個鍵分塊，lane $\ell$ 會為鍵 $j = j_0 + \ell$ 計算 $S_{ij}$ 與 $dP_{ij}$，再計算 $dS_{ij} = e^{S_{ij} - L_i}(dP_{ij} - D_i)/\sqrt d$。每個 $dS_{ij}$ 都會 shuffle 給所有 lane，讓它們將 $dQ_i \mathrel{+}= dS_{ij}K_j$ 累加至各自負責的 $d/32$ 欄。
3. **`gradKV`**（鍵平行）：一個 warp 負責鍵資料列 $j$，並串流處理**查詢**分塊（連同其 $L_i$、$D_i$）。lane $\ell$ 會計算與查詢 $i_0 + \ell$ 的配對，各 lane 再累加 $dV_j \mathrel{+}= P_{ij}\,dO_i$ 與 $dK_j \mathrel{+}= dS_{ij}\,Q_i$。

**不使用原子操作。** $dQ$ 資料列取決於所有鍵，而 $dK$/$dV$ 資料列取決於所有查詢。對 $dQ$ 使用查詢平行走訪，對 $dK$/$dV$ 使用鍵平行走訪，即可讓每個輸出資料列恰好由一個 warp 負責。代價是重複計算兩次 $S$ 與 $dP$，這是常見的取捨（FlashAttention-2 則在單次走訪中對 $dQ$ 使用原子操作）。

## 成本分析

$$
W \approx \underbrace{4MNd}_{\text{rowStats}} + \underbrace{6MNd}_{\text{gradQ}} + \underbrace{8MNd}_{\text{gradKV}} = 18MNd, \qquad
\text{extra memory} = 8M\ \text{bytes}\ (L, D)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數：每次走訪都會為所有配對重算 $QK^{\mathsf T}$（需要時也重算 $dO V^{\mathsf T}$） |
| 額外記憶體 | 唯一的中間狀態：每個查詢資料列兩個浮點數 |

效能測試：$W \approx 7.7\times10^{10}$ FLOP。此核心受運算能力限制，成本約為前向傳播的 2.5 倍，符合一般「反向傳播約為前向傳播 2–3 倍」的經驗法則。

## 常見陷阱

- **縮放位置。** $S$ 使用 $1/\sqrt d$（預先套用到共享緩衝區中的 $Q$，或在 `gradKV` 中相乘），而 $dQ$ 與 $dK$ 都各自帶有另一個 $1/\sqrt d$。這兩個因子很容易混淆。
- **從 $O_i$ 取得 $D_i$。** 若直接計算 $\sum_k P_{ik}dP_{ik}$，就需要完整走訪一次所有鍵；$dO_i\cdot O_i$ 是相同的數值。
- 超過 $N$（或 $M$）的**遮罩 lane**必須貢獻零權重，而非 NaN。

## 驗證

所有 LeetGPU 測試案例都以 `1e-4` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，且梯度已使用隨機輸入與 `torch.autograd` 交叉比對。

## 延伸閱讀

- [Softmax 注意力](../006-softmax-attention/)（前向傳播）、[多頭注意力](../012-multi-head-attention/)。
