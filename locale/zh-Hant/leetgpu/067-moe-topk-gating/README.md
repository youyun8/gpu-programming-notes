---
title: MoE Top-K 閘控
platform: LeetGPU
upstream: medium/67_moe_topk_gating
url: https://leetgpu.com/challenges/moe-top-k-gating
difficulty: medium
tags: [moe, top-k, softmax, warp-intrinsics, llm]
status: solved
---

# MoE Top-K 閘控

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/moe-top-k-gating)

## 題意

混合專家層的路由器：對 $M$ 個 token 中的每一個，選出最大的 $k$ 個專家
logit；專家總數為 $E$（降冪排列；相同值時依 `torch.topk` 的行為選擇較小索引），
再對這 $k$ 個值做 softmax，得到混合權重
（$M \le 10^4$、$E \le 256$、$k \le E$；基準測試為 $M = 1024$、$k = 2$；
容許誤差 `1e-5`）。輸出為 `topk_indices` 與 `topk_weights`，兩者形狀皆為
$M \times k$。

## 圖解

![MoE 閘控：為每個 token 挑出 k 個最佳專家，只對它們的 logits 做 softmax](figure.svg)

對單一 token，最大的兩個 logits（綠色）決定所選專家。專家 1 與 4 同為 2.1，依規則由較小的索引優先；softmax 只在被選中的 logits 上計算。

## 數學表述

$$
(j_0, \dots, j_{k-1}) = \operatorname{TopK}(\mathbf z, k), \qquad z_{j_0} \ge z_{j_1} \ge \dots \ge z_{j_{k-1}}
$$

$$
w_t = \frac{e^{z_{j_t} - z_{j_0}}}{\sum_{u=0}^{k-1} e^{z_{j_u} - z_{j_0}}}, \qquad 0 \le t < k
$$

| 符號 | 意義 |
|---|---|
| $M$ | Token 數 |
| $E$ | 專家數 |
| $k$ | 每個 token 選出的專家數 |
| $\mathbf z$ | 一列 logit（長度為 $E$） |
| $j_t$ | 第 $t$ 個選出專家的索引；logit 相等時，較小索引排在前面 |
| $w_t$ | 專家 $j_t$ 的混合權重；$\sum_t w_t = 1$ |
| $z_{j_0}$ | 最大 logit，作為 softmax 位移 |

接著，MoE 層中該 token 的輸出為
$\sum_t w_t\,\operatorname{Expert}_{j_t}(\mathbf x)$。只有 $k$ 個專家會執行，
而專家總數為 $E$，因此 MoE 處理每個 token 的成本很低。

## 解題思路

**每個 token 使用一個 warp。** $E \le 256$ 表示每個 lane 最多處理 8 個 logit，
全都保留在暫存器中：

1. Lane $\ell$ 將 $z_{\ell}, z_{\ell+32}, \dots$ 載入 `vals[8]`
   （合併存取），以 $-\infty$ 填補超出 $E$ 的位置。
2. **進行 $k$ 輪 warp arg-max。** 每個 lane 找出自己尚未使用的最佳候選項目
   （以 `used` 位元遮罩標記已選項目）。接著以 5 步
   `__shfl_xor_sync` 蝶形歸約 `(value, index)` 配對，規則為
   「較大的值勝出；值相等時，較小的索引勝出」。蝶形歸約完成後，
   每個 lane 都持有勝出者。擁有該項目的 lane
   （`index % 32 == lane`）會設定對應的 `used` 位元。
3. 第一個勝出值就是最大值，並作為位移。每一輪都由 lane 0 寫入
   $e^{z_{j_t} - z_{j_0}}$ 與 $j_t$，同時累加總和。最後將 $k$ 個權重
   重新縮放 $1/\text{sum}$。

此方法不需要排序或共享記憶體，全部 $k$ 輪都在暫存器中完成。
當 $k = 2$ 時，每個 token 只需兩次 warp 歸約。

## 成本分析

$$
W \approx M\,k\,(8 + 5\cdot 3), \qquad Q = 4ME + 8Mk\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 運算量：每輪掃描 8 個暫存器，加上 5 個蝶形步驟（各含 2 次 shuffle 與 1 次比較） |
| $Q$ | 位元組數：讀取 logit，寫入權重與索引 |

基準測試 $M = 1024$、$E \le 256$ 時約為 1 MB，因此受啟動成本限制。
當 $k$ 很大（最高為 $E$）時，$k$ 輪的成本成為 $O(kE)$。
此時對 256 個值進行 bitonic sort 會更合適，但路由器通常使用 $k \le 8$。

## 常見陷阱

- **同值時的選擇規則。** `torch.topk` 在 CPU 參考路徑上，對相同值會先回傳
  較小索引。蝶形歸約在值相等時會比較索引。測試使用小整數值時，
  重複 logit 很常見。
- **Softmax 只對選出的 $k$ 個值執行**，不是對全部 $E$ 個值。
- **逐 warp 處理每列的核心中提早 `return`**，只有在整個 warp 共用同一個
  `row` 時才安全。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆通過，
包括 $k = E$（完整排序）、$E = 1$ 與含重複 logit 的列。

## 延伸閱讀

- [Top-K 選擇](../029-top-k-selection/)、[Softmax](../005-softmax/)、
  [SwiGLU MLP 區塊](../084-swiglu-mlp-block/)（專家執行的內容）。
