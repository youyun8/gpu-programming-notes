---
title: Softmax 注意力
platform: LeetGPU
upstream: medium/6_softmax_attention
url: https://leetgpu.com/challenges/softmax-attention
difficulty: medium
tags: [attention, flash-attention, online-softmax, shared-memory, warp-shuffle]
status: solved
---

# Softmax 注意力

**平台：** LeetGPU · **難度：** medium · [題目說明](https://leetgpu.com/challenges/softmax-attention)

## 問題

以 float32 計算單頭縮放點積注意力：
$Q \in \mathbb{R}^{M\times d}$，$K, V \in \mathbb{R}^{N \times d}$，
輸出 $\in \mathbb{R}^{M\times d}$（$1 \le M, N \le 10^5$，
$1 \le d \le 128$；基準 $M = 512$、$N = 256$）。容許誤差為 `1e-4`。
當 $M = N = 10^5$ 時，僅分數矩陣就占 40 GB，因此**絕不能將它具體化**。
這正是 FlashAttention 的核心概念。

## 公式

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}_{\text{row}}\!\left(\frac{QK^{\mathsf T}}{\sqrt d}\right) V
$$

針對每個查詢列 $r$：

$$
s_{rj} = \frac{1}{\sqrt d}\sum_{c=0}^{d-1} Q_{rc} K_{jc}, \qquad
p_{rj} = \frac{e^{s_{rj} - m_r}}{\sum_{j'} e^{s_{rj'} - m_r}}, \qquad
O_{rc} = \sum_{j=0}^{N-1} p_{rj} V_{jc}
$$

| 符號 | 意義 |
|---|---|
| $M$ | 查詢數（$Q$ 與輸出的列數） |
| $N$ | 鍵／值數量（$K$、$V$ 的列數） |
| $d$ | 頭維度（$Q$、$K$、$V$ 與輸出的欄數），$\le 128$ |
| $r,\ j,\ c$ | 查詢索引、鍵索引、特徵（欄）索引 |
| $s_{rj}$ | 查詢 $r$ 對鍵 $j$ 的縮放注意力分數 |
| $m_r$ | $\max_j s_{rj}$，用於維持數值穩定性 |
| $p_{rj}$ | 注意力權重；每列總和為 1 |
| $O_{rc}$ | 輸出元素（`output[r*d + c]`） |

### 對鍵分塊執行線上 Softmax

以每塊 32 個鍵的 $\mathcal{T}_1, \mathcal{T}_2, \dots$ 依序處理。
每完成一塊，就更新目前最大值 $m$、分母 $\ell$ 與未正規化輸出向量
$\mathbf{a} \in \mathbb{R}^d$：

$$
m' = \max\!\Bigl(m,\ \max_{j\in\mathcal T} s_{rj}\Bigr), \quad
\alpha = e^{m - m'}, \quad
\ell' = \alpha\,\ell + \sum_{j\in\mathcal T} e^{s_{rj}-m'}, \quad
\mathbf a' = \alpha\,\mathbf a + \sum_{j\in\mathcal T} e^{s_{rj}-m'}\, V_{j,:}
$$

最後 $O_{r,:} = \mathbf a / \ell$。

| 符號 | 意義 |
|---|---|
| $\mathcal T$ | 目前的鍵索引分塊（32 個連續鍵） |
| $m,\ m'$ | 分塊前後的目前最大分數（起始為 $-\infty$） |
| $\alpha$ | 將目前所有累積值重新縮放至新最大值的修正因子 |
| $\ell,\ \ell'$ | 目前 softmax 分母（起始為 0） |
| $\mathbf a,\ \mathbf a'$ | 長度為 $d$ 的未正規化輸出列（起始為 0） |
| $V_{j,:}$ | $V$ 的第 $j$ 列 |

這是 [Softmax](../005-softmax/) 的配對合併，加入同樣由 $\alpha$
重新縮放的向量承載值 $\mathbf a$。

## 方法

### 工作對應

| 層級 | 責任 |
|---|---|
| 區塊（128 個執行緒 = 4 個 warp） | 4 個連續查詢列；為四列各暫存一次 K/V 分塊 |
| Warp | 一個查詢列 $r$ |
| Lane $\ell$ | 計算分塊中的鍵 $\ell$；負責輸出欄 $\ell, \ell+32, \ell+64, \ell+96$ |

### 每個分塊

1. **暫存。** 先執行 `__syncthreads()`，再由區塊將 $K$ 與 $V$ 的
   32 列複製到共享記憶體（`k_tile`、`v_tile`），超過 $N$ 的部分補 0；
   接著再執行一次 `__syncthreads()`。
2. **分數：每個 lane 一個鍵。** Lane $\ell$ 計算查詢
   （預先乘上 $1/\sqrt d$ 並保存在共享記憶體）與鍵 $\ell$ 的完整
   $d$ 維點積。整個分塊只需一次 `warpMax` 和一次 `warpSum`
   （各 5 次 shuffle），而不是每個鍵各做一次 5-shuffle 歸約。
3. **重新縮放。** 套用上述公式：$\alpha$ 乘上 $\ell$ 與 4 個累加器暫存器。
4. **累加 $PV$。** 對分塊中的每個鍵 $j$，以
   `__shfl_sync(…, p, j)` 從 lane $j$ 廣播 $p_j$。接著每個 lane
   對自己負責的 4 欄執行 $a_c \mathrel{+}= p_j V_{jc}$。

### 避免 bank 衝突

步驟 2 中，lane $\ell$ 讀取 `k_tile[ℓ][c]`，而其他 lane 也讀取相同
$c$，即分塊的一欄。若列間距為 128 個浮點數，所有 lane 都會命中
bank $c \bmod 32$。因此間距設為 **129**（奇數），將 32 個 lane
分散到 32 個 bank。

## 成本分析

$$
W = 4MNd, \qquad
Q_{\text{DRAM}} \approx 4\left(Md + \left\lceil\frac{M}{4}\right\rceil\cdot 2Nd + Md\right), \qquad
\text{mem}_{\text{scores}} = 0
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP：$QK^{\mathsf T}$ 為 $2MNd$，$PV$ 也為 $2MNd$ |
| $Q_{\text{DRAM}}$ | 位元組數：讀取 $Q$、每個區塊重讀全部 $K$ 與 $V$，再寫入輸出 |
| $\lceil M/4 \rceil$ | 區塊數（每個區塊 4 個查詢列） |
| $\text{mem}_{\text{scores}}$ | $M \times N$ 分數矩陣的額外記憶體；從未儲存 |

基準大小下（$M = 512$、$N = 256$、$d \le 128$），
$W \approx 67$ MFLOP，而 $K$/$V$（256 KB）會留在 L2。
讓 4 列共用每個分塊，可將共享記憶體暫存流量減少 4 倍。
更大的列區塊（例如 FlashAttention-2 以張量核心處理 64 個查詢）
能進一步提高重用；請見[多頭注意力](../012-multi-head-attention/)。

## 常見問題

- **非作用中 warp 的屏障。** 列索引 $\ge M$ 的 warp 仍須執行分塊迴圈中
  每個 `__syncthreads()`。它們只跳過最後儲存；提早返回會讓區塊死鎖。
- **不完整的最後分塊。** 超過 $N$ 的 lane 取得分數 $-\infty$ 與
  $p = 0$，而 $PV$ 迴圈只處理 `tile_keys` 個有效鍵。
- **縮放順序。** 預先將 $Q$ 乘一次 $1/\sqrt d$，不要對每個分數都乘。
- **初始狀態。** $m = -\text{FLT\_MAX}$、$\ell = 0$。第一個分塊的
  $\alpha = e^{-\text{FLT\_MAX} - m'}$ 會乾淨地反向溢位成 0。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上通過，
包括 $d = 1$、$d = 128$、$N < 32$ 及 $M$ 非 4 倍數。
執行時使用 `--reverse` 排程，確認暫存屏障足夠。

## 相關內容

- [多頭注意力](../012-multi-head-attention/)、[因果注意力](../053-casual-attention/)、
  [滑動視窗注意力](../059-sliding-window-attn/)、[GQA](../080-grouped-query-attention/)。
- Tensara [縮放點積注意力](../../tensara/scaled-dot-attention/)。
