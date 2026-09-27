---
title: 推測式解碼驗證
platform: LeetGPU
upstream: medium/87_speculative_decoding_verification
url: https://leetgpu.com/challenges/speculative-decoding-verification
difficulty: medium
tags: [sampling, llm, speculative-decoding, scan]
status: solved
---

# 推測式解碼驗證

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/speculative-decoding-verification)

## 題意

實作**推測式解碼**的接受步驟。小型草稿模型為每個序列提出 $T$ 個 token；
大型目標模型則在一次前向傳播中為它們全部計分。對 $B$ 個序列中的每一個，
依標準拒絕規則由左至右接受草稿 token。第一次拒絕時，從殘差分布重新取樣
並停止。若全部 $T$ 個 token 都被接受，則從目標分布抽取一個額外 token。
輸出形狀為 $(B, T+1)$ 的 token id，並以零補齊。給定相同的均勻樣本時，
結果必須完全相符。

## 圖解

![推測式解碼驗證：由左至右接受草稿 token，遇到第一次拒絕就停止](figure.svg)

草稿 token t0 與 t1 通過（u < α）；t2 失敗，因此從殘差分佈重新取樣，t3 不再檢查。輸出列為 t0、t1、r，其餘補零。

## 數學表述

在草稿位置 $i$，草稿 token 為 $t_i$：

$$
\alpha_i = \min\!\Bigl(1,\ \frac{q_i(t_i)}{p_i(t_i)}\Bigr), \qquad \text{accept } t_i \iff u_i < \alpha_i
$$

若第一次拒絕發生在位置 $i$，則從**殘差分布**取樣：

$$
r_i(v) = \frac{\max\bigl(0,\ q_i(v) - p_i(v)\bigr)}{\sum_{v'}\max\bigl(0,\ q_i(v') - p_i(v')\bigr)}\quad (\text{uniform } 1/V \text{ if the sum is } 0)
$$

若全部 $T$ 個 token 都被接受，則從 $q_{T-1}$ 取樣一個額外 token。
取樣使用反 CDF：

$$
\operatorname{sample}(\pi, u) = \min\Bigl\{\, v : \sum_{v' \le v} \pi(v') \ge u \,\Bigr\}\ \ (\text{clamped to } V-1)
$$

| 符號 | 意義 |
|---|---|
| $B,\ T,\ V$ | 批次大小、草稿 token 數、詞彙表大小 |
| $t_i$ | 位置 $i$ 的草稿 token |
| $p_i(v)$ | 草稿模型在位置 $i$ 對 token $v$ 的機率 |
| $q_i(v)$ | 目標模型的機率 |
| $u_i$ | 位置 $i$ 接受測試所用的均勻樣本 |
| $\alpha_i$ | 接受機率 |
| $r_i$ | 拒絕後使用的殘差分布 |
| $u_T$ | 用於重新取樣或額外 token 的額外均勻樣本（索引 $T$） |
| Sample$(\pi, u)$ | 反 CDF 抽樣（`torch.searchsorted(cumsum(π), u)`） |

**為何結果精確。** 以機率 $\min(1, q/p)$ 接受，否則從 $r$ 取樣，會產生
完全遵循目標分布 $q$ 的 token（Leviathan 等人、Chen 等人，2023）。
大型模型的輸出分布得以保留，同時每次前向傳播可驗證 $T$ 個 token。

## 解題思路

**每個序列使用一個含 1024 個執行緒的區塊：**

1. 將輸出列清為零。
2. 循序走訪 $i = 0..T-1$（鏈會在第一次拒絕時停止）：
   - 接受測試是純量計算（每個執行緒都計算出相同結果，因此控制流程一致）；
   - 接受 → 執行緒 0 寫入 $t_i$；
   - 拒絕 → 以**區塊歸約**計算整個 $V$ 上的
     $\sum\max(0, q - p)$；接著 `inverseCdf` 對權重執行分塊的**區塊掃描**
     （warp `__shfl_up_sync` 掃描、warp 總值，以及跨越每 1024 個詞彙項目
     區塊的進位），並透過 `atomicMin` 找出第一個累加總和 $\ge u_T$ 的
     $v$，且可提早結束。寫入該值後執行 `return`。
3. 若全部 $T$ 個 token 都被接受，則對 $q_{T-1}$ 執行相同的反 CDF，
   以取得額外 token。

權重函式以 lambda 傳給 `inverseCdf`，因此相同的掃描程式碼可用於殘差分布、
均勻分布與目標分布。

## 成本分析

$$
W \le B\,\bigl(T + 3V\bigr), \qquad Q \le 4B\,(T + 2V\cdot 2)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 工作量：$T$ 次純量測試，加上每個序列至多一次對 $V$ 的殘差歸約與一次掃描 |
| $Q$ | 位元組數：每個序列最多完整讀取兩列詞彙表的 $p$ 與 $q$（張量為 $B\times T\times V$，但只會讀取實際走訪的列） |

真實 LLM 的 $V$ 約為 $3\times10^4$–$1.5\times10^5$，因此掃描佔主要成本。
每個序列使用單一區塊，可保持實作簡單且不需跨區塊同步。

## 常見陷阱

- **使用哪個均勻樣本。** 接受測試使用 $u_0..u_{T-1}$；重新取樣與額外
  token 都使用 $u_T$，與參考實作相同。
- **除以 $p(t_i) = 0$** 會得到 $\infty$，因此 $\alpha = 1$（一律接受），
  符合 Python 的浮點語意。
- **`searchsorted` 語意**：找出 cumsum **$\ge$** $u$ 的第一個索引
  （左側插入點），並限制在 $V-1$，避免 CDF 尾端的捨入問題。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 token
完全相等的方式通過，包括全部接受（額外 token 路徑）、立即拒絕，以及
零殘差（退回均勻分布）案例。

## 延伸閱讀

- [Top-p 取樣](../060-top-p-sampling/)、[前綴和](../016-prefix-sum/)、
  [加法器 Transformer](../076-adder-transformer/)（自迴歸解碼）。
