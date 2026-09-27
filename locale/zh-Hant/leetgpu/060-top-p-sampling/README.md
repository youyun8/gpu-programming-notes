---
title: Top-P 取樣
platform: LeetGPU
upstream: medium/60_top_p_sampling
url: https://leetgpu.com/challenges/top-p-sampling
difficulty: medium
tags: [sampling, softmax, selection, llm, bit-tricks]
status: solved
---

# Top-P 取樣

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/top-p-sampling)

## 題意

進行 nucleus（top-$p$）取樣：從含 $V$ 個 logit 的詞彙表選出一個 token
（$3 \le V \le 5\times10^4$，logit 值域為 $[-100, 100]$，
$0 < p \le 1$；基準測試為 $V = 5\times10^4$）。先轉成機率，保留累計機率
達到 $p$ 所需的最小高機率 token 集合，重新正規化後再使用指定種子取樣。
這是多數 LLM API 的預設解碼策略。教科書實作會排序整個詞彙表，
此解法則**不排序就找出 nucleus**。

## 圖解

![Top-p（nucleus）取樣：依機率由高到低保留 token，直到累積機率達到 p](figure.svg)

token 依機率由高至低排列。p = 0.7 時，前四個 token（綠色）的累積機率 0.77 首次達到 p，因此被保留；重新正規化後再從中取樣。

## 數學表述

$$
\pi_t = \frac{e^{z_t - m}}{\sum_{u} e^{z_u - m}}, \qquad
\pi_{(0)} \ge \pi_{(1)} \ge \dots, \qquad
c = \min\Bigl\{ c' : \sum_{r=0}^{c'} \pi_{(r)} \ge p \Bigr\}, \qquad
\mathcal N = \{(0), \dots, (c)\}
$$

$$
\Pr[\text{sample} = t] = \frac{\pi_t}{\sum_{u\in\mathcal N}\pi_u}\ \ \text{for } t\in\mathcal N, \qquad 0 \text{ otherwise}
$$

| 符號 | 意義 |
|---|---|
| $V$ | 詞彙表大小 |
| $z_t$ | token $t$ 的 logit |
| $m$ | $\max_t z_t$（softmax 穩定項） |
| $\pi_t$ | token $t$ 的 softmax 機率 |
| $\pi_{(r)}$ | 第 $r$ 大的機率（降冪次序統計量） |
| $c$ | 截止排名：參考實作中的 `searchsorted(cumsum, p)` |
| $\mathcal N$ | Nucleus：前 $c + 1$ 個 token |
| $p$ | Nucleus 機率質量門檻 |

### 以門檻表示 Nucleus

因為 nucleus 是一個*最高值集合*，所以它等於 $\{t : \pi_t \ge T\}$，
其中門檻 $T = \pi_{(c)}$ 是能讓其上方機率質量充足的最大值：

$$
T = \max\Bigl\{ \tau : \sum_{t\,:\,\pi_t \ge \tau} \pi_t \ \ge\ p \Bigr\}
$$

| 符號 | 意義 |
|---|---|
| $T$ | Nucleus 中最低機率 token 的機率 |
| $\tau$ | 候選門檻 |

**正浮點數的比較順序與其位元模式相同**（符號位元為 0，指數位於尾數上方）。
因此可對 32 位元模式進行**逐位元二分搜尋**，從最高有效位元往下找出 $T$：
嘗試設定位元 $b$，透過區塊歸約量測 $\{\pi_t \ge \text{candidate}\}$
的機率質量；若質量仍 $\ge p$，便保留該位元。32 個步驟後，位元模式恰好就是 $T$。

### 以反向 CDF 取樣

抽取 $u \in [0, 1)$，並以索引順序找出第一個使 nucleus 累計機率超過
$u \cdot \sum_{\mathcal N}\pi$ 的 token。Nucleus 採用任何固定順序，
都能得到正確分布。

## 解題思路

所有工作都在**含 1024 個執行緒的單一區塊**中執行；$V \le 50\,000$
代表每個執行緒約處理 50 個元素：

1. **Softmax 統計量。** 先對 $z$ 做區塊最大值歸約，再對 $e^{z - m}$
   做區塊總和歸約，以得到 $1/\sum$。
2. **搜尋門檻。** 進行 32 次迭代，每次都以網格步幅重新計算 $\pi_t$
   （比儲存 50k 個 float 更便宜），再做區塊總和。
3. **Nucleus 質量。** 計算 $\sum_{\pi_t \ge T}\pi_t$。
4. **亂數。** 對種子執行 SplitMix64，得到 64 位元雜湊；取最高 24 位元，
   形成 $[0, 1)$ 內的 float，再乘以 nucleus 質量。
5. **反向 CDF** 依索引順序每批處理 1024 個 token。區塊內含式掃描
   （warp shuffle 加上 warp 總和，再加 carry）會找出執行中質量首次超過
   目標的索引，透過共享「hit」欄位的 `atomicMin` 記錄並提早停止。
   若四捨五入使目標略高於總和，則後備處理會回傳 nucleus 最後一個 token。

## 成本分析

$$
W \approx V\,(c_{\exp} + 1)\cdot(2 + 32 + 1 + 1), \qquad Q = 4V \ \text{bytes (L1/L2-resident after the first pass)}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 工作量：約 36 趟中的每一趟都會為每個 token 重新計算一次 `expf` |
| $c_{\exp}$ | 一次精確 `expf` 的成本 |
| $Q$ | Logit 只從 DRAM 讀取一次（200 KB），之後都命中快取 |

總計約 $1.8$M 次指數運算，在單一 SM 上需數十微秒。完整排序 50k 個鍵會更久，
也需要數個核心。

## 常見陷阱

- **與參考實作完全相同的 token。** 參考實作在設定 PyTorch 產生器種子後，
  使用 `torch.multinomial` 取樣（GPU 上是 Philox，CPU 上是 mt19937）。
  CUDA C++ 無法重現該亂數串流，因此只有在 $\lvert\mathcal N\rvert = 1$
  時，正確取樣器才會回傳*相同 token*。所以本機執行器檢查 token
  是否**位於 nucleus 中**，這才是正確取樣器能保證的性質。平台自身的評分器
  可能不同。
- **$T$ 上的同值。** 所有 $\pi_t = T$ 的 token 都會納入。
  以排序實作時可能只納入其中一部分。對連續輸入而言，這種差異的測度為零。
- **`>=` 與 `>`。** `searchsorted(right=False) + 1` 對應「累計和 ≥ p
  的最短前綴」，也就是門檻定義中的 ≥。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例都會檢查
nucleus 成員資格與恰好 $\lvert\mathcal N\rvert = 1$ 的案例。Nucleus
本身（集合）也針對數千個隨機詞彙表，與以排序為基礎的 Python 實作比較過。

## 延伸閱讀

- [Top-K 選擇](../029-top-k-selection/)（對位元模式進行 radix select）、
  [Softmax](../005-softmax/)、[推測式解碼驗證](../087-speculative-decoding-verification/)。
