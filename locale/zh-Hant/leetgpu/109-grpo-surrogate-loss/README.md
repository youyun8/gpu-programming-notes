---
title: GRPO 代理損失
platform: LeetGPU
upstream: medium/109_grpo_surrogate_loss
url: https://leetgpu.com/challenges/grpo-surrogate-loss
difficulty: medium
tags: [reduction, rl, loss, rlhf]
status: solved
---

# GRPO 代理損失

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/grpo-surrogate-loss)

## 題意

**GRPO**（群組相對策略最佳化，DeepSeekMath/DeepSeek-R1）目標。對 $B$ 個提示中的每一個，都會取樣並評分 $G$ 個回應。每個回應的優勢值，是其獎勵在**所屬群組內**標準化後的結果，因此不需要評論家網路。該優勢會套用到回應的所有 $S$ 個權杖，構成 PPO 風格的截斷目標，並加上對參考策略的 KL 懲罰。輸入為獎勵 $(B, G)$ 與三個對數機率張量 $(B, G, S)$。輸出為純量損失（容許誤差 `1e-4`）。

## 圖解

![GRPO：優勢值是在每組 G 個回答內標準化後的獎勵](figure.svg)

同一個提示詞的四個獎勵被標準化為優勢值（綠色列），再廣播到該回答的全部 S 個 token（長條），代入類似 PPO 的目標函數。

## 數學表述

$$
\mu_b = \frac1G\sum_g R_{b,g}, \qquad \sigma_b = \sqrt{\frac1G\sum_g (R_{b,g} - \mu_b)^2}, \qquad
A_{b,g} = \frac{R_{b,g} - \mu_b}{\sigma_b + 10^{-8}}
$$

$$
r_{b,g,s} = e^{\log\pi - \log\pi^{\text{old}}}, \qquad
d_{b,g,s} = \log\pi^{\text{ref}} - \log\pi, \qquad
K_{b,g,s} = e^{d} - d - 1
$$

$$
\mathcal L = -\frac{1}{BGS}\sum_{b,g,s}\Bigl[\min\bigl(rA_{b,g},\ \operatorname{clip}(r, 1\pm\varepsilon)A_{b,g}\bigr) - \beta K_{b,g,s}\Bigr]
$$

| 符號 | 意義 |
|---|---|
| $B,\ G,\ S$ | 提示數、每個提示的回應數（群組大小）、每個回應的權杖數 |
| $R_{b,g}$ | 對提示 $b$ 的回應 $g$ 之純量獎勵 |
| $\mu_b,\ \sigma_b$ | 群組平均值與**母體**標準差 |
| $A_{b,g}$ | 群組相對優勢，由該回應的所有 $S$ 個權杖共用 |
| $\log\pi,\ \log\pi^{\text{old}},\ \log\pi^{\text{ref}}$ | 目前策略、取樣策略與參考策略下，權杖 $(b, g, s)$ 的對數機率 |
| $r$ | 重要性比率 |
| $\varepsilon$ | 截斷範圍（`clip_eps`） |
| $d$ | 參考策略相對於目前策略的對數比率 |
| $K$ | 「$k_3$」KL 估計量 $e^d - d - 1 \ge 0$（期望值是 $\mathrm{KL}(\pi\,\Vert\,\pi^{\text{ref}})$ 的無偏估計，且一律非負） |
| $\beta$ | KL 懲罰權重 |
| $\mathcal L$ | 損失（目標的負值） |

## 解題思路

1. **`groupAdvantages`**：每個提示使用一個執行緒（$G$ 很小，例如 8–64）。它會計算 float64 的平均值與母體變異數，再將 $G$ 個優勢寫入暫存緩衝區。
2. **`tokenSums`**：以網格跨距走訪全部 $BGS$ 個權杖（64 位元索引）。優勢為 `adv[i / S]`（回應索引 = 權杖索引 / $S$）。計算比率、截斷、最小值與 $k_3$ 項；每個執行緒以 float32 累加，再透過 float64 區塊歸約寫入部分結果。
3. **`finalize`**：計算 $-\text{sum}/(BGS)$。

## 成本分析

$$
Q \approx 12BGS + 8BG\ \text{bytes}, \qquad W \approx BGS\,(2c_{\exp} + 10)
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 位元組數：三個權杖層級陣列，另加獎勵與優勢 |
| $W$ | 每個權杖：兩次 `expf`，另加約 10 次浮點運算 |

核心受記憶體頻寬限制。優勢讀取 `adv[i / S]` 會連續 $S$ 個權杖使用同一個值，因此一律會命中快取。

## 常見陷阱

- 使用**母體標準差**（`unbiased=False`），並將 $10^{-8}$ 加在 $\sigma$ 上，而不是 $\sigma^2$ 上。
- **KL 估計量。** 它是 $k_3 = e^{d} - d - 1$，其中 $d = \log\pi^{\text{ref}} - \log\pi$。$d$ 的正負號很重要。
- **常數群組**（$\sigma = 0$）會得到 $A = 0/10^{-8} = 0$，這是正確結果，而非 NaN。

## 驗證

所有 LeetGPU 測試案例都以 `1e-4` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $G = 1$（所有優勢皆為 0）與獎勵為常數的群組。

## 延伸閱讀

- [PPO 截斷損失](../107-ppo-clipped-surrogate-loss/)、[DPO 損失](../108-dpo-sequence-loss/)。
