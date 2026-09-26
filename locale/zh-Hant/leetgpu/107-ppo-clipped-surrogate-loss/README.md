---
title: PPO 截斷代理損失
platform: LeetGPU
upstream: medium/107_ppo_clipped_surrogate_loss
url: https://leetgpu.com/challenges/ppo-clipped-surrogate-loss
difficulty: medium
tags: [reduction, rl, loss, rlhf]
status: solved
---

# PPO 截斷代理損失

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/ppo-clipped-surrogate-loss)

## 問題

**PPO**（近端策略最佳化）的截斷代理損失，用於 LLM 的 RLHF。給定每個權杖的優勢值，以及目前策略與舊策略下已取樣權杖的對數機率（皆為 $B\times S$ float32），傳回純量損失（容許誤差 `1e-4`）。

## 公式

$$
r_{b,s} = \exp\bigl(\log\pi_{b,s} - \log\pi^{\text{old}}_{b,s}\bigr), \qquad
\hat r_{b,s} = \operatorname{clip}(r_{b,s},\ 1-\varepsilon,\ 1+\varepsilon)
$$

$$
\mathcal L = -\frac{1}{BS}\sum_{b=0}^{B-1}\sum_{s=0}^{S-1}\min\bigl(r_{b,s}A_{b,s},\ \hat r_{b,s}A_{b,s}\bigr)
$$

| 符號 | 意義 |
|---|---|
| $B,\ S$ | 批次大小與回應長度 |
| $\log\pi_{b,s}$ | 目前策略下權杖 $(b, s)$ 的對數機率 |
| $\log\pi^{\text{old}}_{b,s}$ | 產生資料的策略下，同一權杖的對數機率 |
| $r_{b,s}$ | 重要性比率 $\pi/\pi^{\text{old}}$ |
| $\varepsilon$ | 截斷範圍（`clip_eps`，例如 0.2） |
| $\hat r$ | 截斷至信賴區域 $[1-\varepsilon, 1+\varepsilon]$ 的比率 |
| $A_{b,s}$ | 優勢估計值（請參閱 [GAE](../110-gae-reverse-scan/)） |
| $\mathcal L$ | 損失：代理目標的負平均值（PPO 會將代理目標*最大化*） |

**為何取最小值。** 當 $A > 0$ 時，目標在 $r$ 超過 $1+\varepsilon$ 後就不再獎勵其增加。當 $A < 0$ 時，目標在比率低於 $1-\varepsilon$ 後就不再獎勵其降低。因此一次更新無法讓策略偏離舊策略太遠。

## 方法

每個權杖的對應運算都是逐元素的，因此可融合到[歸約](../004-reduction/)的兩階段歸約中：

1. **`surrogateSums`**：以網格跨距走訪 $BS$ 個權杖。以 float32 計算 `expf`、限制範圍，並對兩個乘積執行 `fminf`，接著累加，再以 float64 區塊歸約產生各區塊的部分總和。
2. **`finalize`**：以 float64 加總部分結果，再計算 $-\text{sum}/(BS)$。

從對數機率的*差值*計算比率（一次 `expf`）在數值上較合理。若將兩個機率相除，長序列會發生下溢。

## 成本分析

$$
Q = 12BS\ \text{bytes}, \qquad W \approx BS\,(c_{\exp} + 6)
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（三個輸入陣列各讀取一次） |
| $W$ | 每個權杖：一次 `expf`、範圍限制、兩次乘法、取最小值、相加 |

在任何實際大小下，核心都受記憶體頻寬限制。在訓練時，它會與產生 $\log\pi$ 的 log-softmax gather 融合。

## 常見問題

- **正負號。** 損失是**負的**平均值。
- **範圍限制順序。** 對有效的 $\varepsilon$ 而言，`fminf(fmaxf(r, 1-ε), 1+ε)` 與 `torch.clamp` 相同。
- **平均值精度。** 與此處所有歸約相同，部分總和使用 float64。

## 驗證

所有 LeetGPU 測試案例都以 `1e-4` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，涵蓋正負優勢，以及截斷範圍兩側的比率。

## 相關內容

- [GRPO 代理損失](../109-grpo-surrogate-loss/)、[DPO 損失](../108-dpo-sequence-loss/)、[GAE 反向掃描](../110-gae-reverse-scan/)、[數值截斷](../062-value-clipping/)。
