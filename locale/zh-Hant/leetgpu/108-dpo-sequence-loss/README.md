---
title: DPO 序列損失
platform: LeetGPU
upstream: medium/108_dpo_sequence_loss
url: https://leetgpu.com/challenges/dpo-sequence-loss
difficulty: medium
tags: [reduction, rl, loss, rlhf, numerical-stability]
status: solved
---

# DPO 序列損失

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/dpo-sequence-loss)

## 問題

**直接偏好最佳化**（Direct Preference Optimisation）損失。$B$ 個偏好配對各有四個加總後的序列對數機率：可訓練策略與凍結參考模型下，獲選和被拒回應各一個。傳回溫度 $\beta$ 下的平均損失（容許誤差 `1e-4`）。DPO 可依人類偏好微調 LLM，而不必訓練獎勵模型或執行 RL。

## 公式

$$
z_i = \beta\Bigl[\bigl(\ell^+_i - \ell^-_i\bigr) - \bigl(\ell^{+,\text{ref}}_i - \ell^{-,\text{ref}}_i\bigr)\Bigr], \qquad
\mathcal L = \frac1B\sum_{i=0}^{B-1} -\log\sigma(z_i) = \frac1B\sum_i \operatorname{softplus}(-z_i)
$$

$$
\operatorname{softplus}(x) = \log(1 + e^{x}) = \max(x, 0) + \log\bigl(1 + e^{-\lvert x\rvert}\bigr)
$$

| 符號 | 意義 |
|---|---|
| $B$ | 偏好配對數 |
| $\ell^+_i,\ \ell^-_i$ | 策略對獲選／被拒回應的對數機率 $\log\pi_\theta(y^\pm\mid x)$ |
| $\ell^{\pm,\text{ref}}_i$ | 凍結參考策略下的相同機率 |
| $\beta$ | 溫度：偏離參考策略的強度 |
| $z_i$ | 隱含獎勵邊際：相較於參考策略，目前策略偏好 $y^+$ 的程度多了多少 |
| $\sigma$ | Logistic sigmoid |
| Softplus | $\log(1+e^x)$；右側形式不會溢位 |
| $\mathcal L$ | 平均 DPO 損失 |

**為何使用穩定形式。** 直接計算 $\log(1 + e^{x})$，在 $x > 88$ 時會溢位（得到 $\infty$），而在 $x < -17$ 時則會失去所有精度（$1 + e^{x}$ 會捨入為 1）。改寫後只需計算 $e^{-\lvert x\rvert} \le 1$，而 `log1pf` 對小引數仍能保持完整的相對精度。

## 方法

$B$ 很小（每個配對只有一個值），因此使用**單一含 1024 個執行緒的區塊**即可：以網格跨距走訪各配對，以 float32 計算 $z$ 與穩定的 softplus，以 float64 累加，再執行 warp shuffle 與共享記憶體歸約。執行緒 0 寫入 $\text{sum}/B$。

## 成本分析

$$
Q = 16B\ \text{bytes}, \qquad W \approx B\,(c_{\exp} + c_{\log} + 8)
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 位元組數：四個浮點陣列各讀取一次 |
| $W$ | 每個配對：一次 `expf`、一次 `log1pf`，以及數次加法與乘法 |

對任何實際的 $B$ 而言都只需數微秒。DPO 最昂貴的部分是計算四個序列對數機率，這會在模型的前向傳播中完成。

## 常見問題

- **直接計算 `-log(sigmoid(z))`**：`sigmoid(-100)` 會下溢為 0，因而得到 $-\log 0 = \infty$。
- **正負號。** 損失為 `softplus(-z)`；當策略比參考策略更強烈偏好獲選回應時，此值很小。

## 驗證

所有 LeetGPU 測試案例都以 `1e-4` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括邊際為 $\pm100$ 的情況。

## 相關內容

- [PPO 截斷損失](../107-ppo-clipped-surrogate-loss/)、[GRPO 損失](../109-grpo-surrogate-loss/)、[類別交叉熵](../025-categorical-cross-entropy-loss/)。Tensara [Softplus](../../tensara/soft-plus/)。
