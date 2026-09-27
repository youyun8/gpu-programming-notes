---
title: Sigmoid 線性單元
platform: LeetGPU
upstream: easy/52_silu
url: https://leetgpu.com/challenges/sigmoid-linear-unit
difficulty: easy
tags: [elementwise, activation, transcendental]
status: solved
---

# Sigmoid 線性單元

**平台：** LeetGPU · **難度：** 簡單 · [題目說明](https://leetgpu.com/challenges/sigmoid-linear-unit)

## 問題

對 $N$ 個 float32 值逐元素套用 SiLU（也稱為 *Swish-1*）
（測試中 $N \le 10^4$，基準測試中為 $5\times10^4$；輸入值域為
$[-100, 100]$；容許誤差 `1e-5`）。SiLU 是 LLaMA 與多數現代 LLM
的 SwiGLU MLP 所使用的活化函數。

## 公式

$$
\operatorname{SiLU}(x) = x\,\sigma(x) = \frac{x}{1 + e^{-x}}, \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

| 符號 | 意義 |
|---|---|
| $x$ | 輸入值 |
| $\sigma$ | Logistic sigmoid |
| $\operatorname{SiLU}(x)$ | 輸出值 |

極限為：$\operatorname{SiLU}(x) \to x$ 當 $x \to +\infty$，且
$\to 0^-$ 當 $x \to -\infty$。其最小值約為 $-0.278$，出現在
$x \approx -1.278$。

## 方法

每個元素使用一個執行緒計算 `x / (1.0f + expf(-x))`，包含一次指數運算與
一次除法，且沒有分支。

**極端值的行為。** 當 $x = -100$ 時，$e^{100}$ 會溢位成
$+\infty$，而 $x/\infty = -0$。這是正確的極限，也與 PyTorch
一致。當 $x = +100$ 時，$e^{-100}$ 會下溢成 0，因此得到 $x/1 = x$。
不需要特別處理。替代形式 `x * (1/(1+e^{-x}))` 的行為相同。

## 成本分析

$$
Q = 8N\ \text{bytes}, \qquad W \approx N\,(c_{\exp} + c_{\div})
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（讀取並寫入每個元素） |
| $W$ | 指令工作量；精確 `expf` 的 $c_{\exp} \approx 10$–$20$ 條指令，IEEE 除法的 $c_{\div} \approx 10$ |

當 $N = 5\times10^4$（200 KB）時，這是由啟動延遲主導的單波核心。
當 $N$ 很大時，在多數 GPU 上會受記憶體限制；但由於包含超越函數，
會比 ReLU 更接近運算與記憶體的平衡點。

## 注意事項

- **`__expf` / `__fdividef`** 是誤差較大的高速內建函式。
  對極大分母，`__fdividef` 也會回傳 0 而不是 $-0$，但沒有影響。
  精確版本的誤差遠低於 `1e-5`。
- **以 `expf(x)/(1+expf(x))` 計算 $\sigma$**，會因
  $\infty/\infty$ 而溢位成 NaN，這會發生在很大的正 $x$。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆以
`1e-5` 通過，包括 $\pm100$。

## 相關內容

- [SwiGLU](../054-swiglu/)、[SwiGLU MLP 區塊](../084-swiglu-mlp-block/)、[Sigmoid](../068-sigmoid/)。
- Tensara [Swish](../../tensara/swish/)。
