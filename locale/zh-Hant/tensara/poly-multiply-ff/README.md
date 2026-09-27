---
title: 有限體上的多項式乘法
platform: Tensara
upstream: poly-multiply-ff
url: https://tensara.org/problems/poly-multiply-ff
difficulty: medium
tags: [finite-field, convolution, mersenne-prime, shared-memory]
status: solved
---

# 有限體上的多項式乘法

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/poly-multiply-ff)

## 題意

將兩個次數為 $n - 1$、係數位於質數體 $\mathbb{F}_p$ 的多項式相乘，其中 $p = 2^{31} - 1$，並回傳乘積的 $2n - 1$ 個係數。係數是 $[0, p)$ 內的 `uint32`，$n$ 是 2 的次方（測試中最大為 1024），且輸出必須完全相符。

## 圖解

![多項式乘法 = 線性卷積：cₖ 是所有 i + j = k 的 aᵢ bⱼ 之和（mod p）](figure.svg)

每一格是一個乘積 aᵢbⱼ。係數 c₄ 是綠色反對角線的總和；每個輸出執行緒負責一條反對角線，並對 p 取模。

## 數學表述

$$
a(x) = \sum_{i=0}^{n-1} a_i x^i, \quad b(x) = \sum_{j=0}^{n-1} b_j x^j, \qquad
c_k = \Bigl(\sum_{\substack{i + j = k \\ 0 \le i, j < n}} a_i\,b_j\Bigr) \bmod p, \qquad 0 \le k \le 2n - 2
$$

| 符號 | 意義 |
|---|---|
| $p$ | Mersenne 質數 $2^{31} - 1 = 2147483647$ |
| $a_i, b_j$ | $[0, p)$ 內的輸入係數 |
| $c_k$ | 輸出係數 $k$（$a$ 與 $b$ 的線性卷積，再對 $p$ 取模） |
| $i, j$ | 滿足 $i + j = k$ 的索引，即 $i \in [\max(0, k-n+1), \min(k, n-1)]$ |

### Mersenne 模數化簡

因為 $2^{31} \equiv 1 \pmod p$，所以數值 $x = h\cdot 2^{31} + \ell$ 滿足 $x \equiv h + \ell$：

$$
\operatorname{fold}(x) = (x \mathbin{\&} p) + (x \gg 31) \equiv x \pmod p
$$

| 符號 | 意義 |
|---|---|
| $x \mathbin{\&} p$ | 低 31 位元 $\ell$ |
| $x \gg 31$ | 高位部分 $h$ |
| Fold | 不需除法的一次縮減；進行兩次 fold，再做一次條件式減法，即可將任何 64 位元值縮減至 $[0, p)$ |

### 為何不使用 NTT？

長度為 $L$ 的數論轉換需要 $L$ 次單位根，而它只有在 $L$ 整除 $p - 1$ 時才存在於 $\mathbb{F}_p$：

$$
p - 1 = 2^{31} - 2 = 2 \cdot 3^2 \cdot 7 \cdot 11 \cdot 31 \cdot 151 \cdot 331
$$

| 符號 | 意義 |
|---|---|
| $p - 1$ | $\mathbb{F}_p$ 乘法群的階 |

其中只含一個因數 2，因此不存在長度為 2 次方的 NTT（必須在適合 NTT 的質數上使用 CRT，或改用 $\mathbb{F}_{p^2}$）。

## 解題思路

1. **每個輸出係數 $k$ 使用一個執行緒**，每個區塊 256 個執行緒。
2. **共享記憶體圖塊**：對每組 1024 元素圖塊 $(i_0, j_0)$，區塊會暫存 $a[i_0 .. i_0+1023]$ 與 $b[j_0 .. j_0+1023]$；每個執行緒走訪其有效 $i$ 範圍 $[\max(i_0, k - j_0 - 1023),\ \min(i_0 + 1023, k - j_0)]$。
3. **延遲縮減**：每個乘積（$< 2^{62}$）先 fold 至小於 $2^{32}$，再加入 64 位元累加器；只有當累加器接近 $2^{62}$ 時才再次 fold。最後進行一次完整縮減以取得 $c_k$。當 $n \le 1024$ 時，總和會保持在 $2^{42}$ 以下，因此不會觸發防護。

## 成本分析

$$
W = n^2\ \text{modular multiply-adds}, \qquad Q = 8n + 4(2n - 1)\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 取模乘加運算次數 |
| $Q$ | DRAM 位元組數（輸入會從共享記憶體重複讀取，而非 DRAM） |

當 $n = 1024$ 時：約需 $10^6$ 次運算，耗時數 µs；核心主要受啟動延遲影響。每一項的 64 位元乘法（`mul.wide.u32`）加上兩次 fold，約需 6 個整數指令。當 $n \gtrsim 10^5$ 時，Karatsuba 或 CRT-NTT 方法會更快。

## 常見陷阱

- **溢位**：$a_ib_j$ 最大可達 $2^{62}$；即使只加總四個未縮減乘積，也會讓 64 位元溢位。每個乘積都應先 fold。
- **輸出長度**為 $2n - 1$，不是 $2n$。
- **最終縮減**必須將 $p$ 本身對應為 0。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [$\mathbb{F}_p$ 上的向量乘法](../vector-multiply-ff/)、[ECC 點取負](../ecc-point-negation/)、[一維卷積](../conv-1d/)。
