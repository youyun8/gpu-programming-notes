---
title: 有限體上的向量乘法
platform: Tensara
upstream: vector-multiply-ff
url: https://tensara.org/problems/vector-multiply-ff
difficulty: medium
tags: [finite-field, elementwise, mersenne-prime]
status: solved
---

# 有限體上的向量乘法

**平台：** Tensara · **難度：** 中等 · [題目敘述](https://tensara.org/problems/vector-multiply-ff)

## 題意

在 $\mathbb{F}_p$ 中逐元素計算兩個 `uint32` 向量的乘積，其中
$p = 2^{31} - 1$、$n = 2^{20} \dots 2^{25}$。輸出必須完全精確。

## 圖解

![在 Fₚ（p = 2³¹ − 1）中相乘：不用除法折疊 62 位元的乘積](figure.svg)

每個執行緒把一對數相乘得到 62 位元的乘積，利用 2³¹ ≡ 1 (mod p) 折疊兩次，最後最多再減一次 p。

## 數學表述

$$
c_i = a_i\,b_i \bmod p, \qquad 0 \le a_i, b_i < p
$$

| 符號 | 意義 |
|---|---|
| $p$ | 梅森質數 $2^{31} - 1$ |
| $a_i, b_i$ | 輸入 |
| $c_i$ | 位於 $[0, p)$ 的輸出 |

不使用除法來歸約 62 位元乘積 $x = a_ib_i$：

$$
x_1 = (x \mathbin{\&} p) + (x \gg 31) < 2^{32}, \qquad
x_2 = (x_1 \mathbin{\&} p) + (x_1 \gg 31) \le p + 1, \qquad
c_i = \begin{cases} x_2 - p, & x_2 \ge p \\ x_2, & \text{otherwise}\end{cases}
$$

| 符號 | 意義 |
|---|---|
| $x$ | 64 位元乘積，$< 2^{62}$ |
| $x_1, x_2$ | 第一次與第二次摺疊後的值（各自與 $x$ 同餘） |

此方法成立是因為 $2^{31} \equiv 1 \pmod p$，因此高位部分可直接加至
低位部分。

## 解題思路

使用網格步進迴圈（256 個執行緒，最多 4096 個區塊）；每個執行緒計算
`mulModMersenne31(a[i], b[i])`。編譯器會產生一個 `mul.wide.u32`，
再加上少量位移、AND、加法與選擇指令；相較之下，64 位元 `%`
在 NVIDIA GPU 上會使用緩慢的軟體除法常式。

## 成本分析

$$
Q = 12n\ \text{bytes}, \qquad T_{\min} = \frac{12n}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：兩個輸入與一個輸出，各 4 位元組 |
| $\beta$ | DRAM 頻寬 |

當 $n = 2^{25}$ 時，資料量為 403 MB；在 2 TB/s 下約為 0.2 ms。
使用摺疊時，核心受頻寬限制；使用 `%` 時則可能受 ALU 限制。

## 常見陷阱

- **32 位元乘法會溢位**：相乘前先轉型成 `uint64_t`。
- **一次摺疊不夠**：第一次摺疊後，值仍可能高達
  $2^{32} - 1 > p$。
- **$x_2 = p$** 必須變成 0。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [有限體上的多項式乘法](../poly-multiply-ff/)、
  [ECC 點取負](../ecc-point-negation/)、[向量加法](../vector-addition/)。
