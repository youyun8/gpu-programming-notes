---
title: ECC 點取負（批次）
platform: Tensara
upstream: ecc-point-negation
url: https://tensara.org/problems/ecc-point-negation
difficulty: easy
tags: [finite-field, integer, elliptic-curve, bandwidth-bound]
status: solved
---

# ECC 點取負（批次）

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/ecc-point-negation)

## 題意

在使用梅森質數 $p = 2^{61} - 1$ 的橢圓曲線
$y^2 \equiv x^3 + 7 \pmod p$ 上，對 $N$ 個點（256 K … 2 M）取負。
座標是 $[0, p)$ 範圍內的 `uint64`。輸出將 $(x_i, -y_i)$ 交錯存放在一個
長度為 $2N$ 的陣列中，並以完全相等為檢查條件。

## 圖解

![橢圓曲線點取負：−(x, y) 是鏡像點 (x, −y mod p)](figure.svg)

為了直觀，曲線畫在實數上：取負就是對 x 軸做鏡射。在有限體 Fₚ 上，−y 變成 p − y（0 仍然是 0）。

## 數學表述

在短 Weierstrass 曲線上，一個點的反元素就是它相對於 $x$ 軸的鏡射：

$$
E:\ y^2 \equiv x^3 + a x + b \pmod p, \qquad -(x, y) = (x,\ -y \bmod p) = \bigl(x,\ (p - (y \bmod p)) \bmod p\bigr)
$$

| 符號 | 意義 |
|---|---|
| $E$ | 曲線；此處 $a = 0$、$b = 7$（形狀與 secp256k1 相同，但使用較小的有限體） |
| $p$ | 有限體模數 $2^{61} - 1$ |
| $(x, y)$ | $E$ 上的一個點，座標屬於 $\mathbb{F}_p = \{0, \dots, p-1\}$ |
| $-(x, y)$ | 加法反元素：$(x, y) + (x, -y) = \mathcal{O}$ |
| $\mathcal{O}$ | 無窮遠點（群的單位元素） |

輸出配置：

$$
\text{out}[2i] = x_i, \qquad \text{out}[2i + 1] = \begin{cases} 0, & y_i \bmod p = 0 \\ p - (y_i \bmod p), & \text{otherwise} \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $x_i, y_i$ | 第 $i$ 個點的座標 |
| out | 交錯排列的結果，共 $2N$ 個字組 |

## 解題思路

使用網格跨步迴圈，每個執行緒處理一個點。執行緒讀取 $x_i$ 與 $y_i$
（兩次合併的 8 位元組載入），再用一次 16 位元組的 `ulonglong2` 儲存寫入
兩個結果。由於輸出陣列以 16 位元組對齊，這項存取自然也是對齊的。
`y == 0 ? 0 : p - y` 分支會編譯成選擇指令。

## 成本分析

$$
Q = 16N + 16N = 32N\ \text{bytes}, \qquad T_{\min} = \frac{32N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：每個點有兩個 8 位元組輸入與一個 16 位元組輸出 |
| $\beta$ | DRAM 頻寬 |

在 $N = 2^{21}$ 時：共 67 MB，以 2 TB/s 計算約需 34 µs。64 位元取模在
NVIDIA GPU 上由軟體指令序列完成（沒有硬體整數除法器），但每位元組的
工作量很少，因此這項成本會隱藏在記憶體延遲之後。

## 常見陷阱

- **$y = 0$** 必須對應至 0，而非不在 $[0, p)$ 內的 $p$。
- **交錯輸出**：若將 $x$ 與 $-y$ 分成兩個陣列，測試就會失敗。
- **無號算術**：因為 $y \bmod p < p$，所以 $p - y$ 絕不會下溢。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [有限體上的多項式乘法](../poly-multiply-ff/)、
  [有限體上的向量乘法](../vector-multiply-ff/)。
