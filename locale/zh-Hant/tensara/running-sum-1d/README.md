---
title: 一維移動總和
platform: Tensara
upstream: running-sum-1d
url: https://tensara.org/problems/running-sum-1d
difficulty: easy
tags: [scan, prefix-sum, sliding-window, fp64-accumulation]
status: solved
---

# 一維移動總和

**平台：** Tensara · **難度：** 簡單 · [題目敘述](https://tensara.org/problems/running-sum-1d)

## 問題

對長度為 $N$（32 K … 512 K）的 float32 訊號計算滑動視窗總和。視窗含
$W = 8191$ 個元素，相當於使用全為 1 的核心執行 `conv1d`，並填補
$\lfloor W/2 \rfloor$ 個 0。檢查條件為 `rtol = 5e-4`、`atol = 3e-2`。

## 公式

$$
h = \left\lfloor \frac{W}{2} \right\rfloor, \qquad
\text{out}[i] = \sum_{j=0}^{W-1} \tilde{x}[\,i + j - h\,], \qquad 0 \le i < L = N + 2h - W + 1
$$

| 符號 | 意義 |
|---|---|
| $x$ | 長度為 $N$ 的輸入訊號；$\tilde{x}$ 是 $x$ 在 $[0, N)$ 之外補 0 後的結果 |
| $W$ | 視窗長度 |
| $h$ | 填補量，$\lfloor W/2\rfloor$ |
| $L$ | 輸出長度（$= N$，當 $W$ 為奇數） |
| $i$ | 輸出索引 |

使用含目前元素的前綴和 $\Pi$，每個視窗都可表示成兩個前綴值之差：

$$
\Pi[t] = \sum_{q=0}^{t} x[q], \qquad
\ell = \max(i - h, 0), \quad r = \min(i - h + W - 1,\ N - 1), \qquad
\text{out}[i] = \Pi[r] - \Pi[\ell - 1]
$$

| 符號 | 意義 |
|---|---|
| $\Pi[t]$ | 含目前元素的前綴和，其中 $\Pi[-1] = 0$ |
| $\ell, r$ | 視窗 $i$ 中第一個與最後一個位於輸入範圍內的索引 |

這會把 $O(NW)$ 次加法降為 $O(N)$。

## 方法

1. **以 fp64 執行含目前元素的掃描**，將結果寫入暫存 `double` 陣列。使用
   [Cumsum](../cumsum/) 的三核心先歸約再掃描方式（每塊 2048 個元素，
   進位值使用 fp64），但把輸出型別改成 `double`。
2. **`windowSums`**：每個輸出由一個執行緒讀取 $\Pi[r]$ 與
   $\Pi[\ell - 1]$，以 `double` 相減，最後只取整一次成 float。

使用 fp64 的原因：$\Pi$ 會成長至 $\sim\sqrt{N}$，隨機資料甚至可能更大，
但視窗總和小得多，相減時會消去高位數字。若使用 fp32，絕對誤差約為
$u\,\lvert\Pi\rvert$：

$$
\bigl\lvert \Delta\text{out} \bigr\rvert \lesssim 2u\,\max_t \lvert \Pi[t] \rvert, \qquad u_{32} = 2^{-24},\ u_{64} = 2^{-53}
$$

| 符號 | 意義 |
|---|---|
| $\Delta\text{out}$ | 使用取整後前綴值計算一個視窗總和時的誤差 |
| $u_{32}, u_{64}$ | float 與 double 的單位捨入誤差 |

## 成本分析

$$
Q \approx 4N + 4N\ (\text{scan reads}) + 8N\ (\text{write }\Pi) + 16N\ (\text{read two }\Pi) + 4N\ (\text{write out}) = 36N\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數（每個輸出讀取的兩個 $\Pi$ 多半會命中 L2，因為相鄰輸出會讀取相鄰前綴值） |

$N = 512$ K 時約為 19 MB，耗時約 10 µs；直接方法需要
$4\times10^9$ 次加法，耗時達毫秒級。[Conv 1D](../conv-1d/) 的共享記憶體
方法也可行，但掃描能善用所有權重皆為 1 的特性。

## 注意事項

- **輸出長度**為 $L = N + 2h - W + 1$（等於 $N$，當 $W$ 為奇數）。
- **邊界視窗**會在兩端裁切（$\ell$ 與 $r$），對總和而言，效果恰好等同補 0。
- **fp32 前綴和**在 $N$ 很大時無法通過容許誤差。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [累積和](../cumsum/)、[一維卷積](../conv-1d/)、[方框模糊](../box-blur/)、
  LeetGPU [子陣列總和](../../leetgpu/047-subarray-sum/)。
