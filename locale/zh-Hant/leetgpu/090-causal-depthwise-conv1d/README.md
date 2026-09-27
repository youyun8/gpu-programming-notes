---
title: 因果深度可分離一維卷積
platform: LeetGPU
upstream: medium/90_causal_depthwise_conv1d
url: https://leetgpu.com/challenges/causal-depthwise-conv1d
difficulty: medium
tags: [convolution, depthwise, ssm, channels-last]
status: solved
---

# 因果深度可分離一維卷積

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/causal-depthwise-conv1d)

## 題意

對通道置後張量 $x \in \mathbb R^{B\times L\times D}$ 執行因果深度可分離一維卷積。每個通道 $d$ 都有自己的 $K$ 點濾波器，輸出位置 $l$ 只能看到輸入 $l-K+1 \dots l$（$B \le 16$、$L, D \le 8192$、$K \le 8$；容許誤差 `1e-4`）。這是 Mamba 在選擇性掃描之前使用的短卷積，用來混合各通道內的局部上下文。

## 圖解

![因果逐通道一維卷積：通道 d 的位置 l 只看得到 l−K+1 … l](figure.svg)

只在左側填補兩個零。輸出 4 以該通道自己的權重組合輸入 2、3、4，因此任何輸出都不會看到未來。

## 數學表述

$$
y_{b,l,d} = \beta_d + \sum_{k=0}^{K-1} w_{d,k}\; x_{b,\,l-k,\,d}, \qquad x_{b,\,l',\,d} = 0 \ \text{for}\ l' < 0
$$

$$
\text{offset}(b, l, d) = (bL + l)\,D + d \qquad (\text{channels-last})
$$

| 符號 | 意義 |
|---|---|
| $B,\ L,\ D$ | 批次、序列長度、通道數 |
| $K$ | 核心點數（$\le 8$） |
| $x_{b,l,d}$ | 輸入；序列開始前補零（因果左側填補） |
| $w_{d,k}$ | 通道 $d$ 在延遲 $k$ 的權重：$w_{d,0}$ 套用於目前位置，$w_{d,K-1}$ 套用於最舊位置 |
| $\beta_d$ | 每通道偏置 |
| $y_{b,l,d}$ | 輸出，配置與 $x$ 相同 |

「深度可分離」表示通道之間不會混合：共有 $D$ 個獨立的一維濾波器。參數量為 $D(K+1)$，完整卷積則為 $D^2K$。

## 解題思路

- 使用網格 $\lceil D/128\rceil \times \lceil L/8\rceil \times B$，每個區塊有 128 個執行緒，`threadIdx.x` 沿著**通道**方向。
- 在通道置後配置中，固定的 $(b, l)$ 是由 $D$ 個浮點數組成的連續資料列，因此一個 warp 對每個點讀取 `x[b, l-k, :]` 時，都是一次合併的 128 位元組區段。
- 每個執行緒只需將其通道的 $\le 8$ 個權重與偏置載入**暫存器**一次，接著計算**連續 8 個位置**並重複使用它們。
- 從最舊的點開始累加（$k = K-1 \to 0$），與 `F.conv1d` 對左側填補且翻轉後的核心所做的運算相符，因此捨入順序也相同。

每個輸入元素最多會被同一執行緒的 $K$ 個不同輸出位置讀取（重疊視窗）。這些重複讀取會命中 L1。

## 成本分析

$$
W = 2BLDK, \qquad Q_{\min} = 8BLD + 4D(K+1)\ \text{bytes}, \qquad I \approx \frac{K}{4}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數 |
| $Q_{\min}$ | 必要位元組數：讀取 $x$ 一次、寫入 $y$ 一次（另加可忽略的權重） |
| $I$ | 算術強度 |

由於 $K \le 8$，所以 $I \le 2$ FLOP/byte，核心受記憶體頻寬限制。當 $B = 16$、$L = 8192$、$D = 8192$ 時：$Q = 8.6$ GB，亦即在 2 TB/s 下約需 4.3 ms。

## 常見陷阱

- **核心方向。** `weight[d, 0]` 乘上的是**目前的**輸入。參考實作會先翻轉核心再呼叫 `conv1d`，因此若直接以 `weight[d, k]` 對位移 $+k$ 做互相關，結果會出錯。
- **因果填補**只加在左側，共 $K-1$ 個零。
- **配置。** 通道置後表示 $d$ 的變化最快。若沿 $l$ 方向配置執行緒，每次存取都會跨越 $D$ 個元素。

## 驗證

所有 LeetGPU 測試案例都以 `1e-4` 容許誤差在 [cuemu](../../tools/cuemu/README.md) 上通過，包括 $K = 1$ 與 $L < K$。

## 延伸閱讀

- [SSM 選擇性掃描](../094-ssm-selective-scan/)、[一維卷積](../009-1d-convolution/)、[線性遞迴](../082-linear-recurrence/)。
