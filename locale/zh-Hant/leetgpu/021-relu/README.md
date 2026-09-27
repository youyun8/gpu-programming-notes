---
title: ReLU
platform: LeetGPU
upstream: easy/21_relu
url: https://leetgpu.com/challenges/relu
difficulty: easy
tags: [elementwise, activation, vectorized, memory-bound]
status: solved
---

# ReLU

**平台：** LeetGPU · **難度：** easy · [題目敘述](https://leetgpu.com/challenges/relu)

## 題意

對 $N$ 個 float32 值逐元素套用修正線性單元
（$1 \le N \le 10^8$；基準測試為 $N = 2.5\times10^7$）。
ReLU 是 CNN 與 MLP 的預設非線性函數。作為核心函式，它與
[向量加法](../001-vector-add/)使用相同模式，只是輸入資料流從兩個減為一個。

## 圖解

![ReLU：正數保留，負數變成 0](figure.svg)

曲線為 y = max(0, x)。橘色點表示負輸入被映射為 0；紅色點表示正輸入原樣通過。

## 數學表述

$$
y_i = \operatorname{ReLU}(x_i) = \max(0, x_i) =
\begin{cases} x_i, & x_i > 0 \\ 0, & x_i \le 0 \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $N$ | 元素數 |
| $x_i$ | 輸入值（float32） |
| $y_i$ | 輸出值（float32） |

其導數（反向傳播需要，但此處不需要）是階躍函數
$\mathbb 1[x > 0]$。

## 解題思路

- **向量化主體。** 執行緒 $t < \lfloor N/4\rfloor$ 載入一個 `float4`，
  對每個 lane 套用 `fmaxf(v, 0.0f)`，再儲存一個 `float4`。`fmaxf`
  是單一 `FMNMX` 指令，因此沒有分支，也沒有分歧。
- **純量尾端。** 執行緒 $t < N \bmod 4$ 也會處理元素
  $4\lfloor N/4\rfloor + t$。
- **網格。** 使用 $\lfloor (\lfloor N/4\rfloor + 256)/256 \rfloor$
  個區塊，足以涵蓋向量部分，並確保即使 $N < 4$，尾端執行緒仍有區塊可執行。

## 成本分析

$$
Q = 8N \ \text{bytes}, \qquad W = N, \qquad I = \frac18\ \text{FLOP/byte}, \qquad T_{\min} = \frac{8N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 傳輸量：讀取 $x$、寫入 $y$ |
| $W$ | 每個元素一次最大值運算 |
| $I$ | 算術強度 |
| $\beta$ | DRAM 頻寬 |

基準測試：$Q = 200$ MB，因此在 2 TB/s 下
$T_{\min} \approx 100\ \mu s$。在實際網路中，ReLU 幾乎都會**融合**
到前一個 GEMM 或卷積的收尾階段（請參閱 Tensara
[GEMM + ReLU](../../tensara/gemm-relu/)），以省下整趟 DRAM 往返。

## 常見陷阱

- **NaN 處理。** `fmaxf(NaN, 0) = 0`，但 `torch.relu(NaN) = NaN`。
  測試輸入不含 NaN。傳播 NaN 的版本可寫成
  `x > 0 ? x : (x != x ? x : 0)`。
- **$-0.0$。** `fmaxf(-0.0f, 0.0f)` 可能傳回任一種零。
  兩者比較時相等，因此都能通過檢查。

## 驗證

所有 LeetGPU 測試案例都已在 [cuemu](../../tools/cuemu/README.md) 通過，
包括 $N = 1, 2, 3$（只走尾端路徑）。

## 延伸閱讀

- [Leaky ReLU](../023-leaky-relu/)、[Sigmoid](../068-sigmoid/)、[SiLU](../052-silu/)。
- Tensara [ReLU](../../tensara/relu/)。
