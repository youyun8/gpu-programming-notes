---
title: RMS 正規化
platform: LeetGPU
upstream: medium/50_rms_normalization
url: https://leetgpu.com/challenges/rms-normalization
difficulty: medium
tags: [normalization, reduction, three-pass]
status: solved
---

# RMS 正規化

**平台：** LeetGPU · **難度：** 中等 · [題目敘述](https://leetgpu.com/challenges/rms-normalization)

## 題意

使用**純量**縮放 $\gamma$ 與平移 $\beta$，對一個長度為 $N$ 的 float32
向量執行 RMS 正規化（$N \le 10^5$、$\varepsilon = 10^{-5}$；
效能評測使用 $N = 10^5$；容許誤差為 `1e-5`）。與 LayerNorm 不同，
RMSNorm 不會減去平均值。LLaMA 類 Transformer 採用的就是這種正規化
（其中使用每個特徵各自的 $\gamma$，而且沒有 $\beta$）。

## 圖解

![單一向量的 RMS 正規化：先做全域歸約，再逐元素處理](figure.svg)

所有輸入共同決定一個統計量 RMS，之後每個輸出都使用同一個值。這種「先歸約、再廣播」的兩階段結構是所有正規化 kernel 的核心。

## 數學表述

$$
\operatorname{rms} = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1} x_i^2 + \varepsilon}, \qquad
y_i = \gamma\,\frac{x_i}{\operatorname{rms}} + \beta
$$

| 符號 | 意義 |
|---|---|
| $N$ | 向量長度 |
| $x_i$ | 輸入值 |
| $\varepsilon$ | 加在根號內的穩定常數 |
| $\operatorname{rms}$ | 輸入的均方根 |
| $\gamma,\ \beta$ | 純量縮放與平移參數 |
| $y_i$ | 輸出值 |

輸出取決於整個向量的**全域**統計值。因此，運算方式是先歸約，
再執行廣播的逐元素處理。

## 解題思路

在同一個 stream 中執行三個核心函式：

1. **`sumSquares`**（最多 512 個區塊）：每個執行緒以網格跨步方式，
   使用 float32 執行 `fmaf(x, x, acc)`，再以 float64 區塊歸約，
   將結果寫入 `g_partials[b]`。
2. **`computeInvRms`**（1 個區塊）：以 float64 加總部分結果，並將
   $1/\sqrt{\text{sum}/N + \varepsilon}$ 儲存到 `__device__` 變數
   `g_inv_rms`。儲存**倒數**可將 $N$ 次除法改成 $N$ 次乘法。
3. **`scaleShift`**：以網格跨步方式計算
   `y = γ·(x·inv_rms) + β`。乘法順序與參考實作的
   `gamma * (input / rms) + beta` 相比只差一次捨入，遠低於 `1e-5`。

此純量透過裝置記憶體在核心函式間傳遞，因此不需與主機端往返。

## 成本分析

$$
Q = 4N + 8N = 12N\ \text{bytes}, \qquad W \approx 5N
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取 $x$（第 1 輪），再讀取 $x$ 並寫入 $y$（第 3 輪） |
| $W$ | FLOP 數（平方使用 FMA，輸出輪次再執行一次乘法與一次 FMA） |

$N = 10^5$ 只有 400 KB，可留在 L2 快取中，且受核心函式啟動延遲限制
（三個核心函式約需 10 µs）。對真實 LLM 層中含有多列的批次而言，
單一核心函式可以讓每一列使用一個區塊（或 warp），直接在 SM 內完成歸約
與縮放（請參閱[融合殘差相加 + RMSNorm](../083-fused-residual-add-rms-norm/)）。

## 常見陷阱

- **減去平均值。** RMSNorm 不會這麼做。若與 LayerNorm 混淆，
  結果會立即失敗。
- **$\varepsilon$ 的位置。** 它應加在平方平均值上，位於平方根內。
- **以 float32 加總 $10^5$ 個最大為 $10^4$ 的平方值。**
  上層改用 float64，可將相對誤差維持在約 $10^{-12}$。

## 驗證

所有 LeetGPU 測試案例均以 `1e-5` 的容許誤差在
[cuemu](../../tools/cuemu/README.md) 通過，包括 $N = 1$。

## 延伸閱讀

- [融合殘差相加 + RMSNorm](../083-fused-residual-add-rms-norm/)、
  [層正規化](../113-layer-normalization/)、
  [批次正規化](../040-batch-normalization/)。
  Tensara [RMS 正規化](../../tensara/rms-norm/)。
