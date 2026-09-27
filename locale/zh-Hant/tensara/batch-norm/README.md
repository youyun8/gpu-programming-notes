---
title: 批次正規化
platform: Tensara
upstream: batch-norm
url: https://tensara.org/problems/batch-norm
difficulty: medium
tags: [normalization, reduction, fp64-accumulation]
status: solved
---

# 批次正規化

**平台：** Tensara · **難度：** medium · [題目敘述](https://tensara.org/problems/batch-norm)

## 問題

對形狀為 $(B, F, D_1, D_2)$ 的 float32 張量執行批次正規化，不使用仿射參數或移動統計值，且 $\epsilon = 10^{-5}$。參考實作是在訓練模式下執行
`nn.BatchNorm2d(F, affine=False, track_running_stats=False)`。該層會在批次和兩個空間軸上，為**每個通道 $f$ 計算一個平均值和一個變異數**。（題目敘述中「每個空間位置」的附註與參考實作不符；以參考實作為準。）檢查誤差為 `rtol = atol = 1e-4`。

## 公式

$$
\mu_f = \frac{1}{N} \sum_{b=0}^{B-1} \sum_{p=0}^{D_1D_2-1} x_{b,f,p}, \qquad
\sigma_f^2 = \frac{1}{N} \sum_{b=0}^{B-1} \sum_{p=0}^{D_1D_2-1} \bigl(x_{b,f,p} - \mu_f\bigr)^2, \qquad
N = B\,D_1 D_2
$$

$$
y_{b,f,p} = \frac{x_{b,f,p} - \mu_f}{\sqrt{\sigma_f^2 + \epsilon}}
$$

| 符號 | 意義 |
|---|---|
| $B, F$ | 批次大小、通道數 |
| $D_1, D_2$ | 空間範圍；$p$ 是扁平化的空間索引 |
| $x_{b,f,p}$ | 批次 $b$、通道 $f$、位置 $p$ 的輸入；扁平偏移量為 $(bF + f)D_1D_2 + p$ |
| $N$ | 共用同一組通道統計值的元素數量 |
| $\mu_f$ | 每個通道的平均值 |
| $\sigma_f^2$ | 每個通道的**有偏**變異數（除以 $N$，而非 $N-1$） |
| $\epsilon$ | $10^{-5}$，避免平方根的輸入為 0 |
| $y_{b,f,p}$ | 輸出，版面配置與 $x$ 相同 |

## 方法

1. **每個通道使用一個執行緒區塊**（$F$ 個區塊，每個 1024 個執行緒）。每個通道的資料由 $B$ 個連續區塊組成，每塊包含
   $D_1D_2$ 個浮點數，區塊之間的跨距為 $FD_1D_2$。執行緒走訪扁平索引 $i \in [0, N)$，映射為
   $b = \lfloor i / D_1D_2 \rfloor$ 和 $p = i \bmod D_1D_2$，因此相鄰執行緒讀取連續位址：每次載入皆為合併存取。
2. **兩趟統計**：先算平均值，再算以平均值為中心的平方和。這可避免
   $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ 的消去誤差。兩個總和皆使用
   `double` 累加，並以 warp shuffle 加上一個 32 項的共享陣列進行縮減。
3. 第三趟執行**正規化**，其中
   $r_f = 1/\sqrt{\sigma_f^2+\epsilon}$ 在每個區塊中只計算一次。

## 成本分析

$$
Q = 3 \cdot 4\,BFD_1D_2 \ (\text{reads}) + 4\,BFD_1D_2 \ (\text{writes}), \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取三趟，寫入一趟 |
| $\beta$ | DRAM 頻寬 |

在 $(16, 64, 256, 256)$ 下，張量為 268 MB，因此
$Q \approx 1.07$ GB，在 2 TB/s 下約需 0.55 ms。由於只有
$F = 32 \dots 256$ 個區塊，$F$ 較小時無法充分利用 GPU。將每個通道拆分到多個區塊（先算部分總和，再執行第二個小型核心函式），或使用單趟 Welford 演算法，都能同時改善這兩個問題。

## 常見陷阱

- **統計值按通道計算，而不是按位置計算**：照字面遵循題目附註會得到不同且錯誤的答案。
- 使用和 PyTorch 訓練模式相同的**有偏變異數**（$1/N$）。
- **精確度**：$N$ 可達 $4\cdot 512^2 \approx 10^6$；fp32 連續累加會產生誤差，fp64 則不會。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 相關內容

- [層正規化](../layer-norm/)、[RMS 正規化](../rms-norm/)、
  LeetGPU [批次正規化](../../leetgpu/040-batch-normalization/)。
