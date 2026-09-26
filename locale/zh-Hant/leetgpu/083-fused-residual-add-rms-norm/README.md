---
title: 融合殘差加法與 RMS 正規化
platform: LeetGPU
upstream: medium/83_fused_residual_add_rms_norm
url: https://leetgpu.com/challenges/fused-residual-add-and-rms-norm
difficulty: medium
tags: [normalization, fusion, row-reduction, llm]
status: solved
---

# 融合殘差加法與 RMS 正規化

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/fused-residual-add-and-rms-norm)

## 問題

將 LLaMA 類 transformer 的「加法與正規化」步驟融合成一個核心：先將
子層輸出 $x$ 加入殘差流 $r$，接著對每一列執行 RMS 正規化，並乘上逐特徵
權重（$N, C \le 65\,536$、$\varepsilon = 10^{-5}$；基準測試
$N = C = 4096$）。中間值 $z = x + r$ 不可寫入全域記憶體，這正是融合的目的。

## 公式

$$
z_{ij} = x_{ij} + r_{ij}, \qquad
\operatorname{rms}_i = \sqrt{\frac1C\sum_{j=0}^{C-1} z_{ij}^2 + \varepsilon}, \qquad
y_{ij} = \frac{z_{ij}}{\operatorname{rms}_i}\, w_j
$$

| 符號 | 意義 |
|---|---|
| $N$ | Token 數（列） |
| $C$ | 隱藏維度（欄） |
| $x_{ij}$ | 子層輸出 |
| $r_{ij}$ | 殘差流 |
| $z_{ij}$ | 更新後的殘差（只保留在暫存器或快取中） |
| $\varepsilon$ | 穩定常數（$10^{-5}$） |
| $\operatorname{rms}_i$ | 第 $i$ 列的均方根 |
| $w_j$ | 逐特徵權重（$\gamma$） |
| $y_{ij}$ | 正規化後的輸出 |

## 方法

**每列使用一個含 256 個執行緒的區塊：**

1. **第一輪。** 串流走訪整列，即時計算 $z = x + r$，並在暫存器中累加
   $\sum z^2$。當 $C$ 是 4 的倍數時使用 `float4` 載入，否則使用純量載入。
   區塊歸約（warp shuffle，再透過共享記憶體銜接）會產生總和，並計算
   `inv_rms = rsqrtf(sum/C + eps)`。
2. **第二輪。** 再次串流走訪整列、重新計算 $z$，並寫入
   $z\cdot\text{inv\_rms}\cdot w_j$。該列剛被讀過，且兩個輸入最多共
   512 KB，因此第二次讀取由 L1/L2 提供，而非 DRAM。

### 融合節省的成本

| 變體 | 每個元素的 DRAM 流量 |
|---|---|
| 獨立加法核心 + RMSNorm 核心 | 讀取 $x, r$、寫入 $z$（12 B）+ 讀取 $z$ 兩次、寫入 $y$（12 B）= 24 B |
| 融合（此核心） | 讀取 $x, r$（8 B）+ 寫入 $y$（4 B）= 12 B |

流量減少一半，而這項運算在每個 transformer 層中都會執行兩次。

## 成本分析

$$
Q = 12NC\ \text{bytes}, \qquad W \approx 5NC, \qquad T_{\min} = \frac{12NC}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 融合核心的 DRAM 位元組數 |
| $W$ | FLOP 數（加法、平方、累加及兩次乘法） |
| $\beta$ | DRAM 頻寬 |

基準測試為 201 MB，在 2 TB/s 下約需 100 µs。

## 常見陷阱

- **平均值是針對每列的 $C$ 個值**，不是像
  [RMS 正規化](../050-rms-normalization/)那樣的全域統計值。
- **超出快取長度的列。** 當 $C = 65\,536$ 時，一列輸入為 512 KB，仍可
  放入 L2。若列更長，第二輪就會再次讀取 DRAM，此時需要將 $z$ 保留在
  共享記憶體或暫存器中的單輪變體。
- **真正的 LLM 核心**也會原地將 $z$ 寫回成新的殘差流。本題不要求此操作。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上通過，
包括 $C = 1$，以及 $C$ 不能被 4 整除的情況（純量路徑）。

## 相關內容

- [RMS 正規化](../050-rms-normalization/)、[LLaMA Transformer 區塊](../093-llama-transformer-block/)、
  [Layer 正規化](../113-layer-normalization/)。Tensara [RMS Norm](../../tensara/rms-norm/)。
