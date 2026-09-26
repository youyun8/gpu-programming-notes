---
title: LoRA 線性層
platform: LeetGPU
upstream: medium/85_lora_linear
url: https://leetgpu.com/challenges/lora-linear
difficulty: medium
tags: [gemm, lora, fusion, fine-tuning]
status: solved
---

# LoRA 線性層

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/lora-linear)

## 問題

實作 **LoRA** 線性層的前向傳播：凍結的基礎權重 $W$，加上以 $s$ 縮放、
可訓練的低秩更新 $BA$
（$x \in \mathbb R^{b\times d_{\text{in}}}$、$W \in \mathbb R^{d_{\text{out}}\times d_{\text{in}}}$、
$A \in \mathbb R^{r \times d_{\text{in}}}$、$B \in \mathbb R^{d_{\text{out}}\times r}$、
$r \le 256$；基準測試 $b = 256$、$d_{\text{in}} = d_{\text{out}} = 4096$、
$r = 64$；容許誤差 `1e-4`）。LoRA 只訓練 $A$ 與 $B$ 來微調大型模型；
此處只有 $2\cdot64\cdot4096 = 0.5$M 個參數，而非 16.8M。

## 公式

$$
Y = xW^{\mathsf T} + s\,(xA^{\mathsf T})B^{\mathsf T} = \begin{bmatrix} x & s\,xA^{\mathsf T}\end{bmatrix}\begin{bmatrix} W & B\end{bmatrix}^{\mathsf T}
$$

| 符號 | 意義 |
|---|---|
| $b$ | 批次大小 |
| $d_{\text{in}},\ d_{\text{out}}$ | 輸入與輸出特徵數 |
| $r$ | LoRA 秩 |
| $x$ | 輸入，$b\times d_{\text{in}}$ |
| $W$ | 凍結的基礎權重（`nn.Linear` 配置，out × in） |
| $A$ | 下投影，$r\times d_{\text{in}}$ |
| $B$ | 上投影，$d_{\text{out}}\times r$ |
| $s$ | `lora_scale`（通常為 $\alpha/r$） |
| $[\,\cdot\ \cdot\,]$ | 沿內部（K）維度水平串接 |
| $Y$ | 輸出，$b\times d_{\text{out}}$ |

右側形式表示：只要先求得小矩陣 $h = s\,xA^{\mathsf T}$，整個線性層就能
視為在串接後的內部維度 $d_{\text{in}} + r$ 上執行的**單一 GEMM**。

### 為何不合併 $W + sBA$？

若推論時使用固定的 adapter，預先計算 $W' = W + sBA$ 可完全消除 LoRA 成本。
但使用多個 adapter（多租戶服務）或進行訓練時，adapter 必須保持獨立，
因此前向傳播需分別計算兩條路徑。

## 方法

1. **核心 1**：$h = s\,xA^{\mathsf T}$（$b\times r$），這是「NT」GEMM
   （右側運算元以列優先格式儲存為 $r\times d_{\text{in}}$），並在結尾運算
   套用縮放比例。
2. **核心 2**：`gemmNtConcat` 計算 $[x\ h][W\ B]^{\mathsf T}$。其 $K$
   迴圈包含**兩段**：先使用運算元 $(x, W)$ 走訪 $d_{\text{in}}$，再使用
   $(h, B)$ 走訪 $r$。兩段都累加到相同的 $4\times4$ 暫存器，因此輸出只
   寫入**一次**，不需額外的加法核心。

兩者皆使用 64 × 64 暫存器分塊 tile。在 NT 配置中，載入 B 側 tile 時，
$k$ 是最快變動的維度，因此讀取權重列可合併。

## 成本分析

$$
W_{\text{flop}} = 2b\,d_{\text{in}}\,d_{\text{out}} + 2b\,r\,(d_{\text{in}} + d_{\text{out}}), \qquad
\frac{W_{\text{LoRA}}}{W_{\text{base}}} = \frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}}d_{\text{out}}}
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{flop}}$ | 基礎與 LoRA 路徑的 FLOP 數 |
| $W_{\text{LoRA}}/W_{\text{base}}$ | Adapter 的相對額外成本 |

基準測試的基礎 GEMM 為 8.6 GFLOP，而 LoRA 增加
$2\cdot 64/4096 = 3.1\%$。串接 K 維度的融合也能避免寫入再讀取一個
$b\times d_{\text{out}}$ 的部分結果（8 MB）。

## 常見陷阱

- **縮放比例的位置。** $s$ 只乘上 LoRA 路徑。在核心 1 對 $h$ 套用它，
  可讓核心 2 只執行一般加總。
- **配置。** $W$ 與 $B$ 為 $(\text{out}, \text{in})$，$A$ 為
  $(r, \text{in})$，因此所有右側運算元都是「轉置」形式（NT）。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-4`
通過，包括 $r = 1$ 與小批次。

## 相關內容

- [簡易推論](../041-simple-inference/)、[SwiGLU MLP](../084-swiglu-mlp-block/)、
  [矩陣乘法](../002-matrix-multiplication/)。
