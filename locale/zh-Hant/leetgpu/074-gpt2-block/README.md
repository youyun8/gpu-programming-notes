---
title: GPT-2 Transformer 區塊
platform: LeetGPU
upstream: hard/74_gpt2_block
url: https://leetgpu.com/challenges/gpt-2-transformer-block
difficulty: hard
tags: [transformer, gpt-2, gemm, fusion, layernorm, attention]
status: solved
cuemu_max_elements: 16777216
---

# GPT-2 Transformer 區塊

**平台：** LeetGPU · **難度：** 困難 · [題目說明](https://leetgpu.com/challenges/gpt-2-transformer-block)

## 問題

以 float32 實作一個完整的 **GPT-2（124M）解碼器區塊**。輸入
$x \in \mathbb R^{S \times 768}$，以及一個包含區塊所有參數的封裝權重緩衝區
（LayerNorm 縮放與位移、以 (in, out) 儲存的 QKV／輸出／MLP 權重及偏置），
產生區塊輸出（容許誤差 `1e-3`）。本題結合 GEMM、LayerNorm、注意力與
活化函數，並展示**核心融合**如何消除大部分逐元素流量。

## 公式

Pre-LayerNorm 殘差區塊，$d = 768$、$H = 12$ 個注意力頭，每頭
$d_h = 64$，MLP 寬度為 $4d = 3072$：

$$
\begin{aligned}
X_1 &= \operatorname{LN}_1(X), & [Q\ K\ V] &= X_1 W_{qkv} + \mathbf b_{qkv} \\
A_h &= \operatorname{softmax}\!\Bigl(\tfrac{Q_h K_h^{\mathsf T}}{\sqrt{d_h}}\Bigr) V_h, & X' &= X + \operatorname{Concat}(A_1..A_H)\,W_o + \mathbf b_o \\
X_2 &= \operatorname{LN}_2(X'), & Y &= X' + \operatorname{GELU}_{\tanh}\!\bigl(X_2W_{fc} + \mathbf b_{fc}\bigr)W_{\text{proj}} + \mathbf b_{\text{proj}}
\end{aligned}
$$

$$
\operatorname{LN}(\mathbf z) = \gamma\odot\frac{\mathbf z - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta, \qquad
\operatorname{GELU}_{\tanh}(u) = \tfrac12 u\Bigl(1 + \tanh\bigl(\sqrt{2/\pi}\,(u + 0.044715\,u^3)\bigr)\Bigr)
$$

| 符號 | 意義 |
|---|---|
| $S$ | 序列長度（`seq_len`） |
| $d$ | 模型寬度，768 |
| $H,\ d_h$ | 注意力頭數（12）及每頭寬度（64） |
| $X$ | 區塊輸入，$S\times d$ |
| $\operatorname{LN}_1,\ \operatorname{LN}_2$ | 各自具有長度為 $d$ 的 $\gamma, \beta$，且 $\varepsilon = 10^{-5}$ 的 LayerNorm |
| $\mu,\ \sigma^2$ | $\mathbf z$ 各列的平均值與有偏變異數 |
| $W_{qkv},\ \mathbf b_{qkv}$ | 融合的 QKV 投影，$d\times 3d$ 與 $3d$ |
| $Q_h, K_h, V_h$ | $Q$、$K$、$V$ 中的欄 $[h d_h, (h+1)d_h)$ |
| $A_h$ | 第 $h$ 頭的注意力輸出（$S\times d_h$）；本題沒有因果遮罩 |
| $W_o,\ \mathbf b_o$ | 注意力輸出投影，$d\times d$ |
| $X'$ | 第一個殘差之後的隱藏狀態 |
| $W_{fc},\ \mathbf b_{fc}$ | MLP 上投影，$d \times 4d$ |
| $W_{\text{proj}},\ \mathbf b_{\text{proj}}$ | MLP 下投影，$4d\times d$ |
| $\operatorname{GELU}_{\tanh}$ | 使用 tanh 近似的 GELU（與參考實作的 `approximate="tanh"` 相同） |
| $Y$ | 區塊輸出 |

## 方法

### 核心執行順序

| # | 核心 | 計算內容 | 融合的結尾運算 |
|---|---|---|---|
| 1 | `layerNormRows` | $X_1 = \operatorname{LN}_1(X)$ | – |
| 2 | GEMM $S\times d\cdot d\times 3d$ | $QKV$ | $+\,\mathbf b_{qkv}$ |
| 3 | Flash attention | $A = \operatorname{Concat}(A_h)$ | – |
| 4 | GEMM $S\times d\cdot d\times d$ | $X'$ | $+\,\mathbf b_o + X$（殘差） |
| 5 | `layerNormRows` | $X_2 = \operatorname{LN}_2(X')$ | – |
| 6 | GEMM $S\times d\cdot d\times 4d$ | MLP 隱藏層 | $+\,\mathbf b_{fc}$，再執行 GELU |
| 7 | GEMM $S\times 4d\cdot 4d\times d$ | $Y$ | $+\,\mathbf b_{\text{proj}} + X'$（殘差） |

### 具有可替換結尾運算的 GEMM

64 × 64 暫存器分塊 SGEMM 是 `gemmKernel<kTransB, Epi>` 範本。累加完成後，
每個執行緒會對自己的 16 個輸出呼叫 `epi(acc, r, c)`。仿函式如下：

- `BiasEpi`：$v + b_c$；
- `BiasGeluEpi`：$\operatorname{GELU}_{\tanh}(v + b_c)$；
- `BiasResidualEpi`：$v + b_c + R_{rc}$。

仿函式會在編譯時內嵌，因此融合本身沒有額外成本。若不融合，每個運算都會
成為獨立的逐元素核心，讀寫一個 $S\times 3d$ 或 $S\times 4d$ 張量。

### 直接從封裝 QKV 緩衝區計算注意力

QKV GEMM 會寫入長度為 $3d$ 的列：$[Q\,|\,K\,|\,V]$。第 $h$ 頭第 $i$ 列的
query 位於 $\text{qkv} + i\cdot 3d + h d_h$，key 再加偏移量 $+d$，value
再加 $+2d$。跨距式 flash-attention 核心（每個 query 列一個 warp、線上
softmax；參見 [多頭注意力](../012-multi-head-attention/)）會把這些跨距當成
參數。這可省去參考實作的 `view/transpose/contiguous` 重排，並直接寫入
串接後的 $S\times d$ 配置。

### 每列一個 Warp 的 LayerNorm

每個 warp 處理一列 768 個值，每個 lane 處理 24 個值。各 lane 以 float64
累加 $\sum z$ 與 $\sum z^2$，並透過 shuffle 歸約。接著使用 $\mu$ 與
$\sigma^2 = E[z^2] - \mu^2$ 正規化；在 float64 中此計算很安全。

## 成本分析

$$
W \approx \underbrace{2S d(3d)}_{QKV} + \underbrace{4S^2 d}_{\text{attention}} + \underbrace{2Sd^2}_{W_o} + \underbrace{2\cdot 2S d(4d)}_{\text{MLP}} = 24Sd^2 + 4S^2d
$$

| 符號 | 意義 |
|---|---|
| $W$ | 區塊的 FLOP 數（LayerNorm 與逐元素項目可忽略） |
| $24Sd^2$ | 常見的「每個 token 約為每層參數量的 $24 \times$」規則：區塊約有 $12d^2$ 個權重 |
| $4S^2 d$ | 注意力分數與 $PV$（對 $S$ 呈平方成長） |

當 $S = 1024$ 時，$W \approx 1.45\times10^{10} + 3.2\times10^{9} \approx 18$
GFLOP。GEMM 佔主要成本，因此 SGEMM 的效率決定執行時間。在 $S = 1024$
時，融合可省下約 $2\cdot 4\,(3d + d + 4d + d)S$ 位元組，也就是約 75 MB
的逐元素流量。

## 常見陷阱

- **權重配置。** 封裝矩陣採 $(\text{in}, \text{out})$，也就是 $X W$，
  不同於 `nn.Linear` 的 $X W^{\mathsf T}$。因此 GEMM 使用「NN」形式。
- **GELU 變體。** GPT-2 使用 **tanh** 近似。精確的 erf GELU 差異可達
  約 $10^{-3}$，正好接近容許誤差。
- **沒有因果遮罩。** 此處參考實作計算雙向注意力，與 GPT-2 推論不同。
- **暫存記憶體**：每次呼叫配置 $S(3d + 3d + 4d)$ 個 float 作為暫存空間。
  `xn` 緩衝區會由兩個 LayerNorm 共用。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-3`
通過。較長的序列由本題的 `cuemu_max_elements` 設定支援。

## 相關內容

- [LLaMA Transformer 區塊](../093-llama-transformer-block/)、[DiT 區塊](../116-dit-block/)、
  [多頭注意力](../012-multi-head-attention/)、[Layer Norm](../113-layer-normalization/)。
