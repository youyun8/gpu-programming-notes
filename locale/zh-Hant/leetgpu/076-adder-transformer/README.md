---
title: 加法器 Transformer 推論
platform: LeetGPU
upstream: medium/76_adder_transformer
url: https://leetgpu.com/challenges/adder-transformer-inference
difficulty: medium
tags: [transformer, inference, decoding, kv-cache, rope]
status: solved
---

# 加法器 Transformer 推論

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/adder-transformer-inference)

## 問題

使用 *AdderBoard* 競賽中一個微型、手工設計的 transformer 執行貪婪
自迴歸推論。它有 **10 個參數**、隱藏大小 2 及一個注意力頭，可將兩個
10 位數相加。對一批各含 31 個數字 token 的提示執行 11 個解碼步驟，
輸出每一步的 logits，形狀為 $[\text{batch}, 11, 10]$（容許誤差 `1e-2`）。
模型本身很簡單，重點在於推論的組織方式：使用 **KV 快取**，且只計算
最後一個位置。

## 公式

模型（單層、pre-norm、$d = 2$、詞彙表為 $\{0..9\}$、共用嵌入）：

$$
e(t) = \begin{bmatrix} w_0 - w_1 t^2 \\ -t \end{bmatrix}, \qquad
\operatorname{UnitRMS}(\mathbf x) = \frac{\mathbf x}{\sqrt{\tfrac12(x_0^2 + x_1^2) + \varepsilon}}
$$

對位置 $p$ 的 token $t_p$，令
$\mathbf n_p = \operatorname{UnitRMS}(e(t_p))$：

$$
\mathbf q_p = R_{p\omega}\,\operatorname{UnitRMS}\!\begin{bmatrix} q_0 n_{p,0} \\ q_1 n_{p,0}\end{bmatrix}, \qquad
\mathbf k_p = R_{p\omega}\,\operatorname{UnitRMS}\!\begin{bmatrix} n_{p,0} \\ 0\end{bmatrix}, \qquad
v_p = v_0\, n_{p,1}, \qquad
R_\theta = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}
$$

最後一個位置 $L$ 對位置 $0..L$ 的注意力（因果），會加到隱藏維度 1：

$$
a_L = \frac{\sum_{j \le L} e^{\lambda\,\mathbf q_L\cdot\mathbf k_j - m}\, v_j}{\sum_{j\le L} e^{\lambda\,\mathbf q_L\cdot\mathbf k_j - m}}, \qquad
\mathbf h = e(t_L) + \begin{bmatrix}0\\ a_L\end{bmatrix}
$$

MLP（「進位」閘門）、最終正規化及共用權重的 logits：

$$
\mathbf m = \operatorname{UnitRMS}(\mathbf h), \quad
g_0 = \alpha m_0 + \gamma m_1, \quad g_1 = (\alpha - \gamma/1000)\,m_0 + \gamma m_1, \quad
h_1 \mathrel{+}= c\,\bigl(\operatorname{SiLU}(g_1) - \operatorname{SiLU}(g_0)\bigr) m_0
$$

$$
\mathbf f = \operatorname{UnitRMS}(\mathbf h)\odot\begin{bmatrix}\nu_0\\ \nu_1\end{bmatrix}, \qquad
\text{logit}_t = \mathbf f\cdot e(t), \qquad t_{L+1} = \arg\max_t \text{logit}_t
$$

| 符號 | 意義 |
|---|---|
| $t$, $t_p$ | 數字 token（0–9）；位置 $p$ 的 token |
| $w_0, w_1$ | 嵌入參數（`w[0]`、`w[1]`） |
| $e(t)$ | 二維 token 嵌入（由於共用嵌入，它也是輸出投影） |
| $\varepsilon$ | $10^{-6}$ |
| $q_0, q_1, v_0$ | Query 與 value 參數（`w[2..4]`）；key 投影沒有參數 |
| $\omega$ | RoPE 角頻率 $2\pi/19$ |
| $R_\theta$ | 二維旋轉（二維注意力頭上的 RoPE） |
| $\lambda$ | 注意力縮放比例，模型常數：$2^{-1/2}\cdot\frac{\ln 10}{\sqrt2\,(\cos 0.3\omega - \cos 0.7\omega)}$ |
| $m$ | 用於 softmax 位移的最大 logit |
| $a_L$ | 注意力輸出（value 向量只有維度 0，並寫入隱藏維度 1） |
| $\alpha, \gamma, c$ | MLP 閘門與進位參數（`w[5..7]`） |
| $\nu_0, \nu_1$ | 最終 RMSNorm 權重（`w[8]`、`w[9]`） |
| $\text{logit}_t$ | 數字 $t$ 的輸出分數，每個解碼步驟都會寫入 |

## 方法

### 只需最後一個位置

參考實作在每個步驟都重新執行整個序列，並只保留最後一列 logits。在
**單層**模型中，位置 $j$ 的 key 與 value 只取決於自己的 token 與索引，
不取決於其他位置。因此，它們可在 token 加入時只計算一次並快取。這正是
LLM 推論的 **KV 快取**。每個解碼步驟因而只需：

1. 計算最後一個位置的 query（$O(1)$）；
2. 對 $\le 41$ 個已快取的 key 計算注意力（$O(L)$）；
3. 計算殘差、MLP、正規化及 10 個 logits（$O(1)$）。

### 每個序列一個執行緒

每個序列的完整狀態不超過 42 個位置 × 3 個 float。一個執行緒在暫存器／
區域記憶體中執行完整的 11 步貪婪迴圈：嵌入提示 token → 快取
$(\mathbf k_j, v_j)$ → 迴圈 {query、注意力、MLP、logits、argmax、附加}。
批次則提供平行度。

### 在數值上符合參考實作

logits 會送入 `argmax`，因此極小的數值差異也可能改變產生的數字，進而
改變之後每一步。核心因此遵循參考實作的運算順序。固定常數
（$\omega$、$\lambda$）會如同 Python 模組一樣，在主機端以 double 計算，
再以 float 傳入。

## 成本分析

$$
W \approx B\left(31\,c_{\text{append}} + \sum_{s=0}^{10}\bigl(c_{\text{step}} + 6\,(31 + s)\bigr)\right)
$$

| 符號 | 意義 |
|---|---|
| $B$ | 批次大小 |
| $c_{\text{append}}$ | 嵌入 token 並快取其 key/value 的成本（約 30 FLOP 加 sin/cos） |
| $c_{\text{step}}$ | 每步固定成本（query、MLP、正規化、10 個 logits，約 100 FLOP 加 3 次 exp） |
| $6(31+s)$ | 對目前長度計算注意力（內積、exp 及兩次累加） |

每個序列只有數千個 FLOP，幾乎可忽略。若不使用 KV 快取，而像參考實作
一樣每步重算所有位置，工作量約會增加 10 倍。

## 常見陷阱

- **貪婪回饋。** 步驟 $s$ 的錯誤會改變送入步驟 $s+1$ 的 token。
  完全一致的運算順序可避免後續結果分歧。
- **Value 向量配置。** $V$ 只有維度 0，而輸出投影會將它移到隱藏維度 1。
  此處完全照參考實作處理。
- **位置。** 提示與產生的 token 都使用以 0 為起點的絕對位置索引 $p$
  來計算 RoPE。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-2`
通過（比對全部 11 個步驟的 logits，也就間接檢查了每個產生的數字）。

## 相關內容

- [RoPE 嵌入](../061-rope-embedding/)、[推測式解碼](../087-speculative-decoding-verification/)、
  [INT8 KV 快取注意力](../096-int8-kv-cache-attention/)。
