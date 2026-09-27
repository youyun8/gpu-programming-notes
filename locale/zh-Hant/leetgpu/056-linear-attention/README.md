---
title: 線性自注意力
platform: LeetGPU
upstream: hard/56_linear_attention
url: https://leetgpu.com/challenges/linear-self-attention
difficulty: hard
tags: [attention, linear-attention, associativity]
status: solved
---

# 線性自注意力

**平台：** LeetGPU · **難度：** 困難 · [題目說明](https://leetgpu.com/challenges/linear-self-attention)

## 問題

使用特徵映射 $\phi(x) = \operatorname{ELU}(x) + 1$ 的線性注意力
（Katharopoulos 等人，〈Transformers are RNNs〉），作用於
$Q, K, V \in \mathbb R^{M\times d}$（$M \le 10^4$、$d \le 128$，
值域為 $[-100, 100]$；基準測試為 $M = 10^4$；容許誤差 `1e-4`）。
將 $\exp(\mathbf q\cdot\mathbf k)$ 換成
$\phi(\mathbf q)\cdot\phi(\mathbf k)$，可利用**結合律**把
$O(M^2 d)$ 注意力降為 $O(Md^2)$。

## 公式

$$
O_{i,:} = \frac{\phi(\mathbf q_i)^{\mathsf T}\, S}{\phi(\mathbf q_i)^{\mathsf T}\, \mathbf z}, \qquad
S = \sum_{j=0}^{M-1} \phi(\mathbf k_j)\,\mathbf v_j^{\mathsf T} = \phi(K)^{\mathsf T} V, \qquad
\mathbf z = \sum_{j=0}^{M-1} \phi(\mathbf k_j)
$$

$$
\phi(x) = \operatorname{ELU}(x) + 1 = \begin{cases} x + 1, & x > 0 \\ e^{x}, & x \le 0 \end{cases}\quad(\text{elementwise})
$$

| 符號 | 意義 |
|---|---|
| $M$ | 序列長度 |
| $d$ | 特徵維度 |
| $\mathbf q_i,\ \mathbf k_j,\ \mathbf v_j$ | $Q$、$K$、$V$ 的列（長度為 $d$ 的欄向量） |
| $\phi$ | 逐元素套用的正值特徵映射 |
| $S$ | $d\times d$「鍵值狀態」 |
| $\mathbf z$ | 長度為 $d$ 的正規化狀態 |
| $O_{i,:}$ | 輸出列 $i$ |

**為何是線性的。** 標準注意力計算
$\sum_j \frac{\operatorname{sim}(\mathbf q_i, \mathbf k_j)}{\sum_{j'}\operatorname{sim}}\mathbf v_j$。
當 $\operatorname{sim}(\mathbf q, \mathbf k) = \phi(\mathbf q)^{\mathsf T}\phi(\mathbf k)$
時，可將查詢提出對 $j$ 的總和：
$\sum_j \phi(\mathbf q_i)^{\mathsf T}\phi(\mathbf k_j)\mathbf v_j^{\mathsf T} = \phi(\mathbf q_i)^{\mathsf T}\bigl(\sum_j \phi(\mathbf k_j)\mathbf v_j^{\mathsf T}\bigr)$。
內部總和只需計算一次，所有查詢都可共用。

## 方法

1. **`kvState`**：$S_{ab}$ 的每個項目各使用一個執行緒（另有 $d$ 個執行緒處理
   $z_a$）。每個執行緒走訪所有 $M$ 列，以 float64 累加
   $\phi(K_{ra})\,V_{rb}$。固定 $r$ 時，相鄰執行緒（連續的 $b$）
   讀取連續的 $V_{rb}$，可合併存取；而一列執行緒使用的
   $\phi(K_{ra})$ 是相同位址，可進行廣播。
2. **`applyState`**：每個查詢列 $i$ 使用一個區塊。先在共享記憶體中計算
   $\phi(\mathbf q_i)$，再由執行緒 0 計算一次分母
   $\phi(\mathbf q_i)\cdot\mathbf z$。執行緒 $b$ 計算
   $\sum_a \phi(q_{ia}) S_{ab}$ 並進行除法。

步驟 1 必須使用 float64：最高為 100 的輸入會使 $\phi(k)$ 最高達 101，
而 $S$ 會加總 $10^4$ 個量級約為 $10^4$ 的乘積。

## 成本分析

$$
W_{\text{linear}} = 2Md^2 + 2Md^2 + O(Md), \qquad W_{\text{softmax attn}} = 4M^2 d, \qquad \frac{W_{\text{softmax}}}{W_{\text{linear}}} = \frac{M}{d}
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{linear}}$ | FLOP 數：建立 $S$（$2Md^2$）並套用至所有查詢（$2Md^2$） |
| $W_{\text{softmax attn}}$ | 供比較的標準注意力 FLOP 數 |

當 $M = 10^4$、$d = 128$ 時，線性注意力便宜約 78 倍。狀態建立核心只有
$d^2 + d \approx 16$K 個執行緒，每個都要走訪 $10^4$ 列。分割 M
的歸約（每個區塊建立部分狀態，再加總）可提高平行度。

## 注意事項

- **$\phi$ 必須為正值**，如此分母才會是正值。ELU + 1 在所有位置皆為正值。
- **$e^{x}$ 不可能溢位**，因為指數分支只會在 $x \le 0$ 時執行。
- **大輸入時 $S$ 的精確度**（如上所述）。

## 驗證

在 [cuemu](../../tools/cuemu/README.md) 上，所有 LeetGPU 測試案例皆以
`1e-4` 通過，包括 $M = 1$ 與 $d = 1$。

## 相關內容

- [Softmax 注意力](../006-softmax-attention/)、[SSM 選擇性掃描](../094-ssm-selective-scan/)
  （另一種線性時間序列模型）、[線性遞迴](../082-linear-recurrence/)。
