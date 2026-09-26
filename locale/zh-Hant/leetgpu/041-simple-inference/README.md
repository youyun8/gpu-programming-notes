---
title: 簡易推論
platform: LeetGPU
upstream: easy/41_simple_inference
url: https://leetgpu.com/challenges/simple-inference
difficulty: easy
tags: [pytorch, linear-layer, gemm]
status: solved
---

# 簡易推論

**平台：** LeetGPU · **難度：** 簡單 · [題目敘述](https://leetgpu.com/challenges/simple-inference)

## 問題

對一個批次執行已訓練 `torch.nn.Linear` 的正向傳播：`input` 為
$B \times d_{\text{in}}$，而 `output` 必須接收
$B \times d_{\text{out}}$
（$B, d_{\text{in}}, d_{\text{out}} \le 1000$；
效能評測使用 $B = 1000$；容許誤差為 `1e-5`）。此挑戰只提供 PyTorch
起始程式，因此解答為 `solution.py`。重點是將該層表示成**一次融合的函式庫
呼叫**，並直接寫入指定的輸出緩衝區。

## 公式

$$
Y = X W^{\mathsf T} + \mathbf 1\,\mathbf b^{\mathsf T}, \qquad Y_{ro} = \sum_{i=0}^{d_{\text{in}}-1} X_{ri}\, W_{oi} + b_o
$$

| 符號 | 意義 |
|---|---|
| $B$ | 批次大小（$X$ 與 $Y$ 的列數） |
| $d_{\text{in}},\ d_{\text{out}}$ | 輸入與輸出特徵大小 |
| $X$ | 輸入批次，$B \times d_{\text{in}}$ |
| $W$ | 權重（`model.weight`），$d_{\text{out}} \times d_{\text{in}}$，採用 PyTorch 的 `[out, in]` 配置 |
| $\mathbf b$ | 偏置（`model.bias`），長度為 $d_{\text{out}}$；可能不存在 |
| $\mathbf 1$ | 全為 1 的欄向量（將偏置廣播至每一列） |
| $Y$ | 輸出，$B \times d_{\text{out}}$ |

## 方法

```python
with torch.inference_mode():
    if model.bias is None:
        torch.matmul(input, model.weight.t(), out=output)
    else:
        torch.addmm(model.bias, input, model.weight.t(), out=output)
```

- `torch.addmm(b, X, Wᵀ)` 會透過一次 cuBLAS GEMM 計算
  $\mathbf b + XW^{\mathsf T}$，對廣播偏置使用 $\beta = 1$，
  而不是先執行 GEMM，再啟動另一個加法核心函式。
- `W.t()` 是一個交換步幅的檢視。cuBLAS 會直接使用轉置運算元
  （「TN」配置），因此不會複製任何資料。
- `out=output` 會寫入測試框架的張量，避免產生暫存值與額外複製。
- `inference_mode()` 會停用自動微分追蹤與版本計數器。

## 成本分析

$$
W_{\text{flop}} = 2B\,d_{\text{in}}\,d_{\text{out}}, \qquad Q \approx 4\,(B d_{\text{in}} + d_{\text{in}}d_{\text{out}} + B d_{\text{out}})
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{flop}}$ | GEMM 的 FLOP 數 |
| $Q$ | 必要的位元組數（讀取 $X$、$W$ 並寫入 $Y$；偏置可忽略） |

當各維度均為 1000 時：2 GFLOP、12 MB，是受運算限制的 GEMM。
cuBLAS 能以接近峰值的效能執行（若啟用，會使用 TF32 Tensor Core）。

## 常見陷阱

- **`bias=False`** 的模型會讓 `model.bias is None`，此時 `addmm` 會失敗。
- **呼叫 `model(input)` 再執行 `output.copy_()`** 雖然結果正確，
  卻會配置並複製額外的 $B \times d_{\text{out}}$ 張量。

## 驗證

已透過執行器的 PyTorch 路徑測試所有 LeetGPU 案例，包括有偏置與無偏置。

## 相關內容

- [矩陣乘法](../002-matrix-multiplication/)（GEMM 內部執行的運算）、
  [LoRA 線性層](../085-lora-linear/)。
