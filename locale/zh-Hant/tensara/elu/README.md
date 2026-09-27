---
title: ELU
platform: Tensara
upstream: elu
url: https://tensara.org/problems/elu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# ELU

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/elu)

## 問題

將指數線性單元逐元素套用至 $M\times N$ float32 矩陣，執行期間會傳入
參數 $\alpha$（測試中為 1.0），結果須與 `F.elu(x, alpha)` 一致。測試
矩陣從 $4096\times4096$ 至 $8192\times8192$（最多 67 M 個元素）。
檢查條件為 `rtol = 1e-4`、`atol = 5e-5`。

## 公式

$$
C_{ij} = \operatorname{ELU}_\alpha(A_{ij}), \qquad
\operatorname{ELU}_\alpha(x) = \begin{cases} x, & x > 0 \\ \alpha\,(e^{x} - 1), & x \le 0 \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，以列為主 |
| $C$ | 輸出矩陣，形狀相同 |
| $x$ | 一個輸入元素 $A_{ij}$ |
| $\alpha$ | 飽和值：當 $x \to -\infty$ 時，$\operatorname{ELU}_\alpha(x) \to -\alpha$ |
| $e^x - 1$ | 使用 `expm1f` 計算 |

接近 $x = 0^-$ 時，直接計算 $e^x - 1$ 會有消去誤差：例如
$x = -10^{-6}$ 時，`expf(x)` 會捨入為 $1 - 2^{-24}\cdot k$，相減後只
留下少數正確位元。`expm1f` 會計算：

$$
\operatorname{expm1}(x) = x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots
$$

| 符號 | 意義 |
|---|---|
| $\operatorname{expm1}(x)$ | 不先建立 $e^x$ 而計算 $e^x - 1$，精確度約為 1 ulp |

## 方法

所有 Tensara 逐元素問題都採用同一種核心形狀：

1. **`float4` 網格跨步迴圈。** 將緩衝區視為 $\lfloor n/4 \rfloor$ 個
   16 位元組向量；每次迭代載入一個 `float4`、將純量函式套用至四條
   lane，再儲存一個 `float4`。`cudaMalloc` 回傳以 256 位元組對齊的
   指標，因此重新解讀型別是安全的。
2. **純量尾端**處理最後 $n \bmod 4$ 個元素。
3. **啟動** 256 執行緒的區塊，最多 4096 個區塊；網格跨步迴圈可處理
   任何大小，4096 × 256 個執行緒也足以讓 DRAM 達到飽和。
4. 函式是 `__forceinline__` 裝置函式，因此除了函式本身的選擇指令之外，
   迴圈主體沒有分支。

## 成本分析

$$
n = MN, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 元素數量 |
| $Q$ | 必要的 DRAM 流量：輸入讀取一次，輸出寫入一次 |
| $\beta$ | DRAM 頻寬（目前資料中心 GPU 約為 2–3 TB/s） |
| $T_{\min}$ | 核心執行時間的頻寬下限 |

在 $8192\times8192$ 時：$Q = 537$ MB，以 2 TB/s 計算約需 0.27 ms。
每個元素的 `expm1f` 需要數十條指令，仍遠低於 GPU 在搬移 8 位元組期間
可執行的約 100 條指令。

## 注意事項

- **接近零時的精確度**：使用 `expm1f`，不要使用 `expf(x) - 1.0f`。
- **條件**：線性分支的條件是 $x > 0$；在 $x = 0$ 時兩個分支都得到 0。
- 函式簽章中的**參數順序**為 `(input, output, n, m, alpha)`。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 相關內容

- [SELU](../selu/)、[Leaky ReLU](../leaky-relu/)、[ReLU](../relu/)、
  [GELU](../gelu/)。
