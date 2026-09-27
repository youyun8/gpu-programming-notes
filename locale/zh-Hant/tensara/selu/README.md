---
title: SELU
platform: Tensara
upstream: selu
url: https://tensara.org/problems/selu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# SELU

**平台：** Tensara · **難度：** 簡單 · [題目敘述](https://tensara.org/problems/selu)

## 問題

對 $M\times N$ 的 float32 矩陣逐元素套用縮放指數線性單元，結果須符合
`torch.selu`。測試矩陣從 $4096\times4096$ 到 $8192\times8192$
（最多 6,700 萬個元素）。檢查條件為 `rtol = 1e-4`、`atol = 8e-5`。

## 公式

$$
C_{ij} = \lambda \begin{cases} x, & x > 0 \\ \alpha\,(e^{x} - 1), & x \le 0 \end{cases}, \qquad x = A_{ij}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，列優先 |
| $C$ | 相同形狀的輸出矩陣 |
| $x$ | 一個輸入元素 $A_{ij}$ |
| $\alpha$ | $1.6732632423543772$（`kAlpha`） |
| $\lambda$ | $1.0507009873554805$（`kScale`） |

這些常數是 Klambauer 等人推導出的固定點（*Self-Normalizing Neural
Networks*，2017）：若輸入的平均值為 0、變異數為 1，輸出亦然，

$$
\mathbb{E}[\operatorname{SELU}(z)] = 0, \qquad \operatorname{Var}[\operatorname{SELU}(z)] = 1, \qquad z \sim \mathcal{N}(0, 1)
$$

| 符號 | 意義 |
|---|---|
| $z$ | 標準常態隨機變數 |
| $\mathbb{E}, \operatorname{Var}$ | 期望值與變異數 |

## 方法

所有 Tensara 逐元素問題都使用同一種核心結構：

1. **`float4` 網格步進迴圈。** 將緩衝區視為 $\lfloor n/4 \rfloor$
   個 16 位元組向量；每次迭代載入一個 `float4`，對四個 lane 套用純量函式，
   再儲存一個 `float4`。`cudaMalloc` 傳回以 256 位元組對齊的指標，
   因此重新解讀型別是安全的。
2. **純量尾端處理**最後的 $n \bmod 4$ 個元素。
3. **啟動** 256 執行緒的區塊，最多 4096 個區塊；網格步進迴圈可涵蓋任意
   大小，而 4096 × 256 個執行緒足以用滿 DRAM。
4. 此函式是 `__forceinline__` 裝置函式，因此除函式本身的選擇操作外，
   迴圈主體沒有分支。

程式是在選擇操作外套用縮放的 ELU 核心：
`kScale * (x > 0 ? x : kAlpha * expm1f(x))`。

## 成本分析

$$
n = MN, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 元素數量 |
| $Q$ | 必要的 DRAM 流量：讀取一次輸入並寫入一次輸出 |
| $\beta$ | DRAM 頻寬（目前資料中心 GPU 約為 2–3 TB/s） |
| $T_{\min}$ | 核心執行時間受頻寬限制的下限 |

對 $8192\times8192$ 而言，$Q = 537$ MB；在 2 TB/s 時約為 0.27 ms。

## 注意事項

- **常數要保留完整 float 精度**：1.67 與 1.05 等截短值無法通過容許誤差。
- 接近 0 時使用 **`expm1f`**，而非 `expf(x) - 1`，做法與
  [ELU](../elu/) 相同。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [ELU](../elu/)、[Leaky ReLU](../leaky-relu/)、[GELU](../gelu/)。
