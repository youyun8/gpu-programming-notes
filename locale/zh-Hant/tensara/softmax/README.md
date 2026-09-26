---
title: Softmax
platform: Tensara
upstream: softmax
url: https://tensara.org/problems/softmax
difficulty: medium
tags: [softmax, online-softmax, strided-reduction]
status: solved
---

# Softmax

**平台：** Tensara · **難度：** 中等 · [題目敘述](https://tensara.org/problems/softmax)

## 問題

對任意秩的 float32 張量，沿任意維度 `dim` 執行 Softmax；張量形狀以
含 `ndim` 個大小的陣列給定。測試同時包含連續歸約（$(4, 256, 256, 256)$
沿 dim 3）與跨步歸約（$(8, 1024, 1024)$ 沿 dim 1、$(256, 50, 50)$
沿 dim 0）。檢查條件為 `rtol = 2e-3`、`atol = 1e-4`。

## 公式

如 [Argmax](../argmax/) 一樣，將張量視為三個軸：

$$
O = \prod_{k<d} S_k, \qquad R = S_d, \qquad I = \prod_{k>d} S_k, \qquad
x[o, j, i] = \text{in}\bigl[(oR + j)I + i\bigr]
$$

$$
\text{out}[o, j, i] = \frac{e^{x[o,j,i] - m_{oi}}}{s_{oi}}, \qquad
m_{oi} = \max_{j} x[o, j, i], \qquad s_{oi} = \sum_{j=0}^{R-1} e^{x[o,j,i] - m_{oi}}
$$

| 符號 | 意義 |
|---|---|
| $S_k$ | 第 $k$ 軸的大小；$d$ 是 `dim` |
| $O, R, I$ | 外部大小、歸約長度、內部大小（歸約軸的步幅） |
| $x[o, j, i]$ | 外部索引 $o$、歸約索引 $j$、內部索引 $i$ 的元素 |
| $m_{oi}$ | 沿歸約軸的最大值 |
| $s_{oi}$ | 正規化值：平移後指數的總和 |

$m$ 與 $s$ 都由一次線上合併取得：
$(m_1, s_1)\oplus(m_2, s_2) = \bigl(M, s_1e^{m_1-M} + s_2e^{m_2-M}\bigr)$，
$M = \max(m_1, m_2)$（請參閱 [Log Softmax](../log-softmax/)）。

| 符號 | 意義 |
|---|---|
| $\oplus$ | 部分（最大值、總和）配對的結合性合併操作 |

## 方法

形狀陣列可能位於主機或裝置上，因此使用 `cudaMemcpyDefault` 複製；
$O$、$R$、$I$ 則在主機端計算。

- **$I = 1$**（連續列）：`softmaxRows`，每列由一個 warp 負責。各 lane
  合併跨步元素，以 shuffle 蝶形操作合併 32 組配對，再寫入
  $e^{x - m}/s$。每個元素讀取兩次、寫入一次。
- **$I > 1$**（跨步）：`softmaxStrided`，每個 $(o, i)$ 欄由一個執行緒
  負責。執行緒走訪 $j$，步幅為 $I$；相鄰執行緒的 $i$ 相鄰，所以每一步
  都是合併的 warp 存取。第二次走訪則寫入輸出。

## 成本分析

$$
Q = 12\,ORI\ \text{bytes (two reads, one write)}, \qquad \#\exp \approx 2\,ORI
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM（或 L2）位元組數 |
| #exp | 指數運算次數：線上走訪一次，寫入走訪一次 |

最大案例 $64\times128^3$ 的輸入為 537 MB；在 2 TB/s 下流量耗時約
0.8 ms。當 $R$ 很小時（例如 $(128, 10)$ 沿 dim 1），每列使用一個 warp
會閒置 32 個 lane 中的 22 個；一個 warp 處理多列會更有效，但這些案例
本身很小。

## 注意事項

- **欄數很少的跨步案例**：$(256, 50, 50)$ 沿 dim 0 只有 2500 欄，
  也就是 2500 個執行緒，因此 GPU 大多處於閒置狀態；可沿 $j$ 拆分歸約。
- **`shape` 指標的類型**：使用 `cudaMemcpyDefault`，絕不可在主機端
  直接取值。
- **穩定性**：減去最大值。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [Log Softmax](../log-softmax/)、[Argmax](../argmax/)、
  [縮放點積注意力](../scaled-dot-attention/)、
  LeetGPU [Softmax](../../leetgpu/005-softmax/)。
