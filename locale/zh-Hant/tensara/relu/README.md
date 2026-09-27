---
title: ReLU
platform: Tensara
upstream: relu
url: https://tensara.org/problems/relu
difficulty: easy
tags: [elementwise, activation, float4]
status: solved
---

# ReLU

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/relu)

## 題意

對 $M\times N$ float32 矩陣逐元素套用 ReLU，結果需符合 `torch.relu`。測試矩陣從 $4096\times4096$ 到 $8192\times8192$（最多 6,700 萬個元素）。檢查條件為 `rtol = 6e-5`、`atol = 3e-5`。這是最簡單的頻寬效能測試：運算只需一次 `max`。

## 圖解

![矩陣上的 ReLU：最簡單的頻寬基準測試](figure.svg)

M × N 矩陣的每個元素各自獨立映射，因此 kernel 只是一連串 float4 的載入與儲存。

## 數學表述

$$
C_{ij} = \operatorname{ReLU}(A_{ij}) = \max(A_{ij}, 0)
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$M\times N$ float32，按列優先排列 |
| $C$ | 輸出矩陣，形狀相同 |
| $\operatorname{ReLU}$ | 修正線性單元 |

## 解題思路

所有 Tensara 逐元素問題都共用同一種核心形態：

1. **`float4` 網格跨步迴圈。**將緩衝區視為 $\lfloor n/4 \rfloor$ 個 16 位元組向量；每次迭代載入一個 `float4`，對四個 lane 套用純量函式，再儲存一個 `float4`。`cudaMalloc` 回傳按 256 位元組對齊的指標，因此重新解讀型別是安全的。
2. 最後 $n \bmod 4$ 個元素使用**純量尾端處理**。
3. **啟動**含 256 個執行緒的區塊，最多 4096 個區塊；網格跨步迴圈可涵蓋任何尺寸，而 4096 × 256 個執行緒足以讓 DRAM 飽和。
4. 函式是 `__forceinline__` 裝置函式，因此除了函式本身的選擇操作外，迴圈主體沒有分支。

`fmaxf(x, 0.0f)` 只需一個指令。整個核心就像帶有篩選條件的記憶體複製，因此速度完全取決於載入與儲存使用 DRAM 的效率：16 位元組存取、足夠的同時傳輸資料量（4096 × 256 個執行緒 × 16 B = 最多 16 MB 的未完成資料），且沒有重複流量。

## 成本分析

$$
n = MN, \qquad Q = 8\,n\ \text{bytes}, \qquad T_{\min} = \frac{Q}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $n$ | 元素數量 |
| $Q$ | 必要的 DRAM 流量：輸入各讀取一次，輸出寫入一次 |
| $\beta$ | DRAM 頻寬（目前資料中心 GPU 約為 2–3 TB/s） |
| $T_{\min}$ | 核心時間的頻寬下限 |

當尺寸為 $8192\times8192$ 時：$Q = 537$ MB，在 2 TB/s 下約需 0.27 ms。一般能達到的最佳效能約為標稱頻寬的 85–92 %。

## 常見陷阱

- **NaN 處理**：`fmaxf(NaN, 0) = 0`，但 `torch.relu(NaN) = NaN`。測試資料不含 NaN；若這點很重要，可寫成 `x > 0 ? x : 0`（遇到 NaN 也會回傳 0）或 `x < 0 ? 0 : x`（會傳播 NaN）。
- 啟動參數 `n, m` 分別是列數與欄數；只會使用兩者的乘積。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [Leaky ReLU](../leaky-relu/)、[向量加法](../vector-addition/)、LeetGPU [ReLU](../../leetgpu/021-relu/)。
