---
title: 矩陣乘法搭配 Swish 啟用函式
platform: Tensara
upstream: matmul-swish
url: https://tensara.org/problems/matmul-swish
difficulty: medium
tags: [matmul, sgemm, fusion, linear-layer]
status: solved
---

# 矩陣乘法搭配 Swish 啟用函式

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/matmul-swish)

## 題意

線性層後接 Swish 與縮放：
$\text{out} = s\cdot\operatorname{swish}(xW^{\mathsf T} + \mathbf{b})$。
$x$ 的大小為 $B\times\text{in}$，$W$ 的大小為 $\text{out}\times\text{in}$
（例如 $B = 128$、in = 1024、out = 512、$s = 2$）。
檢查條件為 `rtol = 3e-4`、`atol = 1e-5`。

## 圖解

![線性層 + Swish + 縮放：z = x Wᵀ + b，out = s · z · σ(z)](figure.svg)

GEMM 使用 nn.Linear 的轉置權重配置；偏差、Swish 激活與縮放都融合在 epilogue 中。

## 數學表述

$$
z_{rc} = \sum_{n=0}^{\text{in}-1} x_{rn}W_{cn} + b_c, \qquad
\text{out}_{rc} = s\,z_{rc}\,\sigma(z_{rc}) = \frac{s\,z_{rc}}{1 + e^{-z_{rc}}}
$$

| 符號 | 意義 |
|---|---|
| $x$ | 輸入，$B\times\text{in}$ |
| $W$ | 權重，$\text{out}\times\text{in}$（`nn.Linear` 配置，因此 GEMM 為「NT」） |
| $\mathbf{b}$ | 偏置，長度為 out |
| $z$ | 線性輸出（保留在暫存器中） |
| $\sigma$ | 邏輯 Sigmoid |
| $s$ | `scaling_factor` |
| out | 結果，$B\times\text{out}$ |

## 解題思路

使用 `kTransB = true` 的共用核心，並在尾聲中加入偏置、套用
$z\,\sigma(z)$，再乘以 $s$。

### 共用 SGEMM 核心

Tensara 上所有矩陣乘法題目都使用同一個以暫存器分塊的 FP32 kernel
（`gemmKernel<kTransB, Epi>`）：

1. **區塊分塊 $64\times64$**，256 個執行緒；每個執行緒負責 $4\times4$ 個輸出，
   位於第 `ty + 16i` 列、第 `tx + 16j` 欄。步幅為 16 的配置讓一個 warp 的每道
   儲存指令都寫到 16 個連續欄位（合併存取），也讓共享記憶體的讀取沒有 bank 衝突。
2. **每次處理 16 個 $k$。** 每一步，區塊把 $A$ 的 $64\times16$ 面板（以
   `a_tile[k][m]` 轉置存放）與 $B$ 的 $16\times64$ 面板複製到共享記憶體
   （每列補 4 個 float），然後同步。
3. **在暫存器中做外積。** 對這 16 個 $k$ 值，每個執行緒從共享記憶體載入 4 個
   $A$ 值與 4 個 $B$ 值，執行 $4\times4 = 16$ 次 FMA。
4. **Epilogue 函式物件。** 累加結果在唯一一次寫出之前，會先經過
   `epi(v, row, col)`。偏差、激活函數、縮放或逐元素乘法都在這裡融合，因此乘積
   不必再經過一次 DRAM 往返。
5. `kTransB = true` 會把 $B$ 當成 $N\times K$ 讀取（「NT」，也就是 `nn.Linear`
   的權重配置），並在載入共享記憶體時完成轉置。

#### 資料重用

階層中每一層的資料重用率：

$$
I_{\text{L2}} = \frac{2\,T_M T_N T_K}{4\,T_K\,(T_M + T_N)} = \frac{T_M T_N}{2\,(T_M + T_N)} = 16\ \tfrac{\text{flop}}{\text{byte}}, \qquad
I_{\text{smem}} = \frac{2 \cdot r_M r_N}{4\,(r_M + r_N)} = 1\ \tfrac{\text{flop}}{\text{byte}}
$$

| 符號 | 意義 |
|---|---|
| $T_M, T_N, T_K$ | 區塊分塊：64、64、16 |
| $r_M, r_N$ | 每個執行緒的暫存器分塊：4 × 4 |
| $I_{\text{L2}}$ | 從 L2/DRAM 載入共享記憶體的每個位元組所對應的運算量 |
| $I_{\text{smem}}$ | 從共享記憶體讀取的每個位元組所對應的運算量（8 次載入對應 16 次 FMA） |

#### 能達到的效能

這個 kernel 約可達到 FP32 峰值的 40–60%。接下來的改進就是
[SGEMM 教學](../../tutorials/04-tiled-matmul.md)所介紹的內容：$128\times128$
分塊搭配每個執行緒 $8\times8$、`float4` 共享記憶體載入、以雙緩衝的 `cp.async`
預先載入，最後在容許誤差允許時改用張量核心（TF32）。

## 成本分析

$$
W = 2B\cdot\text{in}\cdot\text{out}, \qquad Q_{\min} = 4\,(B\cdot\text{in} + \text{out}\cdot\text{in} + B\cdot\text{out})\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算數（一次 FMA = 2 次浮點運算） |
| $Q_{\min}$ | 必要的 DRAM 位元組數（每個運算元讀取一次，輸出寫入一次） |
| $F$ | FP32 峰值（目前 GPU 為數十 TFLOP/s） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | Roofline 模型的時間下限 |

在 $128\times1024\times512$ 時：$W = 134$ MFLOP，只需數微秒。
網格只有 $8\times2 = 16$ 個 $64\times64$ 區塊，遠少於 SM 數量：
此尺寸受延遲限制，可用 split-K（沿歸約方向讓多個區塊處理同一輸出圖塊）改善。

## 常見陷阱

- **嚴格的 `atol = 1e-5`**：當 $z$ 很小時輸出也很小；偏置 → swish → 縮放
  的順序必須與參考實作一致。
- **`const float scaling_factor`** 位於輸出指標之前。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [矩陣乘法 + Swish + 縮放](../matmul-swish-scaling/)、[Swish](../swish/)、
  [GEMM + ReLU](../gemm-relu/)、LeetGPU [SwiGLU MLP 區塊](../../leetgpu/084-swiglu-mlp-block/)。
