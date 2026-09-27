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

## 問題

線性層後接 Swish 與縮放：
$\text{out} = s\cdot\operatorname{swish}(xW^{\mathsf T} + \mathbf{b})$。
$x$ 的大小為 $B\times\text{in}$，$W$ 的大小為 $\text{out}\times\text{in}$
（例如 $B = 128$、in = 1024、out = 512、$s = 2$）。
檢查條件為 `rtol = 3e-4`、`atol = 1e-5`。

## 公式

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

## 方法

使用 `kTransB = true` 的共用核心，並在尾聲中加入偏置、套用
$z\,\sigma(z)$，再乘以 $s$。

### 共用 SGEMM 核心

Tensara 上所有矩陣乘法頁面都使用同一個以暫存器分塊的 FP32 核心
（`gemmKernel<kTransB, Epi>`）：

1. **區塊圖塊 $64\times64$**，使用 256 個執行緒；每個執行緒負責輸出中
   列為 `ty + 16i`、欄為 `tx + 16j` 的 $4\times4$ 小區塊。
   步距 16 的配置使 warp 的每個儲存指令都會存取連續 16 欄（合併存取），
   並讓共用記憶體讀取不發生衝突。
2. **寬度 16 的 K 切片。** 每個切片中，區塊會將 $A$ 的
   $64\times16$ 面板（轉置儲存為 `a_tile[k][m]`）及 $B$ 的
   $16\times64$ 面板複製到共用記憶體（每列填補 4 個 float），接著同步。
3. **在暫存器中計算內積。** 對 16 個 $k$ 值中的每一個，執行緒會從共用記憶體
   載入 4 個 $A$ 值與 4 個 $B$ 值，並執行 $4\times4 = 16$ 次 FMA（外積）。
4. **尾聲函式物件。** 累加器在唯一一次儲存前會經過 `epi(v, row, col)`。
   偏置、啟用函式、縮放或逐元素乘法都在此融合，因此乘積不必往返 DRAM。
5. `kTransB = true` 會將 $B$ 當作 $N\times K$（「NT」，即 `nn.Linear`
   的權重配置）讀取，並在暫存時轉置。

階層中各層級的資料重用率：

$$
I_{\text{L2}} = \frac{2\,T_M T_N T_K}{4\,T_K\,(T_M + T_N)} = \frac{T_M T_N}{2\,(T_M + T_N)} = 16\ \tfrac{\text{flop}}{\text{byte}}, \qquad
I_{\text{smem}} = \frac{2 \cdot r_M r_N}{4\,(r_M + r_N)} = 1\ \tfrac{\text{flop}}{\text{byte}}
$$

| 符號 | 意義 |
|---|---|
| $T_M, T_N, T_K$ | 區塊圖塊：64、64、16 |
| $r_M, r_N$ | 每個執行緒的暫存器圖塊：4 × 4 |
| $I_{\text{L2}}$ | 從 L2/DRAM 載入共用記憶體時，每位元組對應的浮點運算數 |
| $I_{\text{smem}}$ | 從共用記憶體讀取時，每位元組對應的浮點運算數（每 8 次載入執行 16 次 FMA） |

此核心可達 FP32 峰值約 40–60%。後續步驟見
[SGEMM 教學](../../tutorials/04-tiled-matmul.md)：使用 $128\times128$ 圖塊、
每執行緒 $8\times8$、`float4` 共用記憶體載入、雙緩衝 `cp.async` 暫存，
以及在容許誤差允許時使用張量核心（TF32）。

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

## 注意事項

- **嚴格的 `atol = 1e-5`**：當 $z$ 很小時輸出也很小；偏置 → swish → 縮放
  的順序必須與參考實作一致。
- **`const float scaling_factor`** 位於輸出指標之前。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [矩陣乘法 + Swish + 縮放](../matmul-swish-scaling/)、[Swish](../swish/)、
  [GEMM + ReLU](../gemm-relu/)、LeetGPU [SwiGLU MLP 區塊](../../leetgpu/084-swiglu-mlp-block/)。
