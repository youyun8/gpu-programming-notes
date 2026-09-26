---
title: GEMM 搭配偏差與 ReLU
platform: Tensara
upstream: gemm-relu
url: https://tensara.org/problems/gemm-relu
difficulty: medium
tags: [matmul, sgemm, fusion, linear-layer]
status: solved
---

# GEMM 搭配偏差與 ReLU

**平台：** Tensara · **難度：** medium · [題目說明](https://tensara.org/problems/gemm-relu)

## 問題

含 ReLU 的全連接層：
$C = \operatorname{ReLU}(AW^{\mathsf T} + \mathbf{b})$，其中 $A$ 的大小
為 $B\times N$（批次 × 輸入特徵）、$W$ 為 $M\times N$（PyTorch
`nn.Linear` 配置），偏差 $\mathbf{b}$ 的長度為 $M$。各維大小：
$B = 512 \dots 1024$、$N$ 最大為 8192、$M$ 最大為 2048。檢查條件為
`rtol = 3e-3`、`atol = 2e-4`。

## 公式

$$
Z_{rc} = \sum_{n=0}^{N-1} A_{rn}\,W_{cn} + b_c, \qquad C_{rc} = \max(Z_{rc}, 0)
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入啟用值，$B\times N$ |
| $W$ | 權重，$M\times N$：第 $c$ 列存放輸出特徵 $c$ 的權重 |
| $\mathbf{b}$ | 偏差，長度為 $M$ |
| $Z$ | 啟用前的值，$B\times M$（不會儲存） |
| $C$ | 輸出，$B\times M$ |
| $r, c, n$ | 批次列、輸出特徵、輸入特徵 |

以 $(c, n)$ 索引 $W_{cn}$ 表示乘積為 $AW^{\mathsf T}$：兩個運算元都沿
連續的 $n$ 軸讀取。以 BLAS 術語來說，這是「NT」GEMM。

## 方法

共享核心使用 `kTransB = true`（逐列讀取 $W$，並在共享磚塊中將它
轉置）以及 epilogue `BiasReluEpi`：`fmaxf(v + bias[c], 0)`。偏差是每欄
載入一個值，由 L1 提供。

### 共用 SGEMM 核心

Tensara 上所有矩陣乘法頁面都使用同一個以暫存器分塊的 FP32 核心
（`gemmKernel<kTransB, Epi>`）：

1. **區塊磚塊 $64\times64$**，使用 256 個執行緒；每個執行緒負責輸出中
   列為 `ty + 16i`、欄為 `tx + 16j` 的 $4\times4$ 區塊。跨距 16 的
   配置讓 warp 的每個儲存指令都寫入 16 個連續欄位（合併存取），並讓
   共享記憶體讀取不發生衝突。
2. **大小為 16 的 K 切片。** 每個切片中，區塊會將 $A$ 的
   $64\times16$ 面板（轉置儲存為 `a_tile[k][m]`）和 $B$ 的
   $16\times64$ 面板複製至共享記憶體（每列補上 4 個 float），然後同步。
3. **在暫存器中計算內積。** 對 16 個 $k$ 值中的每一個，執行緒都會從
   共享記憶體載入 $A$ 的 4 個值與 $B$ 的 4 個值，並執行
   $4\times4 = 16$ 次 FMA（外積）。
4. **Epilogue 函子。** 累加值在唯一一次儲存前會先經過
   `epi(v, row, col)`。偏差、啟用函式、縮放或逐元素乘法都在此融合，
   因此乘積不必來回存取 DRAM。
5. `kTransB = true` 會將 $B$ 視為 $N\times K$ 讀取（「NT」，
   `nn.Linear` 的權重配置），並在暫存時將它轉置。

階層中各層級的資料重複使用率：

$$
I_{\text{L2}} = \frac{2\,T_M T_N T_K}{4\,T_K\,(T_M + T_N)} = \frac{T_M T_N}{2\,(T_M + T_N)} = 16\ \tfrac{\text{flop}}{\text{byte}}, \qquad
I_{\text{smem}} = \frac{2 \cdot r_M r_N}{4\,(r_M + r_N)} = 1\ \tfrac{\text{flop}}{\text{byte}}
$$

| 符號 | 意義 |
|---|---|
| $T_M, T_N, T_K$ | 區塊磚塊：64、64、16 |
| $r_M, r_N$ | 每個執行緒的暫存器磚塊：4 × 4 |
| $I_{\text{L2}}$ | 從 L2/DRAM 載入共享記憶體的每位元組浮點運算次數 |
| $I_{\text{smem}}$ | 從共享記憶體讀取的每位元組浮點運算次數（每 8 次載入執行 16 次 FMA） |

這可達到 FP32 峰值的大約 40–60%。下一步是
[SGEMM 教學](../../tutorials/04-tiled-matmul.md)中介紹的方法：
使用 $128\times128$ 磚塊、每個執行緒負責 $8\times8$、以 `float4`
載入共享記憶體、使用雙緩衝的 `cp.async` 暫存，最後在容許誤差允許時
使用張量核心（TF32）。

## 成本分析

$$
W = 2BNM, \qquad Q_{\min} = 4\,(BN + MN + BM)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數（一次 FMA = 2 次浮點運算） |
| $Q_{\min}$ | 必要的 DRAM 位元組數（每個運算元讀取一次，輸出寫入一次） |
| $F$ | FP32 峰值（目前 GPU 為數十 TFLOP/s） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | Roofline 模型的下限 |

不融合時（先 GEMM，再加偏差，最後 ReLU），會對 $B\times M$ 個 float
增加兩趟額外的讀寫，也就是 $4\cdot 4BM$ 位元組；在
$1024\times2048$ 時為 34 MB，約需 17 µs，相較之下 GEMM 本身約需
0.4 ms。

## 注意事項

- **$W$ 的配置**：將 $W$ 當作 $N\times M$ 會讀取錯誤元素；測試中的
  $W$ 並非全都是方陣，因此錯誤會很明顯。
- **偏差是依輸出欄** $c$ 套用，而非依列。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 相關內容

- [GEMM × LeakyReLU](../gemm-multiply-leakyrelu/)、
  [MatMul + Swish](../matmul-swish/)、[ReLU](../relu/)、
  LeetGPU [GEMM](../../leetgpu/022-gemm/)。
