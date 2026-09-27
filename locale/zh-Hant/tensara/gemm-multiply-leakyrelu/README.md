---
title: GEMM 搭配逐元素相乘與 LeakyReLU
platform: Tensara
upstream: gemm-multiply-leakyrelu
url: https://tensara.org/problems/gemm-multiply-leakyrelu
difficulty: medium
tags: [matmul, sgemm, fusion, activation]
status: solved
---

# GEMM 搭配逐元素相乘與 LeakyReLU

**平台：** Tensara · **難度：** medium · [題目說明](https://tensara.org/problems/gemm-multiply-leakyrelu)

## 題意

計算 $O = \operatorname{LeakyReLU}_\alpha\bigl((AB)\odot C\bigr)$，其中
$A$ 的大小為 $M\times K$、$B$ 為 $K\times N$，逐元素乘數 $C$ 為
$M\times N$（各維大小為 512 … 1024）。檢查條件為
`rtol = 3e-4`、`atol = 1e-4`。

## 圖解

![GEMM、逐元素相乘與 LeakyReLU，全部在暫存器中對分塊完成](figure.svg)

輸出分塊的計算與一般 GEMM 相同；右側方框是 epilogue：先乘上 C 再套用 LeakyReLU，最後只寫出一次。

## 數學表述

$$
G_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}, \qquad
H_{ij} = G_{ij}\,C_{ij}, \qquad
O_{ij} = \begin{cases} H_{ij}, & H_{ij} \ge 0 \\ \alpha H_{ij}, & H_{ij} < 0 \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $A, B$ | GEMM 運算元，以列為主 |
| $G$ | 乘積 $AB$（保留在暫存器中） |
| $C$ | 逐元素乘數，$M\times N$ |
| $\odot$ | Hadamard（逐元素）乘積 |
| $H$ | $G\odot C$ |
| $\alpha$ | LeakyReLU 斜率 |
| $O$ | 輸出，$M\times N$ |

## 解題思路

使用共享核心與 epilogue `MulLeakyEpi{C, ld, alpha}`：它從與儲存相同的
合併存取位置載入 $C_{ij}$，執行乘法並套用斜率。$C$ 恰好讀取一次，
$G$ 與 $H$ 都不會寫入記憶體。

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
W = 2MNK, \qquad Q_{\min} = 4\,(MK + KN + 2MN)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數（一次 FMA = 2 次浮點運算） |
| $Q_{\min}$ | 必要的 DRAM 位元組數（每個運算元讀取一次，輸出寫入一次） |
| $F$ | FP32 峰值（目前 GPU 為數十 TFLOP/s） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | Roofline 模型的下限 |

矩陣很小（$1024^3$ 只有 2.1 GFLOP），因此 $64\times64$ 磚塊會產生
$16\times16 = 256$ 個區塊，每個 SM 約分到 2 個；延遲與尾端效應會很
明顯，使用 split-K 或較小的磚塊可更充分利用 GPU。

## 常見陷阱

- **參數順序**：`(A, B, C, alpha, output, M, N, K)`；`alpha` 位於輸入
  和輸出之間。
- **`>=` 與 `>`**：在 $H = 0$ 時兩者都得到 0，因此都可以使用。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 延伸閱讀

- [GEMM + ReLU](../gemm-relu/)、[Leaky ReLU](../leaky-relu/)、
  [MatMul + Swish + Scaling](../matmul-swish-scaling/)。
