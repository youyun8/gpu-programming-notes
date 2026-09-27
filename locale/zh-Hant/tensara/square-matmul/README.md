---
title: 方形矩陣乘法
platform: Tensara
upstream: square-matmul
url: https://tensara.org/problems/square-matmul
difficulty: medium
tags: [matmul, sgemm, register-blocking]
status: solved
---

# 方形矩陣乘法

**平台：** Tensara · **難度：** 中等 · [題目敘述](https://tensara.org/problems/square-matmul)

## 問題

計算兩個 $N\times N$ 矩陣的 FP32 乘積，其中 $N$ 從 4096 到 9216。
檢查條件為 `rtol = 2e-4`、`atol = 5e-3`。這是 $M = N = K$ 的
[矩陣乘法](../matrix-multiplication/)；$N = 6144, 7168, 9216$ 都是
64 的倍數，因此不會出現不完整區塊。

## 公式

$$
C_{ij} = \sum_{k=0}^{N-1} A_{ik}\,B_{kj}
$$

| 符號 | 意義 |
|---|---|
| $A, B$ | 輸入，$N\times N$，列優先 |
| $C$ | 輸出 $AB$ |
| $N$ | 矩陣邊長 |

長度為 $N$ 的點積，其累積捨入誤差上限為

$$
\bigl\lvert \hat{C}_{ij} - C_{ij} \bigr\rvert \le \gamma_N \sum_k \lvert A_{ik}\rvert\,\lvert B_{kj}\rvert, \qquad \gamma_N = \frac{N u}{1 - N u}
$$

| 符號 | 意義 |
|---|---|
| $\hat{C}_{ij}$ | 計算所得（已取整）的值 |
| $u$ | 單位捨入誤差；fp32 為 $2^{-24}$ |
| $\gamma_N$ | 最壞情況的成長係數（一般誤差約以 $\sqrt{N}u$ 成長） |

這說明了為何 $N = 9216$ 時需要 `atol = 5e-3`，即使核心本身正確：
加總順序與 cuBLAS 不同會改變最後幾個位元。

## 方法

使用 `M = N = K = n` 與 `NoEpi` 的共用核心。

### 共用 SGEMM 核心

Tensara 上所有矩陣乘法頁面都使用同一個暫存器分塊 FP32 核心
（`gemmKernel<kTransB, Epi>`）：

1. **區塊分塊為 $64\times64$**，含 256 個執行緒；每個執行緒負責
   `ty + 16i` 列、`tx + 16j` 欄的一塊 $4\times4$ 輸出。步幅 16 的配置
   讓 warp 的每個儲存指令命中 16 個連續欄（合併存取），也讓共享記憶體
   讀取不會發生 bank 衝突。
2. **K 切片寬度為 16。** 每個切片由區塊將一個 $64\times16$ 的 $A$ 面板
   （轉置儲存成 `a_tile[k][m]`）以及一個 $16\times64$ 的 $B$ 面板複製到
   共享記憶體（每列填補 4 個 float），再同步。
3. **在暫存器內計算內積。** 對 16 個 $k$ 值中的每一個，執行緒從共享
   記憶體載入 4 個 $A$ 值與 4 個 $B$ 值，執行 $4\times4 = 16$ 次 FMA
   （外積）。
4. **結尾函式物件。** 累加器在唯一一次儲存前，會經過
   `epi(v, row, col)`。偏差、啟用函式、縮放或逐元素乘法會在此融合，
   因此乘積不必往返 DRAM。
5. `kTransB = true` 會把 $B$ 當成 $N\times K$（「NT」，
   `nn.Linear` 的權重配置）讀取，並在暫存時轉置。

階層各層級的資料重用如下：

$$
I_{\text{L2}} = \frac{2\,T_M T_N T_K}{4\,T_K\,(T_M + T_N)} = \frac{T_M T_N}{2\,(T_M + T_N)} = 16\ \tfrac{\text{flop}}{\text{byte}}, \qquad
I_{\text{smem}} = \frac{2 \cdot r_M r_N}{4\,(r_M + r_N)} = 1\ \tfrac{\text{flop}}{\text{byte}}
$$

| 符號 | 意義 |
|---|---|
| $T_M, T_N, T_K$ | 區塊分塊：64、64、16 |
| $r_M, r_N$ | 每個執行緒的暫存器分塊：4 × 4 |
| $I_{\text{L2}}$ | 從 L2/DRAM 載入共享記憶體時，每位元組的 Flops |
| $I_{\text{smem}}$ | 從共享記憶體讀取時，每位元組的 Flops（每 8 次載入執行 16 次 FMA） |

此作法可達 FP32 峰值的約 40–60%。後續改進方向見
[SGEMM 教學](../../tutorials/04-tiled-matmul.md)：使用 $128\times128$
分塊、每執行緒 $8\times8$、`float4` 共享載入、雙緩衝 `cp.async` 暫存，
最後在容許誤差允許時使用張量核心（TF32）。

## 成本分析

$$
W = 2N^3, \qquad Q_{\min} = 4\,(3N^2)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數（一次 FMA = 2 flops） |
| $Q_{\min}$ | 必要的 DRAM 位元組數（每個運算元讀取一次、輸出寫入一次） |
| $F$ | FP32 峰值（目前 GPU 為數十 TFLOP/s） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | Roofline 模型的時間下限 |

$N = 9216$ 時，$W = 1.57$ TFLOP，明顯受運算能力限制。

## 注意事項

- **只有一個大小引數**：`solution(a, b, c, n)`。
- **捨入**：請參閱上方的誤差界限；不要為了「符合」cuBLAS 而硬調核心的
  累加順序，這是不可能的。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [矩陣乘法](../matrix-multiplication/)、[對稱矩陣乘法](../symmetric-matmul/)、
  [下三角矩陣乘法](../lower-trig-matmul/)。
