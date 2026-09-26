---
title: 矩陣乘法
platform: Tensara
upstream: matrix-multiplication
url: https://tensara.org/problems/matrix-multiplication
difficulty: medium
tags: [matmul, sgemm, register-blocking, shared-memory]
status: solved
---

# 矩陣乘法

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/matrix-multiplication)

## 問題

計算一般 FP32 矩陣乘積 $C = AB$；$A$ 的大小為 $M\times K$，$B$ 的大小為
$K\times N$，採列優先配置，尺寸從 $4096^3$ 到 $8192^3$。
檢查條件為 `rtol = 2e-4`、`atol = 5e-3`，嚴格到無法使用 TF32
張量核心（10 位元尾數）：這是真正的 SGEMM。

## 公式

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik}\,B_{kj}, \qquad 0 \le i < M,\ 0 \le j < N
$$

| 符號 | 意義 |
|---|---|
| $A$ | 左運算元，$M\times K$，列優先（`input_a`） |
| $B$ | 右運算元，$K\times N$，列優先（`input_b`） |
| $C$ | 輸出，$M\times N$（`output_c`） |
| $M, N, K$ | $C$ 的列數、$C$ 的欄數、歸約長度 |
| $i, j, k$ | 列、欄及歸約索引 |

分塊會將總和改寫成圖塊乘積之和，而核心正是依此計算：

$$
C_{\mathcal{I}\mathcal{J}} = \sum_{s=0}^{\lceil K/T_K\rceil - 1} A_{\mathcal{I},\,\mathcal{K}_s}\,B_{\mathcal{K}_s,\,\mathcal{J}}
$$

| 符號 | 意義 |
|---|---|
| $\mathcal{I}, \mathcal{J}$ | 一個輸出圖塊的 64 列與 64 欄 |
| $\mathcal{K}_s$ | 第 $s$ 個切片，含 16 個歸約索引 |

## 方法

啟動核心時使用 `NoEpi`（恆等尾聲）與 `kTransB = false`。

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
W = 2MNK, \qquad Q_{\min} = 4\,(MK + KN + MN)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算數（一次 FMA = 2 次浮點運算） |
| $Q_{\min}$ | 必要的 DRAM 位元組數（每個運算元讀取一次，輸出寫入一次） |
| $F$ | FP32 峰值（目前 GPU 為數十 TFLOP/s） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | Roofline 模型的時間下限 |

在 $8192^3$ 時：$W = 1.1$ TFLOP、$Q_{\min} = 805$ MB。運算強度
$W/Q_{\min} \approx 1365$ flop/byte 遠高於任何 GPU 的效能轉折點，
因此核心受運算限制；速度取決於內部迴圈能多接近每個 lane 每週期一次 FMA。

## 注意事項

- **TF32**：啟用 `allow_tf32` 時，cuBLAS 可能使用 TF32；參考實作會停用
  autocast，而且在 $K$ 很大時，容許誤差不接受 TF32。
- **`int` 溢位**：$8192^2 = 2^{26}$ 個元素放得下，但位元組位移放不下；
  核心使用 `size_t` 相乘。
- **不完整圖塊**：超出範圍的列、欄與 $k$ 在載入時補零，儲存時略過。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [方陣乘法](../square-matmul/)、[3D 矩陣乘法](../matmul-3d/)、
  [GEMM + ReLU](../gemm-relu/)、
  LeetGPU [矩陣乘法](../../leetgpu/002-matrix-multiplication/)、
  教學 [04 – 分塊矩陣乘法](../../tutorials/04-tiled-matmul.md)。
