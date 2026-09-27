---
title: 矩陣的 N 次方
platform: Tensara
upstream: matrix-power
url: https://tensara.org/problems/matrix-power
difficulty: medium
tags: [matmul, sgemm, exponentiation-by-squaring]
status: solved
---

# 矩陣的 N 次方

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/matrix-power)

## 題意

對 $512\times512$ 的 FP32 矩陣計算 $A^P$，其中
$P \in \{2, 4, 8\}$（程式碼可處理任何 $P \ge 0$）。
檢查條件為 `rtol = 1e-4`、`atol = 1e-3`，因此若要符合參考實作的捨入結果，
乘法順序很重要。

## 圖解

![以反覆平方求 A^P（P = 8）：只需 3 次 GEMM，而非 7 次](figure.svg)

連續平方三次得到 A²、A⁴、A⁸。對其他指數，則由 P 的二進位位數決定要把哪些平方相乘。

## 數學表述

$$
A^0 = I, \qquad A^{P} = \prod_{t\,:\,\beta_t = 1} A^{2^t}, \qquad P = \sum_{t} \beta_t\,2^t
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入矩陣，$n\times n$（$n = 512$） |
| $I$ | 單位矩陣 |
| $P$ | 指數 |
| $\beta_t$ | $P$ 的第 $t$ 個位元（0 或 1） |
| $A^{2^t}$ | 由重複平方取得：$A^{2^{t+1}} = A^{2^t}A^{2^t}$ |

二進位冪運算所需的次數為

$$
\#\text{GEMMs} = \lfloor \log_2 P \rfloor + \operatorname{popcount}(P) - 1 \quad \text{instead of } P - 1
$$

| 符號 | 意義 |
|---|---|
| $\operatorname{popcount}(P)$ | $P$ 中值為 1 的位元數 |

對 $P = 8$ 而言，只需平方 3 次，不必相乘 7 次。

## 解題思路

1. **特殊情況**：$P = 0$ 寫入單位矩陣；$P = 1$ 複製；$P = 2$
   執行一次乘法；$P = 3$ 計算 $(AA)A$。
2. **一般情況**依照 `torch.linalg.matrix_power` 的順序：
   從最低有效位元開始逐一處理 $P$ 的位元；$z$ 透過平方依序成為
   $A, A^2, A^4, \dots$；每遇到值為 1 的位元，就將累計結果乘以 $z$
   （或將結果初始化為 $z$）。四個暫存緩衝區（兩個用於 $z$、兩個用於結果）
   交替使用，使乘積不會寫入自己的輸入。
3. 每次乘法都使用下述 $64\times64$、以暫存器分塊的 SGEMM
   （針對方陣特製的本機複本）。

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
W = 2n^3\cdot\#\text{GEMMs}, \qquad Q_{\min} = 4\,(2n^2)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算數（一次 FMA = 2 次浮點運算） |
| $Q_{\min}$ | 必要的 DRAM 位元組數（每個運算元讀取一次，輸出寫入一次） |
| $F$ | FP32 峰值（目前 GPU 為數十 TFLOP/s） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | Roofline 模型的時間下限 |

在 $n = 512$ 時，一次 GEMM 由 64 個區塊執行 0.27 GFLOP，區塊數少於 SM，
因此每次 GEMM 都受延遲限制（數十 µs）。相依鏈是序列式的；
只有減少 GEMM 次數（二進位冪運算）才有幫助。

## 常見陷阱

- **別名問題**：`matmul(z, z, z)` 會在讀取 $z$ 時覆寫它，因此需要交替緩衝區。
- **乘法順序**：$A^4 = (A^2)^2$ 與 $((AA)A)A$ 的捨入結果不同；
  仿照 PyTorch 演算法可讓誤差充分落在容許範圍內。
- **$P = 0$** 傳回 $I$，而非 $A$。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [方陣乘法](../square-matmul/)、LeetGPU [矩陣次方](../../leetgpu/037-matrix-power/)。
