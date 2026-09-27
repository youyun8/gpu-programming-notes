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
