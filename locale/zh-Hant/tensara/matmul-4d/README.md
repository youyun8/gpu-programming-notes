---
title: 4D 張量矩陣乘法
platform: Tensara
upstream: matmul-4d
url: https://tensara.org/problems/matmul-4d
difficulty: hard
tags: [matmul, sgemm, einsum, reshape]
status: solved
---

# 4D 張量矩陣乘法

**平台：** Tensara · **難度：** 困難 · [題目說明](https://tensara.org/problems/matmul-4d)

## 題意

對形狀為 $B\times I\times J\times L$ 的 4D 張量 $A$ 與形狀為
$L\times K$ 的矩陣計算 `einsum("bijl,lk->bijk", A, B)`（最大案例：
$16\times256\times512\times256$ 乘以 $256\times768$）。
檢查條件為 `rtol = 2e-4`、`atol = 6e-4`。

## 圖解

![einsum("bijl,lk->bijk")：把自由索引 b、i、j 攤平成一個列索引](figure.svg)

A 的所有自由索引都位於被縮併的索引 l 之前，因此不必搬移任何資料就能合併成單一列索引；這個 einsum 就成了一般的 GEMM。

## 數學表述

$$
C_{bijk} = \sum_{l=0}^{L-1} A_{bijl}\,W_{lk}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入張量 $B\times I\times J\times L$（列優先，$l$ 連續） |
| $W$ | 矩陣運算元，$L\times K$（在簽章中稱為 `B`） |
| $C$ | 輸出張量 $B\times I\times J\times K$ |
| $b, i, j$ | 保留不變的自由索引 |
| $l$ | 縮併索引；$k$ 為輸出欄索引 |

$A$ 的所有自由索引都位於縮併索引之前，因此可攤平成單一列索引：

$$
\rho = (bI + i)J + j, \qquad C_{\rho k} = \sum_l A_{\rho l}W_{lk}, \qquad 0 \le \rho < BIJ
$$

| 符號 | 意義 |
|---|---|
| $\rho$ | 攤平後的列索引 |
| $BIJ$ | 攤平後 GEMM 的列數（最大案例中為 2 M） |

## 解題思路

只需啟動共用核心一次，設定 `rows = b*i*j`、`inner = l`、`cols = k`。
這些形狀高而窄（$L = 32 \dots 256$），因此每個區塊只執行 2–16 個 K 切片；
每個切片的載入與同步開銷比方形 GEMM 更為重要。

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
W = 2BIJLK, \qquad Q_{\min} = 4\,(BIJL + LK + BIJK)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算數（一次 FMA = 2 次浮點運算） |
| $Q_{\min}$ | 必要的 DRAM 位元組數（每個運算元讀取一次，輸出寫入一次） |
| $F$ | FP32 峰值（目前 GPU 為數十 TFLOP/s） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | Roofline 模型的時間下限 |

最大案例：$W = 2\cdot 2^{21}\cdot256\cdot768 = 0.82$ TFLOP，
而 $Q_{\min} \approx 8.6$ GB。運算強度（約 95 flop/byte）高於 FP32 的
效能轉折點，但高出不多，因此運算與頻寬都很重要。

## 常見陷阱

- **名稱衝突**：簽章中的 `B` 是矩陣；批次大小是 `b`。
- **引數順序** `(A, B, C, b, i, j, l, k)`：$l$ 位於 $k$ 之前。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [3D 矩陣乘法](../matmul-3d/)、[矩陣乘法](../matrix-multiplication/)。
