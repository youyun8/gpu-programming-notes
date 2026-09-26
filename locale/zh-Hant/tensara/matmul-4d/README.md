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

## 問題

對形狀為 $B\times I\times J\times L$ 的 4D 張量 $A$ 與形狀為
$L\times K$ 的矩陣計算 `einsum("bijl,lk->bijk", A, B)`（最大案例：
$16\times256\times512\times256$ 乘以 $256\times768$）。
檢查條件為 `rtol = 2e-4`、`atol = 6e-4`。

## 公式

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

## 方法

只需啟動共用核心一次，設定 `rows = b*i*j`、`inner = l`、`cols = k`。
這些形狀高而窄（$L = 32 \dots 256$），因此每個區塊只執行 2–16 個 K 切片；
每個切片的載入與同步開銷比方形 GEMM 更為重要。

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

## 注意事項

- **名稱衝突**：簽章中的 `B` 是矩陣；批次大小是 `b`。
- **引數順序** `(A, B, C, b, i, j, l, k)`：$l$ 位於 $k$ 之前。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [3D 矩陣乘法](../matmul-3d/)、[矩陣乘法](../matrix-multiplication/)。
