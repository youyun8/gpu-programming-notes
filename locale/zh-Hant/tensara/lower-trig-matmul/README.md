---
title: 下三角矩陣乘法
platform: Tensara
upstream: lower-trig-matmul
url: https://tensara.org/problems/lower-trig-matmul
difficulty: medium
tags: [matmul, sgemm, triangular, work-skipping]
status: solved
---

# 下三角矩陣乘法

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/lower-trig-matmul)

## 問題

將兩個 $N\times N$ 的 FP32 下三角矩陣相乘（$N$ = 2048 … 8192）。
參考實作會先對兩個輸入套用 `torch.tril`，再執行稠密矩陣乘法。
檢查條件為 `rtol = 8e-4`、`atol = 2e-2`。

## 公式

$$
L_{ij} = 0 \ \text{for } i < j, \qquad
C_{ij} = \sum_{k=0}^{N-1} L^{A}_{ik} L^{B}_{kj} = \begin{cases} \displaystyle\sum_{k=j}^{i} A_{ik}B_{kj}, & i \ge j \\ 0, & i < j \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $L^{A} = \operatorname{tril}(A)$, $L^{B} = \operatorname{tril}(B)$ | 將嚴格上三角部分歸零後的輸入 |
| $C$ | 乘積，同樣是下三角矩陣 |
| $k \in [j, i]$ | 兩個因數都可能非零的唯一歸約索引：$A_{ik} \ne 0$ 需要 $k \le i$，$B_{kj} \ne 0$ 需要 $k \ge j$ |

對列範圍為 $[r_0, r_0 + 64)$、欄範圍為 $[c_0, c_0 + 64)$ 的完整輸出圖塊，
這些範圍的聯集為

$$
k \in \bigl[\,c_0,\ \min(N,\ r_0 + 64)\,\bigr)
$$

| 符號 | 意義 |
|---|---|
| $r_0, c_0$ | 圖塊的第一列與第一欄 |

## 方法

使用共用 SGEMM 的特製版本（`triMatmul`）：

1. **完全位於對角線上方的圖塊**（$c_0 \ge r_0 + 64$）直接寫入零，
   不存取 $A$ 或 $B$。
2. **其他圖塊**只在 $[\lfloor c_0/16\rfloor\cdot16,\ \min(N, r_0 + 64))$ 範圍內迭代 $k$。
3. **載入時遮罩另一側三角形**（$k > i$ 的 $A_{ik}$ 與 $k < j$ 的 $B_{kj}$
   載入為 0），與參考實作中的 `tril` 完全一致，因此輸入上三角部分的垃圾值不會滲入。

其餘部分（64 × 64 圖塊、寬度 16 的 K 切片、每個執行緒 4 × 4）皆採用下述共用核心。

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
W_{\text{dense}} = 2N^3, \qquad
W_{\text{tri}} = 2\sum_{d=0}^{N-1} (N - d)(d + 1) \approx \frac{N^3}{3}, \qquad
\frac{W_{\text{tri}}}{W_{\text{dense}}} \approx \frac{1}{6}
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{dense}}$ | 稠密 $N\times N$ 乘積的浮點運算數 |
| $W_{\text{tri}}$ | 實際所需的浮點運算數：三角形中的輸出 $(i, j)$ 與對角線相距 $d = \lvert i - j\rvert$，需要 $d + 1$ 次 FMA，而這類輸出共有 $N - d$ 個 |
| $d$ | 與對角線的距離 |

使用寬度 64 的圖塊時，核心會多做一些運算（完整圖塊與對齊 16 的 $k$ 範圍），
但在 $N \ge 2048$ 時很接近 $1/6$ 的下限。$N = 8192$ 時約為 0.18 TFLOP，
而非 1.1 TFLOP。

## 注意事項

- **遮罩輸入**：只略過 $k$ 範圍仍不足夠；圖塊內的列 $i$ 與欄 $j$ 仍不相同，
  因此載入時必須將各三角形外的元素歸零。
- **較寬鬆的容許誤差**（`atol = 2e-2`）：參考實作會以不同順序加總 $N$ 項，
  其中多數為零。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [上三角矩陣乘法](../upper-trig-matmul/)、[方陣乘法](../square-matmul/)、
  [對角矩陣乘法](../diagonal-matmul/)。
