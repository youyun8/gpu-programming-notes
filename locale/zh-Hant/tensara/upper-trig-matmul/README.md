---
title: 上三角矩陣乘法
platform: Tensara
upstream: upper-trig-matmul
url: https://tensara.org/problems/upper-trig-matmul
difficulty: medium
tags: [matmul, sgemm, triangular, work-skipping]
status: solved
---

# 上三角矩陣乘法

**平台：** Tensara · **難度：** 中等 · [題目敘述](https://tensara.org/problems/upper-trig-matmul)

## 題意

將兩個上三角 $N\times N$ FP32 矩陣相乘（$N$ = 2048 … 8192）。
參考實作會對兩者套用 `torch.triu`，再執行稠密矩陣乘法。
檢查條件為 `rtol = 1e-4`、`atol = 1e-3`。

## 圖解

![上三角矩陣乘法：每個分塊只需 r₀ ≤ k < c₀ + 64](figure.svg)

灰色格是為零的下三角。位於對角線下方的分塊直接寫入 0，其餘分塊只在兩個因子都可能非零的 k 範圍內迴圈。

## 數學表述

$$
U_{ij} = 0 \ \text{for } i > j, \qquad
C_{ij} = \begin{cases} \displaystyle\sum_{k=i}^{j} A_{ik}B_{kj}, & i \le j \\ 0, & i > j \end{cases}
$$

| 符號 | 意義 |
|---|---|
| $U$ | 上三角矩陣 |
| $A, B$ | 套用 `triu` 後的輸入 |
| $C$ | 上三角乘積 |
| $k \in [i, j]$ | $A_{ik} \ne 0$ 要求 $k \ge i$，$B_{kj} \ne 0$ 要求 $k \le j$ |

對於列範圍為 $[r_0, r_0+64)$、欄範圍為 $[c_0, c_0+64)$ 的分塊：

$$
k \in \bigl[\,r_0,\ \min(N,\ c_0 + 64)\,\bigr)
$$

| 符號 | 意義 |
|---|---|
| $r_0, c_0$ | 分塊的第一列與第一欄 |

## 解題思路

這是[下三角矩陣乘法](../lower-trig-matmul/)的鏡像版本：嚴格位於對角線
下方的分塊填入 0，其餘分塊在上述範圍內迭代 $k$；載入時遮蔽
$A_{ik}$（$k < i$）與 $B_{kj}$（$k > j$）。

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
W_{\text{dense}} = 2N^3, \qquad
W_{\text{tri}} = 2\sum_{d=0}^{N-1} (N - d)(d + 1) \approx \frac{N^3}{3}, \qquad
\frac{W_{\text{tri}}}{W_{\text{dense}}} \approx \frac{1}{6}
$$

| 符號 | 意義 |
|---|---|
| $W_{\text{dense}}$ | 稠密 $N\times N$ 乘積的 Flops |
| $W_{\text{tri}}$ | 實際需要的 Flops：三角形中的輸出 $(i, j)$ 與對角線距離為 $d = \lvert i - j\rvert$，需要 $d + 1$ 次 FMA，而這類輸出有 $N - d$ 個 |
| $d$ | 與對角線的距離 |

使用寬度為 64 的分塊時，核心會多做少量工作（完整分塊與 16 對齊的
$k$ 範圍）；仍接近 $1/6$ 的下限（當 $N \ge 2048$）。在 $N = 8192$ 時，
工作量約為 0.18 TFLOP，而非 1.1。

## 常見陷阱

- 容許誤差比下三角問題**更嚴格**（`atol = 1e-3`）；因為略過的項目
  都是精確的 0，所以仍可通過。
- 與下三角案例一樣，**在載入時套用遮罩**。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [下三角矩陣乘法](../lower-trig-matmul/)、[方形矩陣乘法](../square-matmul/)。
