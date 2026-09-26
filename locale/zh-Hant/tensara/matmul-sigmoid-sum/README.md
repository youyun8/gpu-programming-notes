---
title: 矩陣乘法搭配 Sigmoid 與加總
platform: Tensara
upstream: matmul-sigmoid-sum
url: https://tensara.org/problems/matmul-sigmoid-sum
difficulty: medium
tags: [matmul, sgemm, fusion, reduction, atomics]
status: solved
---

# 矩陣乘法搭配 Sigmoid 與加總

**平台：** Tensara · **難度：** 中等 · [題目說明](https://tensara.org/problems/matmul-sigmoid-sum)

## 問題

對大小為 $M\times K$ 的 $A$ 與 $K\times N$ 的 $B$（尺寸為 512 … 1024），
傳回單一純量 $\sum_{i,j}\sigma\bigl((AB)_{ij}\bigr)$。
檢查條件較寬鬆：`rtol = 5e-2`、`atol = 1e-2`。

## 公式

$$
G_{ij} = \sum_{k=0}^{K-1} A_{ik}B_{kj}, \qquad
\text{result} = \sum_{i=0}^{M-1}\sum_{j=0}^{N-1} \sigma(G_{ij})
$$

| 符號 | 意義 |
|---|---|
| $A, B$ | GEMM 運算元 |
| $G$ | 乘積（不會儲存） |
| $\sigma$ | 邏輯 Sigmoid |
| Result | 純量輸出 |

總和依輸出圖塊拆分：

$$
\text{result} = \sum_{t} S_t, \qquad S_t = \sum_{(i,j)\in t} \sigma(G_{ij})
$$

| 符號 | 意義 |
|---|---|
| $t$ | 一個 $64\times64$ 輸出圖塊（一個執行緒區塊） |
| $S_t$ | 圖塊的部分總和 |

## 方法

使用共用 SGEMM 的變形，其尾聲不儲存任何內容：

1. 每個執行緒對自己的 16 個累加值套用 $\sigma$ 並加總。
2. 區塊歸約 256 個逐執行緒值：先以 float warp-shuffle 蝶形運算歸約，
   再由執行緒 0 以 `double` 加總 8 個 warp 總和。
3. 每個區塊由一個執行緒對 `double` 執行 `atomicAdd(&g_sum, S_t)`
   （sm_60+ 原生支援）。
4. 單執行緒核心將 `g_sum` 轉成 float 並寫入 `output`（每次呼叫開始時，
   會先由重設核心將其歸零）。

$M\times N$ 的中間結果從不具體存在；全域寫入只有每個區塊一次的原子操作。

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
W = 2MNK, \qquad Q_{\min} = 4\,(MK + KN)\ \text{bytes}, \qquad T_{\min} = \max\left(\frac{W}{F},\ \frac{Q_{\min}}{\beta}\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算數（一次 FMA = 2 次浮點運算） |
| $Q_{\min}$ | 必要的 DRAM 位元組數（每個運算元讀取一次，輸出寫入一次） |
| $F$ | FP32 峰值（目前 GPU 為數十 TFLOP/s） |
| $\beta$ | DRAM 頻寬 |
| $T_{\min}$ | Roofline 模型的時間下限 |

輸出只有一個 float，因此 $Q_{\min}$ 不含 $MN$ 項：在 $1024^3$ 時，
未融合的管線會兩度寫入並重新讀取 4 MB，與此尺寸 GEMM 本身的輸入傳輸量相當。

## 注意事項

- **非確定性**：原子加法的順序每次執行都不同，因此 double 總和的最低幾位會改變；
  捨入成 float 後看不出差異。
- 每次呼叫都要**重設累加器**；`__device__` 全域變數會在多次啟動之間保留值。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [矩陣乘法 + Swish](../matmul-swish/)、[Sigmoid](../sigmoid/)、
  [Frobenius 範數](../frobenius-norm/)（使用相同的兩層歸約）。
