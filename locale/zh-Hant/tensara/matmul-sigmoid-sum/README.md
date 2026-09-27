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

## 題意

對大小為 $M\times K$ 的 $A$ 與 $K\times N$ 的 $B$（尺寸為 512 … 1024），
傳回單一純量 $\sum_{i,j}\sigma\bigl((AB)_{ij}\bigr)$。
檢查條件較寬鬆：`rtol = 5e-2`、`atol = 1e-2`。

## 圖解

![Σ σ(AB)：每個區塊對自己的輸出分塊套用 σ 並歸約，再以一次原子操作累加](figure.svg)

乘積從不寫回記憶體。每個區塊在暫存器中對輸出分塊套用 σ，歸約成一個數，再以一次 atomicAdd 加到結果上。

## 數學表述

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

## 解題思路

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

## 常見陷阱

- **非確定性**：原子加法的順序每次執行都不同，因此 double 總和的最低幾位會改變；
  捨入成 float 後看不出差異。
- 每次呼叫都要**重設累加器**；`__device__` 全域變數會在多次啟動之間保留值。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [矩陣乘法 + Swish](../matmul-swish/)、[Sigmoid](../sigmoid/)、
  [Frobenius 範數](../frobenius-norm/)（使用相同的兩層歸約）。
