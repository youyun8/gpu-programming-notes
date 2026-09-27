---
title: 對稱矩陣乘法
platform: Tensara
upstream: symmetric-matmul
url: https://tensara.org/problems/symmetric-matmul
difficulty: medium
tags: [matmul, sgemm, register-blocking]
status: solved
---

# 對稱矩陣乘法

**平台：** Tensara · **難度：** 中等 · [題目敘述](https://tensara.org/problems/symmetric-matmul)

## 題意

將兩個對稱的 $N\times N$ FP32 矩陣相乘（$N$ = 4096 … 9216）。
檢查條件為 `rtol = 1e-6`、`atol = 5e-3`，因此實際上以絕對容許誤差
為主。值得探討的是對稱性是否能提供幫助。

## 圖解

![對稱的輸入、一般的輸出：只有 A 與 B 可交換時，AB 才會對稱](figure.svg)

即使 A 與 B 都是對稱矩陣，乘積通常並不對稱，因此仍以一般的分塊 SGEMM 計算全部 N² 個輸出。

## 數學表述

$$
A = A^{\mathsf T},\quad B = B^{\mathsf T}, \qquad C_{ij} = \sum_{k=0}^{N-1} A_{ik}B_{kj}
$$

| 符號 | 意義 |
|---|---|
| $A, B$ | 對稱輸入，$N\times N$ |
| $C$ | 乘積，$N\times N$（通常**不**對稱） |
| $^{\mathsf T}$ | 轉置 |

只有兩個對稱矩陣可交換時，其乘積才會對稱：

$$
C^{\mathsf T} = (AB)^{\mathsf T} = B^{\mathsf T}A^{\mathsf T} = BA \ne AB \ \text{in general}
$$

| 符號 | 意義 |
|---|---|
| $BA$ | 顛倒順序的乘積 |

因此必須計算全部 $N^2$ 個輸出，對稱性不能把工作量減半
（BLAS `SSYMM` 節省的是儲存空間，而非 flops）。對稱性只允許將 $B$
當作 $B^{\mathsf T}$ 讀取，也就是可選用「NN」或「NT」暫存路徑。

## 解題思路

不加修改地使用共用核心（`NoEpi`、NN 配置）。

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

與[方形矩陣乘法](../square-matmul/)相同。

## 常見陷阱

- **不要只計算一半的 $C$ 再鏡射**：$C$ 並不對稱。
- **`rtol = 1e-6`** 看似嚴格，但對大小為 $O(\sqrt{N})$ 的元素而言，
  `atol = 5e-3` 才是主導條件。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [方形矩陣乘法](../square-matmul/)、[下三角矩陣乘法](../lower-trig-matmul/)、
  [上三角矩陣乘法](../upper-trig-matmul/)。
