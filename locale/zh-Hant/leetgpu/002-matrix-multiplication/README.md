---
title: 矩陣乘法
platform: LeetGPU
upstream: easy/2_matrix_multiplication
url: https://leetgpu.com/challenges/matrix-multiplication
difficulty: easy
tags: [gemm, shared-memory, tiling, register-blocking]
status: solved
---

# 矩陣乘法

**平台：** LeetGPU · **難度：** easy · [題目說明](https://leetgpu.com/challenges/matrix-multiplication)

## 問題

將兩個以列優先儲存的 float32 矩陣相乘：$A$ 是 $M \times N$，
$B$ 是 $N \times K$，而 $M \times K$ 的乘積寫入 `C`
（$1 \le M, N, K \le 8192$；基準為 $M = 8192$、$N = 6144$、$K = 4096$）。
請注意命名：本題的 **$N$ 是共用（內部）維度**，$K$ 是輸出欄數，
恰好與一般 BLAS 慣例相反。參考實作是 `torch.matmul`，
容許誤差為 `atol = rtol = 1e-4`。

GEMM 是深度學習中最重要的核心，而本題是學習**資料重用**的經典範例：
在相同硬體上，樸素核心比平鋪版本慢約 30 倍。

## 公式

$$
C_{rc} = \sum_{k=0}^{N-1} A_{rk}\, B_{kc}, \qquad 0 \le r < M,\ \ 0 \le c < K
$$

| 符號 | 意義 |
|---|---|
| $M$ | $A$ 與 $C$ 的列數 |
| $N$ | $A$ 的欄數 = $B$ 的列數（歸約維度） |
| $K$ | $B$ 與 $C$ 的欄數 |
| $r,\ c$ | 輸出元素的列與欄 |
| $k$ | 沿內部維度的加總索引 |
| $A_{rk}$ | $A$ 的元素，以列優先方式儲存在偏移量 $rN + k$ |
| $B_{kc}$ | $B$ 的元素，儲存在偏移量 $kK + c$ |
| $C_{rc}$ | $C$ 的元素，儲存在偏移量 $rK + c$ |

### 平鋪

將三層迴圈切成區塊。一個執行緒區塊負責一個
$T_M \times T_K$ 輸出分塊，並以寬度 $T_N$ 的切片走過內部維度：

$$
C_{\text{tile}} = \sum_{s=0}^{\lceil N/T_N\rceil - 1} A[\,r_0 : r_0{+}T_M,\ sT_N : (s{+}1)T_N\,]\ \cdot\ B[\,sT_N : (s{+}1)T_N,\ c_0 : c_0{+}T_K\,]
$$

| 符號 | 意義 |
|---|---|
| $T_M,\ T_K$ | 每個區塊的輸出分塊大小：此處為 64 × 64 |
| $T_N$ | 每一步暫存到共享記憶體的內部維度切片：此處為 16 |
| $s$ | 切片索引 |
| $r_0,\ c_0$ | 區塊分塊的左上角：$r_0 = 64\,\texttt{blockIdx.y}$，$c_0 = 64\,\texttt{blockIdx.x}$ |
| $X[a{:}b,\ c{:}d]$ | 列為 $a \dots b-1$、欄為 $c \dots d-1$ 的子矩陣 |

## 方法

### 平行分解

| 層級 | 負責內容 | 大小 |
|---|---|---|
| 網格 | 整個 $C$ | $\lceil K/64\rceil \times \lceil M/64\rceil$ 個區塊 |
| 區塊（256 個執行緒） | $C$ 的一個 64 × 64 分塊 | 4096 個輸出 |
| 執行緒 | 間隔為 16 的 4 × 4 輸出 | 暫存器中的 16 個累加器 |

執行緒 $(t_x, t_y) = (\texttt{tid} \bmod 16,\ \lfloor\texttt{tid}/16\rfloor)$
負責列 $t_y + 16i$ 與欄 $t_x + 16j$，其中 $i, j \in \{0,1,2,3\}$。
使用 16 的間隔（而不是連續的 4 × 4 區塊）可讓連續執行緒對應連續欄，
使最後寫入 `C` 時能合併存取。

### 每個內部切片的主迴圈

1. **暫存。** 全部 256 個執行緒合作，將 $A$ 的 64 × 16 切片與
   $B$ 的 16 × 64 切片複製到共享記憶體。超出範圍的元素存為 0，
   因此計算迴圈不需特別處理邊緣分塊。
   - $A$ 的切片以**轉置**方式儲存（`a_tile[k][r]`）。如此一來，
     兩個運算元在內部迴圈中都沿共享記憶體的連續列讀取。
   - 每列補齊至 64 + 4 個浮點數，避免轉置儲存時的 bank 衝突。
2. `__syncthreads()`：確認切片完整。
3. **計算。** 對 16 個 $k$ 值逐一處理：將 4 個 $A$ 值與 4 個
   $B$ 值載入暫存器，執行外積的 $4 \times 4 = 16$ 次 FMA。
4. `__syncthreads()`：避免有人在其他執行緒尚未讀完時覆寫切片。

### 為何使用暫存器分塊

每個內部步驟中，一個執行緒用 8 次共享記憶體載入完成 16 次 FMA，
即每次載入可做 2 次 FMA。經典的「每執行緒一個輸出」平鋪核心，
每次 FMA 需要 2 次載入，共享記憶體流量高出 4 倍。
共享記憶體頻寬正是較簡單核心的限制。

## 成本分析

$$
W = 2MNK, \qquad
Q_{\text{naive}} \approx 2MNK \cdot 4, \qquad
Q_{\text{tiled}} \approx 4\left(MN\,\frac{K}{T_K} + NK\,\frac{M}{T_M} + MK\right), \qquad
I_{\text{tiled}} \approx \frac{W}{Q_{\text{tiled}}} \approx \frac{1}{4}\cdot\frac{2}{1/T_K + 1/T_M}
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP：每個 $(r, c, k)$ 三元組包含一次乘法與一次加法 |
| $Q_{\text{naive}}$ | 若每次 FMA 都從 DRAM 取得兩個運算元時的全域記憶體位元組數 |
| $Q_{\text{tiled}}$ | 平鋪後的位元組數：每一欄分塊會重讀一次 $A$（共 $K/T_K$ 次），每一列分塊會重讀一次 $B$（共 $M/T_M$ 次），$C$ 則寫入一次 |
| $I_{\text{tiled}}$ | 算術強度（FLOP/byte），忽略 $C$ 項 |
| 4 | 每個 float32 的位元組數 |

當 $T_M = T_K = 64$ 時，$I_{\text{tiled}} \approx 16$ FLOP/byte，
樸素版本則為 $0.25$，DRAM 流量減少 64 倍。（L2 快取讓實際樸素核心
優於 0.25，但仍遠低於 16。）在基準大小下，
$W = 2 \cdot 8192 \cdot 6144 \cdot 4096 \approx 4.1 \times 10^{11}$ FLOP。
若 GPU 的 fp32 FMA 吞吐量約為 20 TFLOP/s，計算下限約為 20 ms。
此核心只能達到其中一部分；要達到 cuBLAS 等級的效能，還需採用
[教學 04](../../tutorials/04-tiled-matmul.md) 中的進一步技巧
（向量化載入、雙緩衝、warp 平鋪）。

## 常見問題

- **維度命名。** 依照 BLAS 習慣把 $K$ 當成內部維度，會在
  $N \ne K$ 時產生錯誤結果。
- **邊緣分塊。** 載入 0 可讓每個執行緒執行相同的屏障。
  若超出範圍的執行緒跳過載入，共享記憶體中會殘留舊資料。
- **每個切片需要兩次屏障。** 第一次讓資料可見；第二次避免較快的
  執行緒覆寫較慢執行緒仍在讀取的切片。
- **64 位元偏移量。** $rN + k$ 最多可達
  $8192 \cdot 8192 = 2^{26}$，本來就安全，但程式仍使用 `size_t`，
  讓核心可安全處理更大的形狀。

## 驗證

所有 LeetGPU 測試案例（包括 $M = 1$ 或 $N = 3$ 等非 64 倍數）
都在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-4` 通過。
核心也以 `nvcc -arch=sm_80` 檢查編譯。

## 相關內容

- [GEMM（fp16、張量核心）](../022-gemm/)、[批次矩陣乘法](../030-batched-matrix-multiplication/)、
  [INT8 矩陣乘法](../032-int8-quantized-matmul/)。
- Tensara [方形矩陣乘法](../../tensara/square-matmul/)、[GEMM + ReLU](../../tensara/gemm-relu/)。
- [教學 04－平鋪矩陣乘法](../../tutorials/04-tiled-matmul.md)及 AMD
  系列[05](../../tutorials/05-amd-cdna3-mfma.md)–[07](../../tutorials/07-hipblaslt-tensilelite.md)。
