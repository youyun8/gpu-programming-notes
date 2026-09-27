# 矩陣乘法 2 – 向量化載入

> **第三部分 · 矩陣乘法** ·
> 程式：[`01-vectorized.cu`](01-vectorized.cu) · 先備知識：[矩陣乘法 1 – 基礎](../04-tiled-matmul.md) ·
> 下一篇：[矩陣乘法 3 – 雙緩衝](02-double-buffering.md)

[矩陣乘法 1 – 基礎](../04-tiled-matmul.md#4-register-tiling)使用
$4\times4$ 暫存器分塊的 kernel，大部分發射槽都花在記憶體
指令，而不是 FMA。本頁把分塊擴大到每個 block $128\times128$、每個 thread
$8\times8$，並讓每次記憶體存取的寬度都成為 16 位元組：

- $A$ 與 $B$ 的全域載入：`LDG.E.128`；
- 共享記憶體 fragment 載入：`LDS.128`，搭配不會發生 bank 衝突的配置；
- 儲存 $C$：`STG.E.128`。

**你將學到**

- 為何限制使用暫存器分塊 GEMM 的不只是位元組數，還有指令數；
- 如何分配 thread 的輸出，使 128 位元共享記憶體載入不會衝突；
- 32 個 bank 如何以每次 8 個 lane 的方式處理 `LDS.128`；
- 為何 $A$ 在存入共享記憶體時要轉置，以及 padding 如何避免衝突；
- 如何讓任何矩陣形狀都能合法使用向量存取（執行時對齊檢查及純量備援路徑）。

## 1. 為何指令數很重要

SM 的每個子分割區每個週期會發射一條 warp 指令。在使用暫存器分塊 GEMM
的內部迴圈中，有效指令是 FMA；每次載入都會占用原本可用來發射 FMA 的槽。
每個 $k$ 步驟中，擁有 $t_M\times t_N$ 分塊的 thread 會執行

$$
n_{\text{FMA}} = t_M t_N, \qquad
n_{\text{LDS}} = \frac{t_M + t_N}{v}, \qquad
\rho = \frac{n_{\text{FMA}}}{n_{\text{FMA}} + n_{\text{LDS}}}
$$

| 符號 | 意義 |
|---|---|
| $t_M, t_N$ | 每個 thread 沿 $M$、$N$ 方向負責的輸出數 |
| $v$ | 每條共享記憶體載入指令讀取的 float 數（`LDS.32` 為 1，`LDS.128` 為 4） |
| $n_{\text{FMA}}, n_{\text{LDS}}$ | 每個 thread 在每個 $k$ 步驟中的 FMA 與共享載入指令數 |
| $\rho$ | 迴圈指令中 FMA 所占比例（不計位址運算） |

| Thread 分塊 | $v$ | $n_{\text{FMA}}$ | $n_{\text{LDS}}$ | $\rho$ |
|---|---|---|---|---|
| $4\times4$（基礎） | 1 | 16 | 8 | 67 % |
| $8\times8$ | 1 | 64 | 16 | 80 % |
| $8\times8$ | 4 | 64 | 4 | 94 % |

加大分塊可提高重用率；加寬載入則能移除大部分剩餘的載入指令。代價是暫存器：
64 個累加器、16 個 fragment 值及暫存用途，每個 thread 約需 120 個暫存器
（在 sm_80 上，ptxas 對此 kernel 回報 117 個），因此每個 SM 最多只能容納
兩個 256-thread block。這沒有問題：每個 warp 在每個 $k$ 步驟有 64 個
互相獨立的 FMA，可自行隱藏延遲（第 01 章第 5 節）。

## 2. 哪個 Thread 負責哪些輸出

Block 有 256 個 thread，排列成 $16\times16$ 網格 $(t_x, t_y)$。直覺的映射會
讓每個 thread 負責一個 $8\times8$ 方塊，但 lane $t_x$ 隨後會讀取
`b_s[kk][8 tx .. 8 tx + 7]`：相鄰 lane 的位址相隔 32 位元組。程式改為把每個
thread 的列與欄分成兩組，每組 4 個，兩組相隔 64：

$$
\text{rows}(t_y) = \{4t_y, \dots, 4t_y + 3\} \cup \{64 + 4t_y, \dots, 64 + 4t_y + 3\}, \qquad
\text{cols}(t_x) = \{4t_x, \dots, 4t_x + 3\} \cup \{64 + 4t_x, \dots, 64 + 4t_x + 3\}
$$

| 符號 | 意義 |
|---|---|
| $t_x, t_y$ | `threadIdx.x % 16` 與 `threadIdx.x / 16` |
| rows, cols | 該 thread 累加乘積的 block 分塊列與欄 |

![輸出歸屬：每個 thread 負責四個相隔 64 的 4 × 4 區塊](../figures/gemm-thread-map.svg)

現在 lane $t_x$ 執行 `float4` fragment 載入時，位元組位址是 $16t_x$
（加上一個常數）；相鄰 lane 會讀取連續的 16 位元組區塊。

## 3. `LDS.128` 如何配合 32 個 Bank

共享記憶體請求會以最多 128 位元組的 *wavefront* 處理，每個 bank 一個
4 位元組 word。對 128 位元載入，硬體每次處理 warp 中的 8 個 lane
（$8\times16$ B = 128 B），因此當且僅當每組 8 個 lane 涵蓋 32 個不同的
bank 時，128 位元載入才不會衝突：

![Lane stride 為 16 位元組與 32 位元組時的 128 位元載入 bank 映射](../figures/gemm-lds128.svg)

對 $A$ fragment 而言，半個 warp 的所有 lane 具有相同 $t_y$，所以會讀取
同一個位址：這是 broadcast，永遠不會衝突。

## 4. 載入、儲存與轉置後的 $A$ 分塊

### 4.1 載入一個切片

每個 block 的每個 $k$ 切片包含一個 $128\times8$ 的 $A$ 分塊，以及一個
$8\times128$ 的 $B$ 分塊：各有 256 個 `float4`，每個 thread 負責一個。

```cpp
const int a_row = tid / 2, a_col = (tid % 2) * 4;   // A: 2 float4 per row of the slice
const int b_row = tid / 32, b_col = (tid % 32) * 4; // B: 32 float4 per row of the slice
...
const float4 av = load4<kVec>(a, m, k, row0 + a_row, k0 + a_col);
const float4 bv = load4<kVec>(b, k, n, k0 + b_row, col0 + b_col);
a_s[a_col + 0][a_row] = av.x;   // A is stored transposed: a_s[k][m]
a_s[a_col + 1][a_row] = av.y;
a_s[a_col + 2][a_row] = av.z;
a_s[a_col + 3][a_row] = av.w;
*reinterpret_cast<float4*>(&b_s[b_row][b_col]) = bv;
```

### 4.2 儲存時的四個細節

- **$B$** 沿著一列以 `float4` 抵達，並維持 row-major，因此只需一個
  `STS.128` 就能儲存。
- **$A$** 的使用方式是沿著一欄（相同 $k$ 上連續 $t_M$ 列），所以會在
  **存入共享記憶體時轉置**。代價是四個純量儲存
  `a_s[a_col + i][a_row]`；交換條件是 $A$ 的每次 fragment 載入都能使用
  `float4`。
- **填補（padding）。** 對固定的 `i`，warp 的 32 個 lane 會在兩個 `a_col` 值上，
  儲存至 `tid / 2` 列（16 個不同值）。若沒有 padding，兩個 $k$ 列相隔
  $4\times128$ 個 word，是 32 的倍數，會產生 2-way 衝突。每列改成
  $128 + 4$ 個 word 後，間距為 528 個 word，也就是相隔 16 個 bank。
  填補也讓每列維持 16 位元組的倍數，使 `float4` 讀取保持對齊。
- **邊界。** `load4<kVec>` 對矩陣外的部分傳回零，因此運算迴圈不需要
  邊界檢查。當 $K$ 與 $N$ 都是 4 的倍數時，template 參數 `kVec` 為 true：
  此時每列都以 16 位元組對齊，且 `float4` 不是完全在矩陣內，就是完全在
  矩陣外。否則載入會退回使用 4 次純量載入。主機端會選擇適當的樣板具現化版本。

### 4.3 內層迴圈

內層迴圈因此只需四個 `LDS.128` 與 64 個 FMA：

```cpp
for (int kk = 0; kk < kBlockK; ++kk) {
    const float4 a_lo = *reinterpret_cast<const float4*>(&a_s[kk][4 * ty]);
    const float4 a_hi = *reinterpret_cast<const float4*>(&a_s[kk][64 + 4 * ty]);
    const float4 b_lo = *reinterpret_cast<const float4*>(&b_s[kk][4 * tx]);
    const float4 b_hi = *reinterpret_cast<const float4*>(&b_s[kk][64 + 4 * tx]);
    ...
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
}
```

## 5. 各層級的流量

$$
I_{\text{L2}} = \frac{2B_MB_NB_K}{4B_K(B_M + B_N)} = \frac{B_MB_N}{2(B_M + B_N)} = 32\ \frac{\text{flop}}{\text{byte}}, \qquad
I_{\text{smem}} = \frac{2t_Mt_N}{4(t_M + t_N)} = 2\ \frac{\text{flop}}{\text{byte}}
$$

| 符號 | 意義 |
|---|---|
| $B_M, B_N, B_K$ | Block 分塊：128、128、8 |
| $I_{\text{L2}}$ | 從 L2/DRAM 載入共享記憶體時，每位元組可執行的 flop 數 |
| $I_{\text{smem}}$ | 從共享記憶體讀入暫存器時，每位元組可執行的 flop 數（$t_M = t_N = 8$） |

$I_{\text{L2}} = 32$ flop/B 高於第 00 章表中每款 GPU 的 FP32 ridge point，
因此只要 L2 能吸收 block 之間的重複讀取，這個 kernel 就不再受頻寬限制；
剩下的問題是延遲（下一頁）與指令開銷。

## 6. 常見陷阱

- **對齊。** 對未以 16 位元組對齊的位址使用
  `reinterpret_cast<const float4*>` 是未定義行為（GPU 上會發生位址未對齊
  錯誤）。務必像 `launchVectorized` 一樣，根據執行時的 leading dimension
  檢查來選擇向量路徑。
- **動態索引 `acc`。** 對 `acc`、`a_frag` 與 `b_frag` 的每個迴圈都必須
  完全展開；只要出現一個非常數索引，陣列就會移到 local memory。
  `-Xptxas -v` 回報「bytes stack frame」就是症狀。
- **暫存器壓力。** 64 個累加器大約是單一 thread 使用 FP32 的上限。
  擴大到 $16\times8$ 會讓累加器數量加倍並發生 spill。

## 重點整理

1. 寬分塊可提高重用率；寬（128 位元）載入則把剩餘載入指令減少 4 倍。
2. 要讓 `LDS.128` 不發生衝突，每組 8 個 lane 必須涵蓋連續 128 位元組；分割歸屬（4tx 與 64 + 4tx）即可達成。
3. 將 $A$ 存入共享記憶體時就轉置，讓兩個 fragment 都能沿列讀取。
4. 向量存取需要 16 位元組對齊：在執行時檢查，並保留純量路徑。

## 練習

1. 將歸屬方式改成相鄰的 $8\times8$ 方塊（`8 * tx + j`），並用 Nsight
   Compute 計算每個 `LDS.128` 的 wavefront 數
   （`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`）。

    <details markdown="1"><summary>答案</summary>

    Lane 之間此時相隔 32 位元組，因此 8 個 lane 橫跨 256 位元組；B 的每個
    `LDS.128` 每個 warp 需要 8 個 wavefront，而不是 4 個，也就是 2-way 衝突。

    </details>
2. 移除 `a_s` 的 `+ 4` padding，並測量儲存衝突。
3. 將 `kBlockK = 8` 改為 16。每個 block 的共享記憶體用量，以及每個 FMA
   對應的 barrier 數會如何變化？

    <details markdown="1"><summary>答案</summary>

    共享記憶體加倍為 $16\times132\times4 + 16\times128\times4 = 16\,640$
    位元組；每個 FMA 對應的 barrier 數減半（每 16 個 $k$ 步驟兩次，而非
    每 8 個步驟兩次）。Loader 也必須讓每個 thread 各搬移兩個 operand 的
    兩個 `float4`。

    </details>
