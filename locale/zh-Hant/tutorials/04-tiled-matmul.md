# 04 – 分塊矩陣乘法

> **第三部分 · 矩陣乘法** · 先備知識：[01](01-execution-model.md)、[02](02-memory-hierarchy.md) ·
> 下一篇：[04.x – GEMM 深入解析](gemm/README.md)

矩陣乘法與第 01–03 章的 kernel 相反：每個位元組有大量運算，因此*可能*
受運算能力限制，但前提是資料能從快速記憶體重複使用。本章會建立標準的
最佳化階梯：

1. 簡單版本；
2. 共享記憶體分塊；
3. 暫存器分塊；
4. 達到 cuBLAS 80–90 % 效能的技巧（每項技巧都在
   [04.x 深入解析](gemm/README.md)中有獨立頁面）。

以下皆假設 $C = AB$，$A$ 大小為 $M\times K$、$B$ 為 $K\times N$、
$C$ 為 $M\times N$，且全部是 row-major FP32。

**你將學到**

- 為何 GEMM 可能受運算能力限制，以及需要多少資料重用；
- 如何用一個公式分析 kernel 在每一層（DRAM/L2、共享記憶體、暫存器）的流量；
- Block（共享記憶體）分塊，包括其 barrier 與 bank 行為；
- 暫存器分塊：outer-product 形式，以及它為何能消除共享記憶體瓶頸；
- 如何把 epilogue（縮放、bias、activation）融合進 kernel；
- 其餘最佳化技巧的全貌。

## 1. 重要數字

### 1.1 工作量與必要流量

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}, \qquad
W = 2MNK, \qquad Q_{\min} = 4\,(MK + KN + MN), \qquad
I_{\max} = \frac{W}{Q_{\min}} \xrightarrow{M = N = K} \frac{n}{6}
$$

| 符號 | 意義 |
|---|---|
| $A, B, C$ | Operand 與結果 |
| $M, N, K$ | $C$ 的列數、$C$ 的欄數、歸約長度 |
| $W$ | Flop 數（每次乘加計為 2） |
| $Q_{\min}$ | 若每個元素都正好讀或寫一次，必要的 DRAM 位元組數 |
| $I_{\max}$ | 理論最佳算術強度；對 $n\times n$ 方陣會以 $n/6$ 成長 |

當 $n = 4096$，$I_{\max} \approx 680$ flop/byte，遠高於任何 GPU 的
ridge point（第 00 章）。關鍵是讓 DRAM 及每層晶片內記憶體所見的*實際*
強度都足夠高。

### 1.2 重用從何而來

每個元素 $A_{ik}$ 會使用 $N$ 次（$C$ 的每欄一次），每個 $B_{kj}$ 則使用
$M$ 次。若 kernel 每次使用都從 DRAM 擷取元素，強度只有 1/4 flop/byte；
若只擷取一次並從晶片內記憶體重複使用，便會接近 $I_{\max}$。本章所有技巧
都是在安排運算順序，讓元素載入快速記憶體後，在被逐出前盡量多用幾次。

若一個工作單位（block、warp 或 thread）負責 $C$ 的 $T_M\times T_N$
分塊並走過 $K$，每個 $k$ 需要 $T_M$ 個 $A$ 值與 $T_N$ 個 $B$ 值，並以
它們執行 $T_MT_N$ 個 FMA：

$$
\frac{\text{FMAs}}{\text{values loaded}} = \frac{T_M T_N}{T_M + T_N}
$$

| 符號 | 意義 |
|---|---|
| $T_M, T_N$ | 該工作單位負責的 $C$ 列數與欄數 |

把這個比例套用到每個層級，就能解釋以下所有結果。

## 2. 簡單 Kernel

### 2.1 每個輸出由一個 Thread 負責

```cpp
__global__ void matmulNaive(const float* a, const float* b, float* c, int m, int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) acc = fmaf(a[row * k + kk], b[kk * n + col], acc);
    c[row * n + col] = acc;
}
```

### 2.2 算術強度

每個 FMA 載入兩個 float（8 位元組），執行 2 個 flop
（$T_M = T_N = 1$）：

$$
I_{\text{naive}} = \frac{2}{8} = 0.25\ \frac{\text{flop}}{\text{byte}}
$$

| 符號 | 意義 |
|---|---|
| $I_{\text{naive}}$ | Load/store 單元所見的強度 |

### 2.3 Cache 的作用

在一個 warp 中（32 個連續 `col`、相同 `row`），
`a[row * k + kk]` 對所有 lane 都是同一位址（一次 broadcast transaction），
`b[kk * n + col]` 則是 32 個連續 float（coalesced）。相鄰 warp 與 block
會重讀相同的 $A$ 列與 $B$ 欄，L1/L2 會處理許多重讀。因此此 kernel
不會像 DRAM 強度 0.25 flop/byte 所暗示的那麼慢，但仍受載入指令速率與
L1/L2 頻寬限制，通常只能達到峰值的 1–5 %。

## 3. 共享記憶體分塊

### 3.1 概念

把 $C$ 拆成 $T\times T$ 分塊，每個分塊交給一個由 $T\times T$ 個 thread
組成的 block，並把 $k$ 迴圈拆成長度為 $T$ 的階段：

$$
C_{\mathcal{I}\mathcal{J}} = \sum_{s=0}^{\lceil K/T \rceil - 1} A_{\mathcal{I},\,\mathcal{K}_s}\; B_{\mathcal{K}_s,\,\mathcal{J}}
$$

| 符號 | 意義 |
|---|---|
| $\mathcal{I}, \mathcal{J}$ | 一個輸出分塊的 $T$ 列與 $T$ 欄 |
| $\mathcal{K}_s$ | 第 $s$ 個由 $T$ 個歸約索引組成的切片 |
| $A_{\mathcal{I},\mathcal{K}_s}$、$B_{\mathcal{K}_s,\mathcal{J}}$ | 暫存在共享記憶體的 $T\times T$ 子矩陣 |

![Block 分塊：一個 block 負責 C 的一個分塊，並逐切片走過 A 對應的橫向 panel 與 B 的縱向 panel](figures/ch04-block-tiling.svg)

### 3.2 Kernel

```cpp
constexpr int kTile = 32;

// launch: block(kTile, kTile), grid(ceil(n / kTile), ceil(m / kTile))
__global__ void matmulTiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < k; k0 += kTile) {
        // Each thread loads one element of each tile; out-of-range -> 0.
        const int a_col = k0 + threadIdx.x, b_row = k0 + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();
        for (int kk = 0; kk < kTile; ++kk) acc = fmaf(a_tile[threadIdx.y][kk], b_tile[kk][threadIdx.x], acc);
        __syncthreads();   // before the next phase overwrites the tiles
    }
    if (row < m && col < n) c[row * n + col] = acc;
}
```

### 3.3 兩個 Barrier

每個階段需要兩個 barrier，分別防止不同 hazard：

1. **載入之後**（read-after-write）：thread 的 FMA 迴圈會讀取其他 thread
   載入的分塊元素。
2. **FMA 之後**（write-after-read）：下一階段會覆寫較慢 thread 可能仍在
   讀取的分塊。

移除第二個 barrier 後，結果大多數時候仍然正確，這是最糟的 bug 類型。
[04.2](gemm/02-double-buffering.md) 說明如何用兩個緩衝區，將每個階段
減少為一個 barrier。

### 3.4 存取模式

- 兩個分塊的載入都是 coalesced：`threadIdx.x` 走過 $A$ 分塊的一列及
  $B$ 分塊的一列。
- 內部迴圈中，`a_tile[ty][kk]` 對整個 warp 是同一位址（broadcast），
  `b_tile[kk][tx]` 則沿列讀取（無衝突）。
- 把越界元素填零，可讓內部迴圈不需邊界檢查：零不會改變總和。

### 3.5 流量與新瓶頸

每個 block 在每個階段載入 $2T^2$ 個 float，並執行 $T^3$ 個 FMA：

$$
Q_{\text{tiled}} = \underbrace{\frac{MN}{T^2}}_{\text{blocks}}\cdot\underbrace{\frac{K}{T}}_{\text{phases}}\cdot\underbrace{2T^2\cdot 4}_{\text{bytes per phase}} = \frac{8MNK}{T}, \qquad
I_{\text{tiled}} = \frac{2MNK}{8MNK/T} = \frac{T}{4}
$$

| 符號 | 意義 |
|---|---|
| $T$ | 分塊寬度（此處為 32） |
| $Q_{\text{tiled}}$ | 從全域記憶體（L2/DRAM）載入的總位元組數 |
| $I_{\text{tiled}}$ | 全域記憶體強度：全域流量減少 $T$ 倍 |

當 $T = 32$，L2/DRAM 強度為 8 flop/byte。新瓶頸是共享記憶體：內部迴圈
仍然**每個 FMA 需要一次共享載入**（`a_tile` broadcast 幾乎免費，但
`b_tile` 讀取不是），而 SM 每週期能發出的共享載入遠少於 FMA
（A100 SM 每週期可執行 64 個 FP32 FMA，但只能從共享記憶體讀取 32 個
word）。此 kernel 通常達到峰值的 10–20 %。

## 4. 暫存器分塊 { #4-register-tiling }

### 4.1 Outer Product

讓每個 thread 負責一個保存在暫存器中的 $t_M\times t_N$ 輸出區塊。每個
$k$ 中，thread 從共享記憶體載入 $t_M$ 個 $A$ 值及 $t_N$ 個 $B$ 值，
並執行 $t_Mt_N$ 個 FMA（outer product）：

![暫存器分塊：每個 k 步驟中，thread 載入 4 個 A 值與 4 個 B 值，並執行 16 個 FMA](figures/ch04-register-tile.svg)

### 4.2 數字

$$
\frac{\text{FMAs}}{\text{shared loads}} = \frac{t_M t_N}{t_M + t_N}, \qquad
I_{\text{L2}} = \frac{2\,B_MB_NB_K}{4\,B_K\,(B_M + B_N)} = \frac{B_MB_N}{2\,(B_M + B_N)}
$$

| 符號 | 意義 |
|---|---|
| $t_M, t_N$ | 每個 thread 沿 $M$、$N$ 方向負責的輸出數（暫存器分塊） |
| $B_M, B_N$ | 每個 block 的輸出數（block 分塊） |
| $B_K$ | 一次放入共享記憶體的 K 切片深度 |
| $I_{\text{L2}}$ | 從 L2/DRAM 載入共享記憶體時，每位元組可執行的 flop 數 |

| 配置 | 每次共享載入的 FMA 數 | $I_{\text{L2}}$（flop/B） |
|---|---|---|
| $T = 32$，每個 thread 一個輸出 | 1（兩個載入都計入則為 0.5） | 8 |
| $64\times64$ block，每個 thread $4\times4$ | 2 | 16 |
| $128\times128$ block，每個 thread $8\times8$ | 4 | 32 |

兩個比例都是第 1.2 節的重用公式，分別套用於 thread 與 block 層級。

### 4.3 內部迴圈

Tensara matmul 頁面採用 $64\times64$ / $4\times4$ 版本
（[Tensara – 矩陣乘法](../tensara/matrix-multiplication/)）。其內部迴圈為：

```cpp
#pragma unroll
for (int kk = 0; kk < kTileK; ++kk) {
    float a_frag[4], b_frag[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) a_frag[i] = a_tile[kk][ty + 16 * i];   // A stored transposed
#pragma unroll
    for (int j = 0; j < 4; ++j) b_frag[j] = b_tile[kk][tx + 16 * j];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
}
```

### 4.4 重要細節

- **轉置 $A$ 分塊**（`a_tile[k][m]`）：兩個 operand 都能沿共享記憶體的列讀取。
- **Stride-16 歸屬**（`ty + 16 * i`、`tx + 16 * j`），而非四個相鄰元素：
  每次儲存指令中，16 個連續 lane 會寫入 16 個連續欄（coalesced），共享
  讀取也不會衝突。
- **共享分塊的 padding**（`[kTileK][64 + 4]`）可避免轉置儲存時的衝突。
- **完全展開**可讓 `acc[4][4]` 留在暫存器；動態索引會迫使它移至
  local memory。

### 4.5 代價：暫存器

$t_M\times t_N$ 分塊需要 $t_Mt_N$ 個累加器、$t_M + t_N$ 個 fragment
暫存器，還有位址：$4\times4$ 約需 40 個，$8\times8$ 約需 120 個。
每個 thread 的暫存器越多，駐留 warp 越少（第 01 章第 6 節）；只要每個
warp 有足夠的獨立 FMA 能自行隱藏延遲，這並無問題。FP32 超過
$8\times8$ 後，累加器放不下，編譯器會 spill。

## 5. 最佳化階梯的其餘部分

以下每項技巧都在 [GEMM 深入解析](gemm/README.md)中有獨立頁面，並附上
可在 CPU 模擬器測試的完整程式：

| 技巧 | 幫助 | 頁面 |
|-----------|-----|---|
| `float4` 共享載入（`LDS.128`） | Fragment 共享載入指令減少 4 倍 | [04.1](gemm/01-vectorized-loads.md) |
| 雙緩衝（兩組分塊） | 運算切片 $s$ 時載入 $s+1$；每個切片由兩個 barrier 減為一個 | [04.2](gemm/02-double-buffering.md) |
| `cp.async`（sm_80+）/ TMA（sm_90） | 不經暫存器、非同步地從全域記憶體複製到共享記憶體 | [04.3](gemm/03-async-copies.md) |
| Warp 分塊 | 每個 warp 負責 $64\times32$ 子分塊；對應 block → warp → thread 硬體階層，並提高暫存器重用 | [04.4](gemm/04-warp-tiling.md) |
| Swizzle 分塊順序（「grouped」啟動） | 同時執行的 block 可在 L2 共用 $A$ 列與 $B$ 欄 | [04.5](gemm/05-tile-swizzling.md) |
| Split-K / Stream-K | 當 $M\cdot N$ 的分塊太少，無法填滿 GPU 時提高平行度 | [04.6](gemm/06-split-k-stream-k.md) |
| Tensor core（WMMA、`mma.sync`、`wgmma`、CUTLASS/CuTe） | FP16/BF16/TF32/FP8 的 FLOP 提高 8–16 倍；整個資料流都會改變 | [04.7](gemm/07-tensor-cores.md) |

在同一台 GPU 上，相對 cuBLAS FP32 的典型進展：

| Kernel | cuBLAS 效能比例 |
|---|---|
| 簡單版本 | 1–5 % |
| 共享記憶體分塊 | 10–20 % |
| $4\times4$ 暫存器分塊 | 40–60 % |
| $8\times8$、`float4`、雙緩衝、warp 分塊 | 80–95 % |

![最佳化階梯每一階通常可達到的 cuBLAS FP32 吞吐量比例](figures/ch04-ladder.svg)

第 05–07 章會在 AMD 硬體上延續此主題，介紹 matrix-core 指令、手寫組合語言
kernel 與 kernel 產生器。

## 6. 融合 Epilogue

實際 GEMM 很少只計算 $AB$。一般形式為

$$
C \leftarrow f\bigl(\alpha\,AB + \beta\,C + \mathbf{1}\,b^{\mathsf T}\bigr)
$$

| 符號 | 意義 |
|---|---|
| $\alpha, \beta$ | 純量（BLAS 慣例） |
| $b$ | 長度為 $N$ 的 bias 向量，加入每一列 |
| $f$ | 逐元素 activation（ReLU、GELU、SiLU……） |

$K$ 迴圈後的所有工作都在累加器仍位於暫存器時執行，成本幾乎為零：

```cpp
// After the K loop: acc[i][j] holds (AB) for row r_i, column c_j of this thread.
for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
        const float v = alpha * acc[i][j] + beta * c[r_i * n + c_j] + bias[c_j];
        c[r_i * n + c_j] = fmaxf(v, 0.0f);          // ReLU
    }
```

若改用另一個 kernel 執行相同操作，就必須再次讀寫 $C$（$8MN$ 位元組）。
對 LLM 推論的窄長 GEMM，這些額外流量的成本可能與 GEMM 本身相同，因此
函式庫會提供「fused epilogue」（[Tensara – GEMM + ReLU](../tensara/gemm-relu/)、
第 07 章 TensileLite 的 activation fusion）。

## 7. 檢查清單

- `threadIdx.x` ↔ 欄，以便 coalesce $B$ 讀取與 $C$ 寫入。
- 準備邊界分塊時，對越界元素填零，不要略過載入（如此 FMA 迴圈便不需
  邊界檢查）。
- 單緩衝時每個切片使用兩次 `__syncthreads()`：載入後一次，下一次載入
  覆寫分塊前一次。
- 當 $MK$、$KN$ 或 $MN$ 可能超過 $2^{31}$ 個元素時，使用 `size_t`
  索引（例如 [Tensara – Matmul 3D](../tensara/matmul-3d/) 中
  $64\cdot4096 \times 4096$ 的 activation）。
- 即使輸入為 FP16/BF16，也使用 FP32 累加。
- 每次修改分塊大小後，都用 `-Xptxas -v` 檢查 spill。

## 重點整理

1. GEMM 對 $O(n^2)$ 資料執行 $O(n^3)$ 工作；只有每個載入元素都重複使用許多次時，才會受運算能力限制。
2. 負責 $T_M\times T_N$ 分塊的工作單位，每載入一個值可執行 $T_MT_N/(T_M+T_N)$ 個 FMA。把它套用到 block 層級（共享記憶體）與 thread 層級（暫存器）。
3. 共享記憶體分塊解決 DRAM 流量，但仍是每個 FMA 一次共享載入；暫存器分塊可解決此問題。
4. Barrier 會保護兩個方向：資料就緒（載入後）與緩衝區可用（運算後）。
5. 結果仍在暫存器時就融合 epilogue。

## 練習

1. 對 $128\times64$ block 分塊與 $8\times4$ thread 分塊，計算每次共享
   載入的 FMA 數及 $I_{\text{L2}}$。

    <details markdown="1"><summary>答案</summary>

    Thread：每次載入 $32/12 \approx 2.7$ 個 FMA。
    Block：$I_{\text{L2}} = 128\cdot64 / (2\cdot192) \approx 21.3$ flop/B。

    </details>

2. 從 `matmulTiled` 移除第二個 `__syncthreads()`，並在
   `CUEMU_REVERSE=1` 下以 cuemu 執行。會發生什麼事？為何 GPU 執行不是
   可靠的測試？

    <details markdown="1"><summary>答案</summary>

    快速 thread 會在慢速 thread 仍在讀取時覆寫分塊，因此部分 partial sum
    會使用下一個切片的資料。GPU 上的 warp 通常在時間上相距不遠，所以
    錯誤只會偶爾出現。

    </details>

3. $16\times8$ FP32 thread 分塊的累加器需要多少暫存器？為何這是問題？

    <details markdown="1"><summary>答案</summary>

    128 個累加器加上 24 個 fragment 值與位址：超過 160 個暫存器，因此
    每個 SM 只能容納 1–2 個 256-thread block，而且通常會 spill。
    若要提高每個暫存器的工作量，應使用 tensor core
    （[04.7](gemm/07-tensor-cores.md)）。

    </details>

## 實作練習

- [LeetGPU – 矩陣乘法](../leetgpu/002-matrix-multiplication/)
- [LeetGPU – GEMM](../leetgpu/022-gemm/)
- [Tensara – 矩陣乘法](../tensara/matrix-multiplication/)
- [Tensara – GEMM + ReLU](../tensara/gemm-relu/)（fused epilogue）
