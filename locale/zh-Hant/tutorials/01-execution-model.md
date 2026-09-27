# 01 – CUDA 執行模型

> **第一部分 · CUDA 基礎** · 先備知識：[00 – 開始使用](00-getting-started.md) ·
> 下一章：[02 – 記憶體階層](02-memory-hierarchy.md)

CUDA kernel 是以*單一*執行緒的程式來撰寫，再同時啟動數百萬份。本章說明這些執行緒如何組織、硬體如何執行它們，以及這對你撰寫的程式碼有何影響。

**你將學到**

- 軟體階層（網格、區塊、warp、執行緒）及其對應的硬體（GPU、SM、warp 排程器、lane）；
- 一維、二維和三維的索引運算，以及網格跨步迴圈；
- SIMT 執行和 warp 分歧的成本；
- GPU 為何需要數萬個執行緒（延遲隱藏與 Little 定律），以及另一種方法（指令層級平行度）；
- 如何計算佔用率，以及為何它是手段而非目標；
- 如何選擇區塊和網格大小；
- 各範圍內的同步和通訊機制，包括 stream。

## 1. 階層

### 1.1 軟體：網格、區塊、Warp、執行緒

```
Grid  ── many Blocks  (scheduled independently onto SMs, in any order)
Block ── up to 1024 Threads (share shared memory, can __syncthreads())
Warp  ── 32 consecutive threads of a block, issued together (SIMT)
```

- **kernel** 是由網格中每個執行緒執行的函式。
- `blockIdx`、`blockDim`、`threadIdx` 和 `gridDim` 是內建 `dim3` 變數，會告訴每個執行緒自己的位置。
- 同一 kernel 內的區塊無法彼此同步（cooperative launch 除外）。若需要全域 barrier，請結束 kernel 再啟動另一個；同一 stream 上的 kernel 會依序執行。

![網格中的區塊會配置到各 SM；區塊中的 warp 則由 SM 的 warp 排程器發出](figures/ch01-hierarchy.svg)

### 1.2 硬體：串流多處理器

GPU 是由一組**串流多處理器**（SM）構成：A100 有 108 個，H100 SXM 有 132 個。每個 SM 分成 4 個**處理區塊**（SM 子分割區），每個處理區塊包含：

- 一個 **warp 排程器**，每個時脈可從一個常駐 warp 發出一條指令；
- 一部分**暫存器檔案**（每個 SM 合計 64 K 個 32 位元暫存器）；
- 執行單元：FP32/INT32 lane、特殊函式單元、載入／儲存單元，以及一個 tensor core。

SM 也包含供其所有區塊使用的 **L1 快取／共享記憶體**（A100 為 192 KB，H100 為 256 KB，可透過設定在兩者間分配）。

### 1.3 兩者如何對應

| 軟體 | 硬體 | 說明 |
|---|---|---|
| 網格 | 整個 GPU | 一次 kernel 啟動 |
| 區塊（CTA） | 一個 SM | 區塊不會遷移；一個 SM 可同時容納多個區塊 |
| Warp | 一個 warp 排程器 slot | 共用一個指令流的 32 個執行緒 |
| 執行緒 | SIMD 單元的一個 lane | 擁有自己的暫存器和 predicate |

kernel 啟動後，區塊排程器會持續把區塊交給 SM，直到各 SM 滿載（第 6 節）；每當一個區塊完成，下一個等待中的區塊就會取代它。順序未指定，因此**正確的程式碼絕不依賴區塊的執行順序**。

### 1.4 SIMT 與獨立執行緒排程

NVIDIA 將此模型稱為*單指令多執行緒*（SIMT）：每個執行緒都有自己的暫存器，而且自 Volta 起也有自己的程式計數器，但一個 warp 每次會為目前執行同一指令的所有 lane *發出*一條指令。這帶來兩項結果：

- 程式碼以單一執行緒為單位撰寫，可使用一般分支和迴圈；硬體會處理走不同路徑的 lane（第 4 節）。
- Warp 中的 lane 不保證以 lock-step 執行。若程式碼會在 lane 間交換資料，必須使用 `*_sync` warp 原語或 `__syncwarp()`，並明確指定 lane mask（整個 warp 為 `0xffffffff`）；絕不可依賴隱含的 lock-step 執行。

## 2. 索引運算

### 2.1 一維

執行緒根據座標找到自己負責的元素：

$$
i = b_x\,B_x + t_x, \qquad
G_x = \left\lceil \frac{n}{B_x} \right\rceil = \left\lfloor \frac{n + B_x - 1}{B_x} \right\rfloor
$$

| 符號 | 意義 |
|---|---|
| $t_x$ | `threadIdx.x`：區塊內的位置 |
| $b_x$ | `blockIdx.x`：區塊在網格中的位置 |
| $B_x$ | `blockDim.x`：區塊大小 |
| $G_x$ | `gridDim.x`：涵蓋 $n$ 個元素所需的區塊數 |
| $i$ | 全域一維索引 |

最後一個區塊通常不完整，因此每個執行緒都必須檢查索引：

```cpp
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

constexpr int kBlockSize = 256;
const int num_blocks = (n + kBlockSize - 1) / kBlockSize;
vectorAdd<<<num_blocks, kBlockSize>>>(d_a, d_b, d_c, n);
```

### 2.2 二維和三維

對矩陣和影像而言，二維區塊與網格可自然對應到列和欄：

$$
\text{row} = b_y\,B_y + t_y, \qquad \text{col} = b_x\,B_x + t_x, \qquad
\text{offset} = \text{row}\cdot\text{ld} + \text{col}
$$

| 符號 | 意義 |
|---|---|
| $t_y, b_y, B_y$ | `threadIdx.y`、`blockIdx.y`、`blockDim.y` |
| Row, col | 全域二維座標 |
| ld | 前導維度：兩列間的元素距離（對密集 row-major 矩陣而言就是寬度） |

`col` 使用 $x$，讓連續執行緒存取連續欄；如此一來，warp 的記憶體存取也會連續（合併存取，第 02 章）。

```cpp
__global__ void matrixAdd(const float* a, const float* b, float* c, int rows, int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) c[row * cols + col] = a[row * cols + col] + b[row * cols + col];
}

const dim3 block(32, 8);                                      // 256 threads
const dim3 grid((cols + 31) / 32, (rows + 7) / 8);
matrixAdd<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
```

### 2.3 從執行緒座標到 Warp

區塊內的執行緒以 $x$ 優先編號，warp 則由連續編號組成：

$$
\tau = t_x + B_x\,(t_y + B_y\,t_z), \qquad w = \left\lfloor \frac{\tau}{32} \right\rfloor, \qquad \ell = \tau \bmod 32
$$

| 符號 | 意義 |
|---|---|
| $\tau$ | 區塊內的線性執行緒索引 |
| $w$ | 區塊內的 warp 索引（`warp_id`） |
| $\ell$ | warp 內的 lane 索引（`lane`） |

因此 $32\times8$ 區塊有 8 個 warp，每個 warp 都是一整列 $t_x$ 值。$16\times16$ 區塊也有 8 個 warp，但每個 warp 會涵蓋*兩*列。

![左：由區塊與執行緒索引求出執行緒的全域列與欄。右：warp 如何從 16 × 16 和 32 × 8 區塊中切分出來](figures/ch01-indexing.svg)

### 2.4 索引寬度

`int` 最多可容納 $2^{31} - 1$ 的索引。$50\,000\times50\,000$ 矩陣有 $2.5\cdot10^9$ 個元素，`row * cols` 會在未警告的情況下溢位。只要乘積可能超過此值，就應立即以 64 位元計算位移：

```cpp
const size_t offset = static_cast<size_t>(row) * cols + col;
```

64 位元運算需要額外指令，因此確知大小不大的 kernel 仍可用 `int` 進行逐執行緒運算。

## 3. 網格跨步迴圈

讓網格大小與問題大小分離：

```cpp
__global__ void relu(const float* in, float* out, size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = fmaxf(in[i], 0.0f);
    }
}
```

每個執行緒處理

$$
k = \left\lceil \frac{n - i_0}{G_x B_x} \right\rceil \quad \text{elements}, \qquad i_0 = b_x B_x + t_x
$$

| 符號 | 意義 |
|---|---|
| $k$ | 第一個索引為 $i_0$ 的執行緒所執行的迭代次數 |
| $G_x B_x$ | 執行緒總數，也就是迴圈跨步 |

優點：

- 適用任何 $n$，包括 $n > 2^{31}$（請用 `size_t`）；
- 網格可依機器大小設定（每個 SM 執行數波區塊），不必依資料大小設定；
- 逐執行緒的設定成本（載入常數、初始化累加器）可攤提到 $k$ 個元素；歸約就是藉此建立各執行緒的部分總和（第 03 章）。

每次迭代中，整個網格會涵蓋一段連續的 $G_xB_x$ 個元素，因此 warp 的存取在每次迭代都保持連續（合併存取）。

## 4. Warp 與分歧

### 4.1 分歧分支如何執行

warp 的 32 個 lane 會執行同一條指令。當 lane 選擇不同分支時，warp 會依序執行每條路徑，並遮罩其他 lane，最後再一起繼續：

```cpp
if (threadIdx.x % 2 == 0) a();   // pass 1: even lanes active, odd lanes masked
else                      b();   // pass 2: odd lanes active, even lanes masked
c();                             // all lanes again
```

$$
T_{\text{warp}} \approx \sum_{p \in \text{paths taken}} T_p, \qquad
\eta_{\text{branch}} = \frac{\text{active lanes per issued instruction}}{32}
$$

| 符號 | 意義 |
|---|---|
| $p$ | 至少一個 lane 採取的不同控制流程路徑 |
| $T_p$ | 執行路徑 $p$ 的時間 |
| $\eta_{\text{branch}}$ | 有效 lane 的平均比例（Nsight Compute：「thread instruction executed / warp instruction executed」） |

### 4.2 分歧何時沒有成本、何時有成本

- 在整個 warp 中**一致**的分支（`if (warp_id == 0)`、`if (blockIdx.x < k)`）不會增加成本：只會採取一條路徑。
- 短分支（`x > 0 ? x : a * x`）會編譯成 predicated select，不會真正分歧。
- 迭代次數依 lane 而異的迴圈，會一直執行到最長的 lane 完成。
- 經典錯誤是依*執行緒*的奇偶或模數分配工作（`tid % 2`），使每個 warp 都發生分歧。改依 *warp* 分配（`tid / 32 % 2`）即可在不分歧的情況下完成相同工作（第 03 章第 4 節會以歸約示範）。

## 5. 為何需要這麼多執行緒：延遲隱藏

### 5.1 延遲

運算需要多個週期才能完成。A100 的概略數字如下：

| 運算 | 延遲（週期） |
|---|---|
| 相依 FP32 FMA | ~4 |
| 共享記憶體載入 | ~20–30 |
| L2 命中 | ~200 |
| DRAM（HBM）載入 | ~400–800 |

需要載入結果的 warp 在結果抵達前無法繼續。GPU 不會等待：每個週期，各 warp 排程器都會選擇下一條指令已就緒的 warp 並發出指令。

![當一個 warp 等待記憶體時，排程器會發出其他 warp 的指令](figures/ch01-latency-hiding.svg)

### 5.2 Little 定律

要讓記憶體系統保持忙碌，進行中的要求必須足以涵蓋延遲：

$$
N_{\text{bytes in flight}} = \beta \times L, \qquad
N_{\text{warps}} \gtrsim \frac{\beta\,L}{n_{\text{SM}}\cdot b_{\text{warp}}}
$$

| 符號 | 意義 |
|---|---|
| $\beta$ | DRAM 頻寬（位元組/秒） |
| $L$ | 記憶體延遲（秒） |
| $N_{\text{bytes in flight}}$ | 任一時刻已要求但尚未傳回的位元組數 |
| $n_{\text{SM}}$ | SM 數量 |
| $b_{\text{warp}}$ | 每個 warp 進行中的位元組數（例如每個 lane 載入一個 `float4` 時為 512） |
| $N_{\text{warps}}$ | 每個 SM 所需的常駐 warp 數 |

對 A100 而言（$\beta \approx 1.5$ TB/s、$L \approx 500$ ns、108 個 SM）：$\beta L \approx 750$ KB，也就是每個 SM 約 7 KB，或約 14 個 warp 且各有一筆 512 位元組載入尚未完成。

### 5.3 另一個手段：指令層級平行度

若每個 warp 同時有更多獨立載入進行中，就只需較少 warp。展開迴圈和 `float4` 載入都能提高 $b_{\text{warp}}$：

```cpp
// Four independent loads are issued back to back; the warp waits once, not four times.
const float4 v0 = in4[i], v1 = in4[i + stride], v2 = in4[i + 2 * stride], v3 = in4[i + 3 * stride];
```

這種**指令層級平行度**（ILP）是佔用率以外的另一種方法，也是快速 GEMM 所依賴的方法：它只執行少量 warp，但每個 warp 都有許多獨立 FMA 和載入；這些技巧可見於矩陣乘法 1 及整條矩陣乘法路徑。

## 6. 佔用率

### 6.1 公式

一個 SM 可執行的區塊數取決於其資源：

$$
n_{\text{blocks/SM}} = \min\left(
\left\lfloor \frac{T_{\max}}{B} \right\rfloor,\
\left\lfloor \frac{R_{\text{SM}}}{r\,B} \right\rfloor,\
\left\lfloor \frac{S_{\text{SM}}}{s} \right\rfloor,\
n_{\max}\right), \qquad
\text{occupancy} = \frac{n_{\text{blocks/SM}}\cdot B}{T_{\max}}
$$

| 符號 | 意義 |
|---|---|
| $B$ | 每區塊執行緒數 |
| $T_{\max}$ | 每個 SM 的最大常駐執行緒數（A100/H100 為 2048） |
| $R_{\text{SM}}$ | 每個 SM 的暫存器數（65 536） |
| $r$ | 每執行緒暫存器數（來自 `-Xptxas -v`；以區塊為單位配置） |
| $S_{\text{SM}}$ | SM 可供區塊使用的共享記憶體（A100 最高約 164 KB，H100 約 228 KB） |
| $s$ | 每個區塊的共享記憶體（靜態加動態） |
| $n_{\max}$ | 每個 SM 的最大常駐區塊數（32） |
| 佔用率 | 使用中的 SM 執行緒 slot 比例 |

### 6.2 計算範例

在 A100 上，$B = 256$、$r = 64$、$s = 32$ KB 會得到 $\min(8, 4, 5, 32) = 4$ 個區塊，也就是 1024 個執行緒、50 % 佔用率：

![每項資源都會限制常駐區塊數；最小的限制決定佔用率](figures/ch01-occupancy.svg)

限制來自暫存器；`__launch_bounds__(256, 6)` 會要求編譯器最多使用 40 個暫存器（但可能造成 spill）。

實際數字可能因兩項細節而略有不同：暫存器會以每 warp 256 個為單位配置（所以 $r$ 實際上會向上取整到 8 的倍數），共享記憶體則以數百位元組為單位配置，且每個區塊保留約 1 KB。

### 6.3 詢問 Runtime

```cpp
int blocks_per_sm = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, myKernel, kBlockSize, dynamic_smem_bytes);
```

Nsight Compute 的 *Occupancy* 小節會顯示相同數字，以及限制它的資源。

### 6.4 佔用率是手段，不是目標

佔用率只在用來隱藏延遲時才有意義。使用暫存器分塊的 GEMM 即使只有 12–25 % 佔用率，仍能有很好的效能，因為每個執行緒都有許多獨立 FMA 和載入正在進行（ILP，第 5.3 節）。相反地，每個執行緒只有一次相依載入且受記憶體限制的 kernel，則需要高佔用率。只有當分析器顯示 warp 正在停滯等待且沒有其他工作可發出時，才應提高佔用率；不要盲目犧牲暫存器（也就是重複使用能力）來換取佔用率。

## 7. 選擇區塊和網格大小

### 7.1 區塊大小

- 使用 32 的倍數（不完整的 warp 會浪費 lane）。典型值是 128–512；256 是安全的預設值。
- 對二維問題，讓 $x$ 範圍至少為 32，使每個 warp 涵蓋一段連續列（$32\times8$ 區塊）。
- 對區塊範圍歸約，較大區塊代表較少部分結果，但樹也較長；常見值為 256–1024。
- 每執行緒使用大量共享記憶體或暫存器的 kernel 通常偏好 128 個執行緒，讓一個 SM 能容納多個區塊。

### 7.2 網格大小與尾端效應

每個 SM 至少要啟動數個區塊。若有 $G$ 個區塊，且每個 SM 可常駐 $c$ 個區塊，工作會分波執行：

$$
\text{waves} = \left\lceil \frac{G}{c\,n_{\text{SM}}} \right\rceil, \qquad
\eta_{\text{tail}} = \frac{G}{c\,n_{\text{SM}}\cdot\text{waves}}
$$

| 符號 | 意義 |
|---|---|
| $G$ | 網格中的區塊數 |
| $c$ | 每個 SM 的常駐區塊數（第 6 節） |
| $\eta_{\text{tail}}$ | 若所有區塊執行時間相同，SM 時間的使用比例 |

1.1 波的網格會浪費第二波近一半的容量（$\eta \approx 55\%$）；10.1 波只浪費少量容量（$\eta \approx 92\%$）。許多小區塊（或具有數波工作量的網格跨步迴圈）可縮短尾端；若區塊必須很大，請參閱 Stream-K（[矩陣乘法 7](gemm/06-split-k-stream-k.md)）。

## 8. 同步與通訊

### 8.1 各範圍的機制

| 範圍 | 機制 |
|---|---|
| Warp | `__shfl_*_sync`、`__ballot_sync`、`__syncwarp()` |
| 區塊 | 共享記憶體 + `__syncthreads()` |
| 網格 | Kernel 邊界；atomic（`atomicAdd`、`atomicMax`……）；以 cooperative launch 啟動的 cooperative groups `grid.sync()` |
| 主機 | `cudaDeviceSynchronize()`、event、stream 排序 |

### 8.2 Barrier

`__syncthreads()` 會等待區塊中的每個執行緒抵達，並讓區塊看見它之前的所有共享與全域記憶體寫入。區塊內**每個**執行緒都必須抵達：放在 `if (threadIdx.x < 16)` 中的 barrier 會造成死結或資料損壞。意外發生此問題最常見的原因，是在 barrier 前提早 `return`。

### 8.3 區塊間的記憶體順序

除非明確指定，否則不保證一個區塊的寫入會依任何順序對另一個區塊可見。`__threadfence()` 會確保一個執行緒先前的寫入在後續寫入前，對整個裝置可見。標準模式是「寫入資料、fence，再以 atomic 設定旗標」；讀取端檢查旗標、fence，再讀取資料。第 03 章（第 5 節）和 [矩陣乘法 7](gemm/06-split-k-stream-k.md) 的 Stream-K kernel 都會使用此模式。

### 8.4 Atomic

全域 atomic 在 L2 快取中執行。不同位址時速度很快，但許多執行緒存取同一位址時會序列化。可先合併資料來降低競爭（先進行 warp 或區塊歸約，再讓每個區塊執行一次 atomic）；也請記住，浮點 atomic 會使加總順序和結果的最後幾個位元不具決定性。

## 9. Stream 與非同步執行

**stream** 是依序執行的 GPU 工作佇列；不同 stream 中的工作可能重疊。

- 啟動和 `cudaMemcpyAsync` 會立即返回；主機可繼續執行。
- *舊式預設 stream*（stream 0）會與所有其他 blocking stream 同步，因此簡單程式看起來會循序執行。
- 若要讓傳輸與計算重疊，需要**釘選**主機記憶體（`cudaMallocHost`）、`cudaMemcpyAsync` 和非預設 stream。
- **Event**（`cudaEventRecord` / `cudaStreamWaitEvent`）可表達 stream 間的相依關係，也可計算工作時間（第 00 章）。

實作題通常會在預設 stream 上執行一個 kernel（或簡短序列），所以很少需要 stream；但實際應用程式會需要。

## 10. 完整範例：向量化向量加法

```cpp
constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 4096;

__global__ void vectorAddVec4(const float* a, const float* b, float* c, size_t n) {
    const size_t num_vec4 = n / 4;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);
    for (size_t i = start; i < num_vec4; i += stride) {
        const float4 x = a4[i], y = b4[i];
        c4[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
    for (size_t i = num_vec4 * 4 + start; i < n; i += stride) c[i] = a[i] + b[i];   // tail
}
```

它結合了以上所有內容：

1. 使用網格跨步迴圈，因此任何 $n$ 都適用；
2. 使用 16 位元組存取，因此每個 warp 有更多位元組正在傳輸（第 5.3 節）；
3. 限制網格大小（$\min(\lceil n/4/256\rceil, 4096)$ 個區塊），只執行數波；
4. 使用 64 位元索引；
5. 若 $n$ 無法被 4 整除，使用純量處理尾端。

包含成本分析的完整解答請見 [Tensara – 向量加法](../tensara/vector-addition/)。

## 重點整理

1. 撰寫單一執行緒的程式；網格、區塊和 warp 結構會決定它如何對應到 SM、排程器和 lane。
2. 讓 `threadIdx.x` 沿連續維度移動，並檢查每個索引。
3. 分歧成本等於 warp 採取的各路徑成本總和；請盡量讓分支在 warp 內一致。
4. 平行度能隱藏延遲：可使用許多 warp（佔用率），或讓每個 warp 有許多獨立運算（ILP）。Little 定律可算出所需數量。
5. 佔用率會受執行緒、暫存器、共享記憶體和區塊 slot 限制；請計算它，但應依分析器顯示的停滯原因最佳化，而非只追求數字。
6. 區塊中的每個執行緒都必須抵達 barrier；跨區塊排序則需要 fence 和 atomic。

## 練習

1. 某 kernel 在 A100 上使用每執行緒 96 個暫存器，以及每個 256 執行緒區塊 20 KB 的共享記憶體。其佔用率是多少？限制來自哪裡？

    <details markdown="1"><summary>答案</summary>

    執行緒：$\lfloor 2048/256 \rfloor = 8$。暫存器：
    $\lfloor 65536 / (96\cdot256) \rfloor = 2$。共享記憶體：
    $\lfloor 164/20 \rfloor = 8$。因此為 2 個區塊 = 512 個執行緒 = 25 %，
    限制來自暫存器。

    </details>

2. 網格有 250 個區塊，每個 SM 可容納 2 個，而 GPU 有 108 個 SM。共有幾波？$\eta_{\text{tail}}$ 是多少？

    <details markdown="1"><summary>答案</summary>

    $250 / 216 = 1.16$，所以有 2 波；$\eta = 250 / 432 \approx 58\%$。若為 216 或 432 個區塊，就沒有尾端。

    </details>

3. 在 `if (threadIdx.x % 4 == 0) x = expensive(x);` 中，執行 `expensive` 時有多少比例的 lane 在做有效工作？請重寫工作分配，使每個 warp 都採取一致路徑。

    <details markdown="1"><summary>答案</summary>

    32 個 lane 中有 8 個（25 %）。改將高成本工作分配給四分之一的 *warp*（`(threadIdx.x / 32) % 4 == 0`），並讓每個 warp 處理四倍的元素。

    </details>

4. 使用 Little 定律估算 H100 SXM（$\beta = 3.35$ TB/s、$L \approx 600$ ns）必須有多少位元組正在傳輸。若每個 lane 載入一個 `float4`，相當於每個 SM 有多少 warp？

    <details markdown="1"><summary>答案</summary>

    $\beta L \approx 2$ MB，分散到 132 個 SM 後每個 SM 約 15 KB，也就是約 30 個各有 512 位元組的 warp；或約 8 個各有四次獨立 `float4` 載入的 warp。

    </details>

## 實作練習

- [LeetGPU – 向量加法](../leetgpu/001-vector-add/)
- [Tensara – 向量加法](../tensara/vector-addition/)
- [LeetGPU – 矩陣加法](../leetgpu/008-matrix-addition/)（二維索引）
- [LeetGPU – 反轉陣列](../leetgpu/019-reverse-array/)（原地操作，只用一半執行緒）
