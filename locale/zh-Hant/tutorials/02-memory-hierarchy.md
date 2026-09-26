# 02 – 記憶體階層與合併存取

> **第一部分 · CUDA 基礎** · 先備知識：[01 – 執行模型](01-execution-model.md) ·
> 下一章：[03 – 平行歸約](03-parallel-reduction.md)

實作題中的大多數 kernel 都**受記憶體限制**：速度取決於搬移多少位元組，以及搬移效率。GPU 執行算術的速度比 DRAM 供應資料快 10–20 倍，因此關鍵是讓每個位元組只移動一次、使用硬體允許的最寬且最規律方式，並盡可能從更快的記憶體重複使用資料。

**你將學到**

- 各記憶體空間（暫存器、區域、共享、L1、L2、全域、常數）的範圍、大小和延遲；
- 合併存取：warp 的位址如何轉成 DRAM transaction，以及如何計算存取模式效率；
- 資料布局（AoS 與 SoA）和對齊；
- 向量化（`float4`）存取；
- 共享記憶體 bank、衝突，以及兩種解法：padding 和 XOR swizzling；
- 一個完整的分塊轉置，逐行分析；
- 哪些分析器指標可判斷 kernel 使用記憶體的效率。

## 1. 記憶體空間

### 1.1 概覽

| 記憶體 | 範圍 | 延遲（約） | 大小（A100，約） | 說明 |
|--------|-------|-------------------|------|-------|
| 暫存器 | 執行緒 | ~1 週期 | 每個 SM 256 KB | Spill 會進入「區域」記憶體（較慢，快取於 L1/L2）。 |
| 共享記憶體 | 區塊 | ~20–30 週期 | 每個 SM 最高 164 KB | 由程式設計師管理，有 32 個 bank。與 L1 共用儲存空間。 |
| L1 快取 | SM | ~30 週期 | 每個 SM 192 KB（含共享記憶體） | 自動運作；快取全域載入。 |
| L2 快取 | 裝置 | ~200 週期 | 40 MB | 自動運作；所有 SM 共用；atomic 在此處完成。 |
| 全域（HBM / GDDR） | 裝置 | ~400–800 週期 | 40–80 GB | 容量大、頻寬高、延遲高。 |
| 常數 | 裝置，唯讀 | 有快取 | 64 KB | 所有 lane 讀取相同位址時速度很快（廣播）。 |

![A100 的記憶體階層，從暫存器到 HBM](figures/ch02-memory-levels.svg)

不同世代的數字各異，但比例才是重點：每往下一層，速度大約慢一個數量級；DRAM 頻寬約比 SM 執行算術的速率低 10–20 倍。

### 1.2 暫存器與區域記憶體

暫存器是最快的儲存空間，也是 FMA 能直接讀取的唯一位置。每個執行緒最多可使用 255 個；實際數量由編譯器決定。以下兩種情況會把資料從暫存器移到**區域記憶體**（每執行緒記憶體，實際位於全域記憶體，快取於 L1/L2）：

- **Spill**：kernel 所需的存活值超過暫存器容量（`-Xptxas -v` 會回報「bytes spill stores」）。
- **動態索引陣列**：例如 `float acc[8]; acc[i] += ...`，而編譯器無法在編譯時決定 `i`。暫存器無法於執行時索引，所以陣列會進入記憶體。完全展開且索引為常數的迴圈則可將它留在暫存器。

### 1.3 共享記憶體

共享記憶體是片上 SRAM，以區塊為單位配置，且區塊中的所有執行緒都能看見：它是由程式設計師管理的快取。

```cpp
__shared__ float tile[32][33];                    // static size
extern __shared__ float dynamic_smem[];           // size given at launch:
myKernel<<<grid, block, bytes>>>(...);            //   third launch parameter
```

它可用來：(1) 重複使用只從全域記憶體載入一次的資料（分塊，第 04 章）；(2) 在區塊中的執行緒間交換資料（歸約，第 03 章）；(3) 重排存取，讓全域存取保持合併（第 5 節的轉置）。需要超過 48 KB 的區塊必須使用動態共享記憶體，並透過 `cudaFuncSetAttribute` 選擇啟用。

### 1.4 L1 與 L2

- **L1** 屬於各 SM，並與共享記憶體共用儲存空間。它會快取全域載入（現今 GPU 預設如此）和區域記憶體。不同 SM 的 L1 不具一致性：另一個 SM 寫入後，某 SM 的 L1 快取值不會更新。
- **L2** 由所有 SM 共用，也是維持一致性的地方：所有全域流量和 atomic 都會經過它。容量達 40–50 MB，可容納許多問題的完整工作集，因此很快再次讀取的資料通常會來自 L2，速度是 DRAM 的數倍。

### 1.5 全域記憶體

全域記憶體（資料中心 GPU 使用 HBM，消費級 GPU 使用 GDDR）是 `cudaMalloc` 配置的位置。它的頻寬很高（1.5–3.35 TB/s），但延遲也很高，而且會以固定大小區塊存取；第 2 節會討論此機制。

### 1.6 常數記憶體

`__constant__` 變數（合計 64 KB）會透過小型常數快取讀取。當 warp 的所有 lane 讀取**相同**位址時，常數載入很快（以廣播傳送）；若讀取不同位址，則會序列化。Kernel 參數也位於常數記憶體。它適合濾波器係數、小型查找表，或所有執行緒共用的純量。

## 2. 合併存取

### 2.1 Sector 與快取列

warp 的全域載入會拆成 **32 位元組 sector**（四個 sector 組成一條 128 位元組快取列）。只要至少有一個 lane 存取某 sector，硬體就會抓取整個 sector，不論實際用了幾個位元組。若 warp 中 lane $\ell$ 在位址 $a_0 + \ell\,s\,e$ 讀取 $e$ 個位元組：

$$
n_{\text{sectors}} \approx \min\left(32,\ \left\lceil \frac{32\,s\,e}{32} \right\rceil\right) \ \ (\text{aligned } a_0), \qquad
\eta = \frac{32\,e}{32\,n_{\text{sectors}}}
$$

| 符號 | 意義 |
|---|---|
| $\ell$ | Lane 索引，$0 \dots 31$ |
| $e$ | 每個 lane 的位元組數（`float` 為 4，`float4` 為 16） |
| $s$ | 連續 lane 間的跨步，以元素計 |
| $a_0$ | lane 0 讀取的位址 |
| $n_{\text{sectors}}$ | 整個 warp 抓取的 32 位元組 sector 數 |
| $\eta$ | 效率：有效位元組除以抓取位元組 |

### 2.2 常見模式

![一次 warp 範圍載入在跨步 1、2 和 32 時所抓取的 sector](figures/ch02-coalescing.svg)

| 模式 | $s$ | Sector | $\eta$ |
|---|---|---|---|
| `x[i]`，`float` | 1 | 4 | 100 % |
| `x[i]`，`float4` | 1 | 16 | 100 %（且指令少 4 倍） |
| `x[2 * i]`，`float` | 2 | 8 | 50 % |
| `x[32 * i]`，`float`（寬度 32 的矩陣的一欄） | 32 | 32 | 12.5 % |
| 偏移 4 位元組，`float` | 1 | 5 | 80 % |

**經驗法則：**讓 `threadIdx.x` 索引變化最快的（連續）維度。若 kernel 必須沿較慢維度讀取（轉置或欄歸約），可以讓*相鄰執行緒*處理相鄰欄，使每次 warp 存取仍然連續（參閱 [Tensara – Argmax](../tensara/argmax/)）；也可以透過共享記憶體暫存資料。

### 2.3 結構陣列與陣列結構

資料布局通常會決定跨步。使用*結構陣列*（AoS）時，warp 讀取連續元素的同一欄位，跨步會等於結構大小：

```cpp
struct Particle { float x, y, z, mass; };          // AoS: 16 bytes per particle
__global__ void updateAos(Particle* p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i].x += 1.0f;                      // stride 16 B: eta = 25 %
}

struct Particles { float *x, *y, *z, *mass; };      // SoA: one array per field
__global__ void updateSoa(Particles p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p.x[i] += 1.0f;                      // stride 4 B: eta = 100 %
}
```

若 kernel 會使用每個元素的*所有*欄位，只要一次載入整個結構，AoS 就沒有問題（此例為每個 particle 一個 `float4`）。若 kernel 每次只使用一個欄位，則應使用 SoA。

交錯布局是特殊情況：32 個 lane 讀取 `rgb[3 * i]` 時，128 個有效位元組會觸及 12 個 sector，但接下來兩條指令（`rgb[3 * i + 1]`、`rgb[3 * i + 2]`）會命中 L1 中相同的快取列，因此 DRAM 流量仍是最佳的（[Tensara – Grayscale](../tensara/grayscale/)）。

### 2.4 對齊

Sector 邊界在位址空間中固定不變。連續但從 sector 中間開始的存取模式，會比必要數量多觸及一個 sector（表格中的「偏移」列）。`cudaMalloc` 傳回 256 位元組對齊的指標，所以問題通常來自位移：例如從第 1 欄開始的子矩陣，或每列位元組長度不是 32 的倍數。函式庫會因此將二維陣列的前導維度（「pitch」）補齊到 32–128 位元組的倍數（`cudaMallocPitch`）。

## 3. 向量化存取

`float4`（或 `int4`、`uint2`……）載入會讓每個 lane 每條指令搬移 16 位元組：

- 每位元組所需的載入和儲存指令較少；
- 每個 warp 有更多位元組正在傳輸，有助隱藏延遲（第 01 章）；
- 需要 16 位元組對齊。`cudaMalloc` 傳回 256 位元組對齊的指標；$M\times K$ 矩陣的一列只有在 $K$ 是 4 的倍數時才會對齊。

```cpp
const float4 v = reinterpret_cast<const float4*>(in)[i];   // i indexes float4s
```

編譯器有時會自行向量化相鄰純量存取，但前提是能證明對齊；明確轉型可確保向量化。務必搭配執行時對齊檢查（或前導維度檢查）和純量備援，就像 [04.1](gemm/01-vectorized-loads.md) 的程式一樣：未對齊的向量存取會造成 fault，而非只是變慢。

## 4. 共享記憶體與 Bank 衝突

### 4.1 Bank

共享記憶體分成 32 個 **bank**，每個寬 4 位元組。連續的 4 位元組 word 會分配到連續 bank：

$$
\operatorname{bank}(a) = \left\lfloor \frac{a}{4} \right\rfloor \bmod 32, \qquad
\text{degree} = \max_{k}\ \bigl\lvert \{\text{distinct words in bank } k \text{ requested by the warp}\} \bigr\rvert
$$

| 符號 | 意義 |
|---|---|
| $a$ | 共享記憶體中的位元組位址 |
| $\operatorname{bank}(a)$ | 提供該位址的 bank |
| Degree | 衝突程度：存取會拆成這麼多筆序列 transaction |

每個 bank 每週期可傳送一個 word，因此 warp 的 32 筆要求若命中 32 個不同 bank，就能一次完成。

### 4.2 廣播

讀取**相同 word** 的 lane 不會衝突：該 word 只讀一次，再廣播（multicast）給所有 lane。讀取**同一 bank 中不同 word** 的 lane 才會衝突。因此「所有 lane 都讀 `s[0]`」沒有額外成本，而「lane $\ell$ 讀 `s[32 * ℓ]`」會發生 32 路衝突。

### 4.3 欄存取問題與 Padding

讀取 `float tile[32][32]` 的一欄是最差情況：元素 $(r, c)$ 位於 word $32r + c$，因此 32 個 lane（不同 $r$、相同 $c$）全都命中 bank $c$，造成 32 路衝突。每列補一個 word 即可修正：

$$
\operatorname{bank}\bigl(\text{tile}[r][c]\bigr) = \bigl(r\,(T + p) + c\bigr) \bmod 32
\ \xrightarrow{\ T = 32,\ p = 1\ }\ (r + c) \bmod 32
$$

| 符號 | 意義 |
|---|---|
| $T$ | Tile 寬度（32） |
| $p$ | 每列補上的 word 數 |
| $r, c$ | Tile 中的列和欄 |

固定 $c$ 並令 $r = 0 \dots 31$ 時，$(r + c) \bmod 32$ 會得到 32 個不同值，因此沒有衝突。

```cpp
__shared__ float tile[kTile][kTile + 1];   // +1 shifts each row by one bank
```

![第 3 欄的 32 個元素所在位置：未 padding 時全在同一 bank；每列補一個 word 後分散到 32 個不同 bank](figures/ch02-bank-conflicts.svg)

### 4.4 XOR Swizzling

Padding 會占用記憶體，也會破壞向量存取所需的對齊（33 個 float 的一列不是 16 位元組對齊）。另一種方法是不補齊列，而以 XOR 置換每列的欄：

$$
c' = c \oplus (r \bmod 32), \qquad
\operatorname{bank}\bigl(\text{tile}[r][c']\bigr) = (c \oplus r) \bmod 32
$$

| 符號 | 意義 |
|---|---|
| $c'$ | 第 $r$ 列邏輯欄 $c$ 的實體儲存欄 |
| $\oplus$ | 位元 XOR |

固定邏輯欄 $c$ 且令 $r = 0 \dots 31$ 時，$c \oplus r$ 的值全都不同，因此讀取一欄不會衝突；讀取一列也仍是 32 個 bank 的置換。Tensor core kernel 會將相同概念套用到 16 位元組區塊而非 word；[04.7](gemm/07-tensor-cores.md#4-swizzled-shared-memory) 有完整推導，而 AMD 產生的 kernel（第 07 章）則會搜尋這類模式。

### 4.5 較寬的存取

warp 的 64 位元共享存取要求 256 位元組，128 位元存取則要求 512 位元組，超過一次跨 32 個 bank 的 128 位元組傳輸。因此硬體會分別以半個 warp（64 位元）或四分之一個 warp（128 位元）處理；規則也變成：每組 16 或 8 個 lane 中，任兩個不同位址都不可共用 bank。無衝突的 128 位元存取需要 4 次傳輸，這是 512 位元組的最低次數。[04.1](gemm/01-vectorized-loads.md) 會說明 GEMM 如何安排 fragment 來符合此規則。

## 5. 完整範例：合併存取的轉置

對 $R\times C$ 矩陣計算 $B = A^{\mathsf T}$。

### 5.1 簡單 Kernel

```cpp
__global__ void transposeNaive(const float* in, float* out, int rows, int cols) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // input column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // input row
    if (x < cols && y < rows) out[static_cast<size_t>(x) * rows + y] = in[static_cast<size_t>(y) * cols + x];
}
```

讀取會合併（連續 `x`），但 warp 的寫入彼此相隔 `rows` 個元素：每個 lane 都寫入自己的 sector，儲存端的 $\eta = 12.5\%$。

### 5.2 透過共享記憶體分塊

透過共享記憶體分塊，可讓兩端都連續：區塊沿著列讀取一個 $32\times32$ tile，再沿著輸出列寫入轉置後的 tile。

![沿列讀取 tile，在共享記憶體中轉置，再沿列寫出](figures/ch02-transpose.svg)

```cpp
constexpr int kTile = 32;
constexpr int kRowsPerPass = 8;
// launch: block(kTile, kRowsPerPass), grid(ceil(cols / kTile), ceil(rows / kTile))
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[kTile][kTile + 1];
    int x = blockIdx.x * kTile + threadIdx.x;              // input column
    for (int dy = threadIdx.y; dy < kTile; dy += kRowsPerPass) {
        const int y = blockIdx.y * kTile + dy;             // input row
        if (x < cols && y < rows) tile[dy][threadIdx.x] = in[static_cast<size_t>(y) * cols + x];
    }
    __syncthreads();
    // Swap the block coordinates so that the write is coalesced too.
    x = blockIdx.y * kTile + threadIdx.x;                  // output column = input row
    for (int dy = threadIdx.y; dy < kTile; dy += kRowsPerPass) {
        const int y = blockIdx.x * kTile + dy;             // output row = input column
        if (x < rows && y < cols) out[static_cast<size_t>(y) * rows + x] = tile[threadIdx.x][dy];
    }
}
```

### 5.3 為何每一行都正確

- **載入**：lane 讀取輸入列中 32 個連續 float（4 個 sector，$\eta = 100\%$）。共享儲存 `tile[dy][threadIdx.x]` 沿著一列進行，沒有衝突。
- **Barrier** 將 tile 的寫入和其他執行緒對轉置 tile 的讀取分開。
- **儲存**：lane 寫入輸出列中 32 個連續 float。共享讀取 `tile[threadIdx.x][dy]` 沿著一欄進行；由於 padding，不會衝突。
- **每個執行緒處理 4 個元素**（以 $32\times8$ 個執行緒處理 $32\times32$ tile），可攤提索引運算，並讓每個 warp 同時進行數次獨立載入。
- **邊緣 tile** 由兩端的邊界檢查處理；tile 中未使用的部分不會寫出。

### 5.4 成本

$$
Q = 2 \cdot 4RC\ \text{bytes}, \qquad T_{\min} = \frac{8RC}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $R, C$ | 輸入的列數和欄數 |
| $Q$ | 必要 DRAM 流量：每個元素讀一次、寫一次 |
| $\beta$ | DRAM 頻寬 |

良好的轉置可達到單純複製頻寬的 80–90 %。簡單版本（合併讀取、寫入跨步為 $R$）的寫入端最多會搬移 8 倍 sector，通常慢 3–5 倍。

## 6. 快取與唯讀資料

- `const T* __restrict__` 告訴編譯器，kernel 執行期間不會透過其他指標寫入資料，因此可使用唯讀（非一致性）路徑，並自由重排載入。
- `__ldg(p)` 可明確強制使用該路徑。
- 在同一指令中由所有執行緒讀取相同位址的資料（濾波 kernel、bias），應放在 `__constant__` 記憶體或以暫存器快取的廣播中。
- 現今資料中心 GPU 的 L2 很大（40–50 MB）。短時間內讀取兩次的 tensor（例如正規化第二個 pass 再次讀取一列）通常會以數倍於 DRAM 的速度來自 L2，因此「雙 pass」kernel 的成本低於位元組數所暗示的成本。
- 自 Ampere 起，可保留部分 L2 給想維持快取的資料（`cudaStreamAttrValue::accessPolicyWindow`），例如供連續 kernel 重複使用的權重矩陣。

## 7. 測量

有效頻寬與第 00 章相同：

$$
\beta_{\text{eff}} = \frac{Q_{\text{read}} + Q_{\text{written}}}{t}, \qquad
\text{efficiency} = \frac{\beta_{\text{eff}}}{\beta_{\text{peak}}}
$$

| 符號 | 意義 |
|---|---|
| $Q_{\text{read}}, Q_{\text{written}}$ | 必要位元組數（不計快取重讀） |
| $t$ | Kernel 時間 |
| $\beta_{\text{peak}}$ | 規格表頻寬，例如 A100 40 GB 約 1.55 TB/s、T4 約 320 GB/s |

Nsight Compute 會回報硬體實際執行的情況：

| 小節／指標 | 能告訴你的事 |
|---|---|
| *Memory Workload Analysis* → DRAM throughput | 實測的 $\beta_{\text{eff}}$，包括重讀 |
| 每次要求的 sector 數（全域載入） | `float` 的理想值是 4，`float4` 是 16；更多代表合併存取不佳 |
| L1 / L2 命中率 | 重讀是否來自快取 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | 共享記憶體 bank 衝突 |
| *Source* 檢視 | 哪些指令未合併或發生衝突 |

## 重點整理

1. 記憶體階層每往下一層約慢 10 倍：請盡可能將資料保留在較高層，且只往下移動一次。
2. 全域記憶體以 32 位元組 sector 抓取；warp 的 lane 應讀取連續位址（讓 `threadIdx.x` 沿連續維度）。
3. 依存取模式選擇資料布局（SoA 或 AoS）。
4. `float4` 存取可將指令數減為四分之一，但需要 16 位元組對齊，並應於執行時檢查。
5. 共享記憶體有 32 個 bank；同一 bank 的不同 word 會序列化，相同 word 則會廣播。使用 padding 或 XOR swizzle 修正欄存取。
6. 即使資料必須重排（轉置），共享記憶體仍可讓 kernel 的讀寫都合併。

## 練習

1. `x` 以 256 位元組對齊時，一個 warp 讀取 `x[4 * i]`（float）。需要多少 sector？$\eta$ 是多少？若每個 lane 改為讀取 `x4[i]` 的一個 `float4` 呢？

    <details markdown="1"><summary>答案</summary>

    跨步 4 個 float = 16 位元組：warp 橫跨 512 位元組 = 16 個 sector，而有效位元組為 128，$\eta = 25\%$。使用 `float4` 時，同樣 16 個 sector 會承載 512 個有效位元組：$\eta = 100\%$。

    </details>

2. 對 `__shared__ float s[32][32]`，`s[threadIdx.x][0]` 的衝突程度是多少？`s[0][threadIdx.x]` 呢？`s[threadIdx.x / 2][0]` 呢？

    <details markdown="1"><summary>答案</summary>

    32（全在 bank 0）；1（32 個不同 bank）；16：lane $2k$ 和 $2k+1$ 讀取相同 word（廣播），但 16 個不同 word 都位於 bank 0。

    </details>

3. 證明使用 $c' = c \oplus (r \bmod 32)$ 的 XOR swizzling，可讓 $32\times32$ tile 的列與欄讀取都不發生衝突。

    <details markdown="1"><summary>答案</summary>

    第 $r$ 列：$c \mapsto c \oplus r$ 是 $0 \dots 31$ 上的雙射，因此 32 個 word 會命中 32 個 bank。第 $c$ 欄：$r \mapsto c \oplus r$ 也是雙射，所以同樣命中 32 個不同 bank。

    </details>

4. 撰寫簡單轉置和分塊轉置，在 GPU 上以 $8192\times8192$ 計時，並與相同大小、裝置到裝置的 `cudaMemcpy` 比較。

## 實作練習

- [LeetGPU – 矩陣轉置](../leetgpu/003-matrix-transpose/)
- [LeetGPU – 矩陣複製](../leetgpu/031-matrix-copy/)
- [Tensara – 灰階](../tensara/grayscale/)（交錯布局）
- [Tensara – Max Dim](../tensara/max-dim/)（跨步歸約、跨執行緒合併存取）
