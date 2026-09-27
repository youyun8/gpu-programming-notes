# 04.5 – 利用 Swizzle 分塊順序提高 L2 重用

> **第三部分 · 矩陣乘法 · 04.x GEMM 深入解析** ·
> 程式：[`05-tile-swizzle.cu`](05-tile-swizzle.cu) · 延續：[04.4](04-warp-tiling.md) ·
> 下一篇：[04.6 – Split-K 與 Stream-K](06-split-k-stream-k.md)

每個 block 都會讀取完整的一個 $A$ 橫向 panel（$128\times K$）及一個 $B$
縱向 panel（$K\times128$）。整個網格中的每個 panel 都會由多個 block
讀取：$A$ 的 panel $i$ 會由分塊列 $i$ 中所有 $\lceil N/128\rceil$ 個 block
讀取。這些重複讀取來自 L2 還是 DRAM，取決於哪些 block *同時*執行；
而這由分配分塊索引的順序決定。

**你將學到**

- 為何輸出分塊的啟動順序會決定 L2 重用率；
- 從啟動索引到分塊的 grouped（swizzle）映射，以及它為何是雙射；
- 如何估算一個 wave 的 L2 footprint，以及 group 為何必須夠高；
- 這與 TensileLite 的 `WorkGroupMapping` 及 MI300 每個 XCD 各自的 L2
  有何關係。

## 1. 啟動順序就是分塊順序

Block 大致會依線性索引順序分派（`blockIdx.x` 優先）。使用一般 2-D grid
時，block $p$ 會計算分塊 $(p / t_N,\ p \bmod t_N)$：第一個 wave 會掃過
完整的分塊列。

![前 12 個 block 的 row-major 與 grouped 分塊順序](../figures/gemm-tile-order.svg)

若一個由 $W$ 個同時執行 block 組成的 wave 涵蓋 $h\times w$ 的分塊矩形，
所需資料為

$$
F(h, w) = (h + w)\cdot B\,K \cdot 4\ \text{bytes}, \qquad hw \approx W
$$

| 符號 | 意義 |
|---|---|
| $W$ | 同時駐留的 block 數（SM 數 × 每個 SM 的 block 數） |
| $h, w$ | Wave 涵蓋的分塊列數與欄數 |
| $B$ | 分塊大小（128）；一個 panel 有 $B\cdot K$ 個 float |
| $F$ | Footprint：wave 讀取的不同 panel 位元組數 |

在 $hw$ 固定時，正方形的 $h + w$ 最小。Row-major 順序的
$h \approx W / t_N$（矮而寬）；grouped 順序則有 $h = g$（group 高度）。

## 2. Grouped 映射

```cpp
__host__ __device__ inline void groupedTile(int pid, int tiles_m, int tiles_n, int group_m,
                                            int& tile_m, int& tile_n) {
    const int per_group = group_m * tiles_n;          // tiles in one group of tile rows
    const int first_m = (pid / per_group) * group_m;
    const int rows = tiles_m - first_m < group_m ? tiles_m - first_m : group_m;
    const int in_group = pid % per_group;
    tile_m = first_m + in_group % rows;               // walk down the group first...
    tile_n = in_group / rows;                         // ...then one tile column right
}
```

以 $s$ 表示啟動索引，公式為：

$$
m_0 = g\left\lfloor \frac{s}{g\,t_N} \right\rfloor, \qquad
h = \min(g,\ t_M - m_0), \qquad
\text{tile}_m = m_0 + (s \bmod g\,t_N) \bmod h, \qquad
\text{tile}_n = \left\lfloor \frac{s \bmod g\,t_N}{h} \right\rfloor
$$

| 符號 | 意義 |
|---|---|
| $s$ | 啟動索引（1-D grid 的 `blockIdx.x`） |
| $g$ | 以分塊列為單位的 group 高度（`group_m`） |
| $t_M, t_N$ | $M$、$N$ 方向的分塊數 |
| $m_0$ | 包含 $s$ 的 group 中第一個分塊列 |
| $h$ | 該 group 的列數（最後一個 group 可能較短） |

它是 $[0, t_Mt_N)$ 上的雙射，因此 kernel 主體不變；只有 `row0` 與 `col0`
改由 `groupedTile(blockIdx.x, ...)` 取得，而非 `blockIdx.y` 與
`blockIdx.x`。這與 Triton matmul 教學的「grouped ordering」及 CUTLASS
threadblock swizzle 是相同映射，也是 TensileLite `WorkGroupMapping`
背後的概念（[第 07 章](../07-hipblaslt-tensilelite.md#16-cache-aware-tile-order-wgm-wgmxcc-staggeru)）。

## 3. 可節省多少

`./tile_swizzle --footprint` 會計算 $4096^3$ 問題中前 132 個分塊
（132-SM H100 上每個 SM 一個 block）所存取的 panel；每個 panel 為 2 MiB：

| `group_m` | A panel 數 | B panel 數 | Footprint |
|---|---|---|---|
| 0（row-major） | 5 | 32 | 74 MiB |
| 4 | 8 | 32 | 80 MiB |
| 8 | 8 | 17 | 50 MiB |
| 16 | 16 | 9 | 50 MiB |

Row-major 順序中，一個 wave 需要超過 H100 的 50 MB L2；group 大小為 8 或
16 時則能放入。此表有兩項重點：

1. **Group 必須比 wave 更高。** 當 `group_m = 4` 時，132 個分塊橫跨
   4 個完整的 $4\times32$ 分塊 group，因此仍需要每個 $B$ panel。
   Group 必須夠大，讓一個 wave 留在一至兩個 group 內。
2. **效益取決於形狀與 GPU。** 當整個 $A$ 與 $B$ 都能放進 L2（小型問題）
   時，順序幾乎沒有影響；$K$ 很大時，panel 隨之增大，順序就更重要。
   函式庫會依形狀選擇 `group_m`；AMD MI300 還會跨 XCD 重新映射，因為
   各 XCD 有獨立 L2（[第 05 章](../05-amd-cdna3-mfma.md)，以及第 07 章的
   `WGMXCC`）。

大型 GEMM 中，row-major 順序會讓 kernel 從 DRAM 重新擷取 panel，因此
效果最明顯；請在自己的 GPU 上測量，不要直接採信其他環境的數字。

## 4. 常見陷阱

- **1-D grid 也有自己的限制。** `gridDim.x` 最大可達 $2^{31} - 1$，
  因此 1-D grid 通常沒問題；2-D grid 的 `gridDim.y` 則限制為 65535。
- **程式設計模型不保證分派順序。** 實務上會依序分派，對快取最佳化而言
  已足夠；正確性絕不能依賴它（可比較 [04.6](06-split-k-stream-k.md)
  的 Stream-K 修正）。

## 重點整理

1. 同時執行的 block 應涵蓋 $C$ 中緊密的矩形，使它們能在 L2 共用 $A$ 與 $B$ panel。
2. Grouped ordering 只改變 kernel 的兩行：`row0` 與 `col0` 改由重新映射的啟動索引取得。
3. 依 wave 大小與問題形狀選擇 group 高度；太矮的 group 沒有幫助。

## 練習

1. 擴充 `--footprint`，計算每個 wave 中有多少 panel 已由前一個 wave
   使用過（作為粗略的 L2 hit 估計）。
2. 使用 `ncu --metrics lts__t_sector_hit_rate.pct`，比較 04.4 與此程式在
   $8192\times8192\times8192$ 問題上的 L2 hit rate。
