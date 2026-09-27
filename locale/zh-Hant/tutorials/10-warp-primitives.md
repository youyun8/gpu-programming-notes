# 10 – Warp 層級原語與協作群組

> **第二部分 · 平行模式** · 先備知識：[01](01-execution-model.md)、[03](03-parallel-reduction.md) ·
> 程式：[`examples/10-warp-primitives.cu`](examples/10-warp-primitives.cu) ·
> 下一章：[11 – 掃描](11-scan.md)

一個 warp 的 32 個 lane 可以直接交換暫存器、投票，以及找出值相同的
lane。這些操作都只需一條指令，不用共享記憶體，也不用
`__syncthreads()`。這些 *warp 層級原語* 是 GPU 提供的最快通訊方式，
也是歸約、掃描、壓縮、直方圖和排序的基礎。協作群組則用具型別、
可組合的 API 包裝這些功能及更多操作。

**你將學到**

- 每個 `*_sync` 原語都遵守的規則：遮罩、收斂及獨立執行緒排程；
- 四種 shuffle、投票（`__ballot_sync`、`__any_sync`、
  `__all_sync`）、`__match_any_sync` 和 `__reduce_*_sync` 系列；
- warp 模式：全歸約與廣播、使用 warp 聚合原子操作進行串流壓縮，
  以及無衝突直方圖；
- 協作群組：執行緒區塊、由 1–32 個執行緒組成的 tile、`cg::reduce`、
  `cg::inclusive_scan`，以及網格和叢集群組；
- 如何在沒有 GPU 的情況下測試 warp 層級程式碼。

## 1. 為什麼要進行 Warp 層級程式設計

### 1.1 不同範圍的通訊成本

| 範圍 | 機制 | 一次交換的成本 |
|---|---|---|
| Warp | Shuffle／投票 | 一條指令；不使用記憶體，也不需屏障 |
| 區塊 | 共享記憶體 + `__syncthreads()` | 一次儲存、一次屏障、一次載入；屏障會等待最慢的 warp |
| 網格 | 全域記憶體 + 原子操作，或第二個 kernel | 數百個週期，或啟動一次 kernel |

使用共享記憶體撰寫的區塊歸約需要 $\log_2 B$ 次屏障
（第 03 章第 2 節）；使用 shuffle 則只需要一次。模式越少跨越 warp
通訊，速度就越快。

### 1.2 Shuffle 會搬移什麼

Shuffle 會為每個 lane 搬移一個 32 位元暫存器。64 位元值（`double`、
`long long`）需要兩次 shuffle，內建函式會代為處理；結構中的每個
32 位元欄位各需要一次 shuffle（第 03 章逐欄位 shuffle 一個
`Moments` 結構）。

## 2. 原語

### 2.1 遮罩與收斂

從 Volta 開始，一個 warp 的 lane 可以位於不同指令
（獨立執行緒排程，見第 01 章第 1.4 節）。因此，每個 warp 層級原語
都會接收明確的 `mask`，指出參與的 lane：

1. 遮罩中列出的每個 lane 都必須執行同一個 `*_sync` 呼叫
   （同一條指令，而不只是同一個函式）；硬體會等待它們。
2. 不在遮罩中的 lane 不得使用該遮罩呼叫它。
3. `0xffffffff` 代表整個 warp。只有 32 個 lane 都處於活動狀態時才正確：
   區塊大小必須是 32 的倍數，且呼叫位置不能位於部分 lane 會略過的分支內。

常見錯誤是提早離開：

```cpp
if (i >= n) return;                                   // WRONG: the last warp may lose lanes
float v = __shfl_down_sync(0xffffffff, x, 1);         // ... which the full mask still names
```

讓每個 lane 都保持活動，向超出範圍的 lane 提供單位元素，並且只防護載入
與儲存（程式中的 `subtractWarpMax` 就是這麼做）。

### 2.2 Shuffle

![四種 shuffle 變體中每個 lane 的讀取來源（8 個 lane）](figures/ch10-shuffles.svg)

| 內建函式 | Lane $\ell$ 會收到哪個 lane 的值 | 常見用途 |
|---|---|---|
| `__shfl_sync(m, v, src)` | `src` | 廣播、任意排列 |
| `__shfl_up_sync(m, v, d)` | $\ell - d$（若 $\ell < d$，則為自己的值） | 包含式掃描（第 11 章） |
| `__shfl_down_sync(m, v, d)` | $\ell + d$（超過末端時為自己的值） | 歸約到 lane 0 |
| `__shfl_xor_sync(m, v, x)` | $\ell \oplus x$ | 蝶形全歸約、轉置 |

可選的最後一個引數 `width`（不超過 32 的 2 次方）會把 warp 分成數個
由 `width` 個 lane 組成的獨立區段。此時，lane 編號和「超過末端」規則
都在各區段內套用。

### 2.3 投票

| 內建函式 | 傳回值（傳給每個參與的 lane） |
|---|---|
| `__ballot_sync(m, p)` | 32 位元遮罩；若 lane $\ell$ 的述詞為真，位元 $\ell$ 就設為 1 |
| `__any_sync(m, p)` | 任一 lane 的述詞為真時傳回非零值 |
| `__all_sync(m, p)` | 每個 lane 的述詞都為真時傳回非零值 |
| `__activemask()` | 目前正在執行這條指令的 lane 遮罩（不會同步） |

把 ballot 與位元技巧結合，就能用兩條指令回答「我之前有多少個 lane
符合條件？」：

$$
\text{rank}(\ell) = \operatorname{popc}\bigl(\text{votes} \mathbin{\&} (2^{\ell} - 1)\bigr), \qquad
\text{total} = \operatorname{popc}(\text{votes})
$$

| 符號 | 意義 |
|---|---|
| votes | Ballot 遮罩 |
| $2^{\ell} - 1$ |「編號比我小的 lane」遮罩（`(1u << lane) - 1`） |
| popc | 位元數量，`__popc` |
| $\text{rank}(\ell)$ | 編號小於 $\ell$ 且投下真值的 lane 數量，也就是 lane $\ell$ 在保留元素中的位置 |

### 2.4 配對與歸約

- `__match_any_sync(m, v)` 會為每個 lane 傳回一個遮罩，包含值與自己
  相同的 lane（sm_70+）。`__match_all_sync` 會指出所有值是否相同。
- `__reduce_add_sync`、`__reduce_min_sync`、`__reduce_max_sync`、
  `__reduce_and_sync`、`__reduce_or_sync`、`__reduce_xor_sync` 可用一條
  指令對整個 warp 的 32 位元**整數**進行歸約（sm_80+）。浮點數仍需
  使用 shuffle 迴圈。
- `__syncwarp(m)` 是 `m` 中所有 lane 的屏障，也會排序它們對共享記憶體
  的存取；當一個 warp 的 lane 透過共享記憶體通訊時使用它。

## 3. 模式

### 3.1 全歸約與廣播

```cpp
__device__ float warpAllReduceMax(float v) {
    for (int lane_mask = 16; lane_mask > 0; lane_mask >>= 1) v = fmaxf(v, __shfl_xor_sync(kFullMask, v, lane_mask));
    return v;
}
```

蝶形操作完成後，每個 lane 都持有結果，因此省下
`__shfl_down_sync` 歸約在所有 lane 都要使用結果時所需的廣播
（`__shfl_sync(m, v, 0)`），例如第 13 章的 softmax。

### 3.2 串流壓縮

把符合述詞的元素密集複製到一起。每個 warp 先投票，再用**一次**
原子操作為所有保留元素預留空間，最後每個保留的 lane 計算自己的位置：

![在一個 warp 中進行串流壓縮：ballot、popc、一次原子操作](figures/ch10-ballot-compaction.svg)

```cpp
__global__ void compactPositive(const float* in, float* out, int* count, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % 32;
    const bool keep = i < n && in[i] > 0.0f;
    const unsigned votes = __ballot_sync(kFullMask, keep);   // bit l: lane l keeps its element
    int base = 0;
    if (lane == 0 && votes != 0) base = atomicAdd(count, __popc(votes));
    base = __shfl_sync(kFullMask, base, 0);                    // broadcast the warp's base
    const unsigned lanes_before = votes & ((1u << lane) - 1u);  // survivors in lanes < mine
    if (keep) out[base + __popc(lanes_before)] = in[i];
}
```

樸素版本 `out[atomicAdd(count, 1)] = in[i]` 會為每個保留元素發出一次
原子操作，而且全都存取同一個位址，因此會在 L2 序列化。如果一半元素
會保留，原子操作數是 $n/2$，而不是 $n/32$。每當許多執行緒要遞增同一個
計數器（佇列、配置器、BFS 前緣）時，都適用這種 **warp 聚合原子操作**
模式。

同一個 warp 內的輸出順序會保留；不同 warp 之間的順序取決於原子操作
的先後。需要穩定順序時，改用掃描計算位置（第 11 章）。大型輸入的
[LeetGPU – 串流壓縮](../leetgpu/072-stream-compaction/) 就應採用此方法。

### 3.3 使用 `__match_any_sync` 的直方圖

許多 lane 寫入相同 bin 時，共享記憶體原子操作會發生衝突，而偏斜資料
正會造成這種情況。值相同的 lane 可以先合併：

![`__match_any_sync` 將鍵相同的 lane 分組；每組編號最小的 lane 負責加總](figures/ch10-match-any.svg)

```cpp
const int key = i < n ? keys[i] : -1;                    // -1: "no element"
const unsigned peers = __match_any_sync(kFullMask, key);  // lanes holding the same key
const int leader = __ffs(peers) - 1;                      // lowest lane of the group
if (key >= 0 && lane == leader) atomicAdd(&local[key], __popc(peers));
```

區塊會先在共享記憶體（`local`）建立直方圖，再為每個非空 bin 使用一次
原子操作，把結果加到全域直方圖。若鍵均勻分布，多數群組只有一個 lane，
不會得到好處；若鍵分布偏斜（文字、含大片均勻區域的影像），發生衝突的
原子操作數最多可降低 32 倍。

### 3.4 Warp 掃描

使用偏移量 1、2、4、8、16 執行 `__shfl_up_sync`，可在 5 個步驟內
計算整個 warp 的前綴和。這是每種區塊與裝置掃描的基礎，第 11 章會
進一步說明。

## 4. 協作群組

### 4.1 為什麼需要群組抽象化

原始內建函式作用於「warp」和「區塊」，遮罩則要手動計算。協作群組
（`#include <cooperative_groups.h>`）把參與的執行緒集合變成物件，
可傳入函式、分割及同步：

![協作群組：一個執行緒區塊分割成多個 tile](figures/ch10-cg-tiles.svg)

| 群組 | 建立方式 | 大小 | 同步方式 |
|---|---|---|---|
| `thread_block` | `cg::this_thread_block()` | 區塊 | `block.sync()`（= `__syncthreads()`） |
| `thread_block_tile<N>` | `cg::tiled_partition<N>(block)` | $N \in \{1, 2, 4, \dots, 32\}$ | `tile.sync()`（= `__syncwarp(mask)`） |
| `coalesced_group` | `cg::coalesced_threads()` | 此處處於活動狀態的 lane | `g.sync()` |
| `grid_group` | `cg::this_grid()` | 協作啟動中的所有執行緒 | `grid.sync()` |
| `cluster_group`（sm_90） | `cg::this_cluster()` | 一個執行緒區塊叢集中的區塊 | `cluster.sync()` |

### 4.2 Tile

`thread_block_tile<N>` 將 warp 內建函式作為成員，並自動填入遮罩與寬度：
`tile.shfl_down(v, d)`、`tile.ballot(p)`、`tile.any(p)`、
`tile.thread_rank()`（0 … N−1）、`tile.meta_group_rank()`（它是區塊中的
第幾個 tile），以及 `tile.meta_group_size()`。小於一個 warp 的 tile
能讓一個 warp 處理多個彼此獨立的小問題：

```cpp
// in is rows x 16; each 16-lane tile reduces one row. Two rows per warp, no shared memory.
__global__ void rowSums16(const float* in, float* out, int rows) {
    const cg::thread_block_tile<16> tile = cg::tiled_partition<16>(cg::this_thread_block());
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 16;
    float v = row < rows ? in[static_cast<size_t>(row) * 16 + tile.thread_rank()] : 0.0f;
    v = cg::reduce(tile, v, cg::plus<float>());
    if (row < rows && tile.thread_rank() == 0) out[row] = v;
}
```

### 4.3 集合操作

`<cooperative_groups/reduce.h>` 和 `<cooperative_groups/scan.h>` 提供
`cg::reduce(tile, v, op)`、`cg::inclusive_scan(tile, v, op)` 及
`cg::exclusive_scan(tile, v, op)`，可搭配 `cg::plus`、`cg::less`、
`cg::greater`、`cg::bit_and` 等操作。它們會選擇最佳的指令序列
（在 sm_80+ 上進行整數加總時，會選擇單指令的 `redux.sync`）。
第 03 章的區塊歸約可改寫為：

```cpp
__global__ void blockSumCg(const float* in, float* block_sums, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float warp_sums[32];

    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) v += in[i];
    v = cg::reduce(warp, v, cg::plus<float>());                  // every lane gets the warp sum
    if (warp.thread_rank() == 0) warp_sums[warp.meta_group_rank()] = v;
    block.sync();
    if (warp.meta_group_rank() == 0) {
        v = warp.thread_rank() < warp.meta_group_size() ? warp_sums[warp.thread_rank()] : 0.0f;
        v = cg::reduce(warp, v, cg::plus<float>());
        if (warp.thread_rank() == 0) block_sums[blockIdx.x] = v;
    }
}
```

### 4.4 超出區塊範圍的群組

- **網格群組。** `cg::this_grid().sync()` 是跨越整個網格的屏障。只有以
  `cudaLaunchCooperativeKernel` 啟動 kernel，且區塊數不超過可同時駐留
  的數量時才合法（如此每個區塊在抵達屏障時都已執行）。它可在迭代演算法
  中取代「結束 kernel 再啟動另一個」，代價是限制網格大小。
- **叢集（sm_90）。** 執行緒區塊叢集是一組保證同時在相鄰 SM 執行的
  區塊。它們可以讀取彼此的共享記憶體（*分散式共享記憶體*，
  `cluster.map_shared_rank(ptr, rank)`），並透過 `cluster.sync()` 同步。
  Hopper GEMM 使用叢集多播 TMA 載入
  （[04.7](gemm/07-tensor-cores.md#6-hopper-wgmma-and-warp-specialization)）。
- **合併與標記群組。** `cg::coalesced_threads()` 是一起抵達目前位置的
  lane 集合，適合在分歧分支中進行 warp 聚合原子操作；
  `cg::labeled_partition(tile, label)` 依值將 lane 分組，類似
  `__match_any_sync`。

## 5. 執行與測試程式

```bash
cd tutorials/examples
nvcc -O3 -arch=sm_80 -std=c++17 10-warp-primitives.cu -o warp_primitives
./warp_primitives            # checks
./warp_primitives --bench    # also times naive vs warp-aggregated compaction
python3 ../../tools/cuemu/cuemu.py run 10-warp-primitives.cu   # the checks, on the CPU
```

cuemu 會讓遮罩中列出的每個 lane 在各次 `*_sync` 呼叫會合，因此，提早
離開的 lane 或包含不存在 lane 的遮罩，會被回報為死結或錯誤，而不會
默默讀到垃圾值。它實作了協作群組的區塊與 tile 群組；網格與叢集群組
則需要 GPU。

## 重點整理

1. `*_sync` 遮罩中列出的每個 lane 都必須執行該呼叫；讓 lane 保持活動，
   並提供單位元素，不要提早離開。
2. Shuffle 在 lane 之間搬移暫存器；`xor` 會進行全歸約，`down` 會歸約
   到 lane 0，`up` 則會進行掃描。
3. `ballot` + `popc` 可把每個 lane 的述詞轉為排名，這是壓縮和 warp
   聚合原子操作的核心。
4. `__match_any_sync` 會在相同的鍵進入原子操作前先合併它們。
5. 協作群組明確指出參與的執行緒，並提供 tile、集合操作，以及
   （透過協作啟動或叢集）超出區塊範圍的群組。

## 練習

1. 使用 `__shfl_down_sync` 和一次 `__shfl_sync` 撰寫
   `warpBroadcastMax`。與蝶形操作相比，需要多少條指令？

    <details markdown="1"><summary>答案</summary>

    要用五次 shuffle 把最大值放入 lane 0，再用一次進行廣播：共六次
    shuffle，而不是五次；`fmaxf` 的數量相同。

    </details>

2. 修改 `compactPositive`，讓同一區塊中跨 warp 的輸出順序保持穩定：
   在共享記憶體中對每個 warp 的計數進行全區塊互斥式掃描，並讓每個
   區塊只執行一次原子操作。

    <details markdown="1"><summary>提示</summary>

    每個 warp 的 lane 0 把 `__popc(votes)` 儲存到 `counts[warp]`；
    經過屏障後，warp 0 掃描這些計數；一個執行緒執行區塊的
    `atomicAdd` 並儲存基底；再經過一次屏障後，每個 lane 寫入
    `base + scanned[warp] + rank`。跨區塊的順序仍依原子操作先後而定；
    第 11 章的解耦回看也能修正這一點。

    </details>

3. 為什麼 `histogramMatch` 會把 `-1` 當作超出末端之 lane 的鍵，而不是
   讓它們略過 `__match_any_sync`？

    <details markdown="1"><summary>答案</summary>

    完整遮罩包含全部 32 個 lane，所以每一個都必須執行該呼叫。`-1`
    會讓超出範圍的 lane 彼此成組，而 `key >= 0` 測試會阻止該群組
    加上任何值。

    </details>

4. 在 `rowSums16` 中用明確的 `tile.shfl_down` 呼叫取代
   `cg::reduce`。底層 `__shfl_down_sync` 呼叫使用的 `width` 是多少？

## 實作練習

- [LeetGPU – 直方圖](../leetgpu/013-histogramming/)、[Tensara – 直方圖](../tensara/histogram/)
- [LeetGPU – 串流壓縮](../leetgpu/072-stream-compaction/)
- [LeetGPU – 計算陣列元素](../leetgpu/043-count-array-element/)（使用 ballot 和 popc 計數）
- [LeetGPU – Top-k 選擇](../leetgpu/029-top-k-selection/)
