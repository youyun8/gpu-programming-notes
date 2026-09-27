# 11 – 掃描（前綴和）

> **第二部分 · 平行模式** · 先備知識：[03](03-parallel-reduction.md)、[10](10-warp-primitives.md) ·
> 程式：[`examples/11-scan.cu`](examples/11-scan.cu) ·
> 下一章：[12 – 卷積與樣板](12-convolution-stencils.md)

掃描會計算序列的每個前綴：$y_i = x_0 + x_1 + \dots + x_i$。
它看似天生只能循序執行（每個輸出都依賴前一個），卻幾乎能像歸約一樣
有效平行化。它也是 GPU 演算法背後的主力：串流壓縮、基數排序、直方圖
偏移量、稀疏矩陣列指標，以及現代序列網路之狀態空間模型所用的線性
遞迴，最終都可化為掃描。

**你將學到**

- 包含式與互斥式掃描，以及它們的用途；
- Kogge-Stone 與工作高效掃描在工作量和深度之間的取捨，以及 GPU
  採用的折衷方式（暫存器內循序執行、執行緒間平行執行）；
- 使用 shuffle 的 warp 掃描、區塊掃描，以及包含 2048 個項目的 tile
  掃描；
- 兩種全裝置策略：先歸約再掃描（三個 kernel），以及以解耦回看進行
  單次走訪；後者的速度與複製相當；
- 使用其他結合性運算子的掃描：分段掃描與線性遞迴。

## 1. 定義與用途

### 1.1 包含式與互斥式

$$
\text{inclusive: } y_i = \bigoplus_{j=0}^{i} x_j, \qquad
\text{exclusive: } z_i = \bigoplus_{j=0}^{i-1} x_j = y_i \ominus x_i, \quad z_0 = e
$$

| 符號 | 意義 |
|---|---|
| $x_i$ | 輸入元素 $i$ |
| $\oplus$ | 結合性運算子（若未另行說明則為加法），單位元素為 $e$ |
| $y_i, z_i$ | 包含式與互斥式前綴 |
| $\ominus$ | $\oplus$ 的反運算（若存在；對加總而言是減法） |

### 1.2 掃描的用途

| 應用 | 掃描對象 | 結果用途 |
|---|---|---|
| 串流壓縮 | 保留旗標（0/1） | 互斥式掃描 = 每個保留元素的輸出索引 |
| 基數排序 | 各數位的計數 | 互斥式掃描 = 每個 bucket 的起始位置 |
| CSR 稀疏矩陣 | 每列的非零元素數 | 互斥式掃描 = 列指標陣列 |
| 累加和（cumsum） | 資料 | 答案 |
| 線性遞迴、SSM | 仿射映射 $(a_t, b_t)$ | 每個狀態 $h_t$（第 6.3 節） |

## 2. 工作量與深度

循序掃描會以長度為 $n - 1$ 的相依鏈執行 $n - 1$ 次加法。
平行掃描以額外工作量換取較小深度：

$$
\text{Kogge-Stone: } W = n\log_2 n - n + 1,\ D = \log_2 n, \qquad
\text{Brent-Kung: } W \approx 2n,\ D \approx 2\log_2 n
$$

| 符號 | 意義 |
|---|---|
| $W$ | 執行的加法次數 |
| $D$ | 最長的相依加法鏈 |
| $n$ | 元素數量 |

Kogge-Stone 簡單且深度最小，但對數百萬個元素而言，額外的
$\log_2 n$ 工作量會耗費大量頻寬和指令。GPU 使用符合 Brent 界限
（第 03 章第 1 節）的混合方式：**每個執行緒循序掃描少量項目**
（工作高效、不需同步），再以最小深度的方法平行掃描較小的每執行緒
總和陣列。對含 32 個 lane 的 warp 而言，Kogge-Stone 額外成本很小，
而 shuffle 讓它十分便宜。

## 3. Warp 掃描

![對 8 個值進行 Kogge-Stone 包含式掃描：log2(8) = 3 個步驟](figures/ch11-kogge-stone.svg)

```cpp
// After step d, lane l holds x[l-2d+1 .. l]: 5 steps for 32 lanes (Kogge-Stone / Hillis-Steele).
__device__ int warpInclusiveScan(int v) {
    const int lane = threadIdx.x % 32;
    for (int d = 1; d < 32; d <<= 1) {
        const int u = __shfl_up_sync(kFullMask, v, d);
        if (lane >= d) v += u;
    }
    return v;
}
```

`__shfl_up_sync` 會把 lane $\ell < d$ 自己的值傳回給它，因此
`if (lane >= d)` 防護可避免重複計算。互斥式掃描可用
`inclusive - x` 計算（或向上 shuffle 一格，並把 lane 0 設為
單位元素）。

## 4. 區塊與 Tile 掃描

### 4.1 區塊掃描

區塊掃描把同樣的概念提升一層：每個 warp 掃描自己的 32 個值，每個
warp 的 lane 31 發布 warp 總和，warp 0 掃描這些總和，然後每個執行緒
加上位於自己之前之 warp 的總和：

```cpp
__device__ int blockInclusiveScan(int v, int* total) {
    __shared__ int warp_totals[32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int num_warps = blockDim.x / 32;
    v = warpInclusiveScan(v);
    if (lane == 31) warp_totals[warp] = v;          // each warp's total
    __syncthreads();
    if (warp == 0) {                                // warp 0 scans the (at most 32) totals
        const int t = lane < num_warps ? warp_totals[lane] : 0;
        warp_totals[lane] = warpInclusiveScan(t);
    }
    __syncthreads();
    if (warp > 0) v += warp_totals[warp - 1];       // add the totals of the warps before
    *total = warp_totals[num_warps - 1];
    __syncthreads();                                // warp_totals may be reused by the caller
    return v;
}
```

不論區塊大小為何，每次區塊掃描都需要兩次屏障。

### 4.2 每個執行緒處理多個項目

若每個執行緒只處理一個項目，大部分時間都會耗在 shuffle 和屏障上。
因此每個執行緒改為擁有連續 8 個項目，讓含 256 個執行緒的區塊掃描
一個含 2048 個項目的 **tile**：

![掃描一個含 2048 個項目的 tile：暫存器、warp、區塊](figures/ch11-hierarchy.svg)

```cpp
for (int i = threadIdx.x; i < kTile; i += blockDim.x)   // coalesced: consecutive threads, consecutive items
    tile[padded(i)] = offset + i < n ? in[offset + i] : 0;
__syncthreads();
int items[kItems];
int running = 0;
for (int j = 0; j < kItems; ++j) {                       // sequential inclusive scan in registers
    running += tile[padded(threadIdx.x * kItems + j)];
    items[j] = running;
}
int total = 0;
const int thread_inclusive = blockInclusiveScan(running, &total);
const int carry = thread_inclusive - running;            // exclusive prefix of this thread
for (int j = 0; j < kItems; ++j) tile[padded(threadIdx.x * kItems + j)] = items[j] + carry;
```

#### 合併存取與 bank 衝突

這裡有兩項細節：

- **合併存取與擁有權。** 從全域記憶體讀取時，連續執行緒會讀取連續
  項目（合併存取），但之後每個執行緒需要 8 個*連續*項目。共享記憶體
  可在兩種配置之間轉換。
- **記憶體庫衝突。** 執行緒 $t$ 讀取項目 $8t + j$ 時，會以步幅 8
  存取，造成 8 路衝突。每 32 個項目加入一個填補字
  （`padded(i) = i + i / 32`），即可把 32 個 lane 分散到 32 個
  記憶體庫。

$$
\operatorname{bank}\bigl(8t + j + \lfloor (8t + j)/32 \rfloor\bigr) = \bigl(8\,(t \bmod 4) + j + \lfloor t/4 \rfloor\bigr) \bmod 32
$$

| 符號 | 意義 |
|---|---|
| $t$ | Lane 索引，0–31 |
| $j$ | 執行緒內的項目索引，0–7（對一條指令而言固定） |

固定 $j$ 後，當 $t$ 走遍整個 warp 時，
$8(t \bmod 4) + \lfloor t/4 \rfloor$ 會取得 32 個不同的值，因此沒有衝突。

## 5. 掃描完整陣列

各個 tile 彼此獨立，只差一個數字：每個 tile 都需要前面所有 tile 的
總和，也就是它的**輸入進位**。

### 5.1 先歸約再掃描

使用三個 kernel：

1. `tileSums`：每個區塊加總自己的 tile。
2. `scanSums`：一個區塊把各 tile 的總和轉為互斥式前綴。
3. `scanTiles`：每個區塊再次掃描自己的 tile，並加上前綴。

```cpp
void scanReduceThenScan(const int* in, int* out, int* sums, int n) {
    const int tiles = ex::ceilDiv(n, kTile);
    tileSums<<<tiles, kThreads>>>(in, sums, n);
    scanSums<<<1, 1024>>>(sums, tiles);
    scanTiles<<<tiles, kThreads>>>(in, out, sums, n);
}
```

它會讀取輸入兩次：

$$
Q_{\text{RTS}} = 4n\,(2 + 1) = 12n\ \text{bytes}, \qquad
Q_{\min} = 4n\,(1 + 1) = 8n\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $Q_{\text{RTS}}$ | 先歸約再掃描的 DRAM 流量（int32）：讀取兩次、寫入一次 |
| $Q_{\min}$ | 必要流量：讀取一次、寫入一次，與複製相同 |

因此，速度最高只能達到複製的 $8/12 = 67\%$（若兩次走訪之間輸入沒有
留在 L2，速度還會更低）。

### 5.2 使用解耦回看的單次走訪

Merrill 與 Garland 的單次走訪掃描（CUB `DeviceScan` 背後的演算法）
只讀取輸入一次。每個 tile 一有能力就發布一個**狀態**，並回看前序 tile
的狀態，而不等待全域走訪：

![解耦回看：tile 5 讀取前序 tile 發布的狀態](figures/ch11-lookback.svg)

| 狀態 | 意義 |
|---|---|
| X（未就緒） | 尚未發布任何內容 |
| A（總和） | 僅此 tile 的總和 |
| P（前綴） | 包含式前綴：截至並包含此 tile 的所有 tile 總和 |

#### 協定

一個 tile 會：(1) 在本地掃描自己；(2) 以自己的總和發布 A；
(3) 往回走訪前序 tile，把它們的 A 值加總，並在第一個 P 停止；
(4) 發布 P；(5) 使用找到的互斥式前綴寫入輸出：

```cpp
if (threadIdx.x == 0) {
    volatile unsigned long long* vstatus = status;
    if (t == 0) {
        atomicExch(&status[0], packStatus(kPrefix, total));
        s_exclusive = 0;
    } else {
        atomicExch(&status[t], packStatus(kAggregate, total));   // let successors start early
        int prefix = 0;
        for (int p = t - 1; p >= 0; --p) {                       // look back
            unsigned long long s;
            do {
                s = vstatus[p];
            } while ((s >> 32) == kNotReady);
            prefix += static_cast<int>(static_cast<unsigned>(s));
            if ((s >> 32) == kPrefix) break;                     // everything before p is included
        }
        atomicExch(&status[t], packStatus(kPrefix, prefix + total));
        s_exclusive = prefix;
    }
}
```

#### 為何正確且不會卡住

以下三項性質保證了這一點：

- **旗標和值會一起傳遞。** 兩者位於同一個 64 位元字中，並由單一原子
  操作寫入，因此讀取者不會只看到旗標而沒有值（兩者之間不需 fence）。
- **Tile 依啟動順序編號。** 區塊會從原子計數器取得 tile 編號，而不是
  使用 `blockIdx.x`。硬體不保證依 `blockIdx` 順序啟動區塊，但使用
  計數器後，一個 tile 的每個前序 tile 都已經啟動，因此最終一定會發布；
  自旋等待必定會結束。
- **回看距離很短。** Tile 通常在幾步內就會找到 P，因為前序 tile 會
  提早發布 A，並在不久後發布 P。CUB 讓一整個 warp 一次回看 32 個前序
  tile（對其狀態進行 warp 歸約）；此處的單執行緒迴圈使用相同協定，
  只是為了清楚而如此撰寫。

#### 成本

流量為 $8n$ 位元組，與複製相同；實務上，良好的單次走訪掃描能達到
`cudaMemcpy` 頻寬的 85–95%。程式的 `--bench` 模式會比較三種方法。

## 6. 其他運算子

### 6.1 任意么半群

以上內容不曾使用 $\oplus$ 是加法這項性質，只要求它具結合性和單位
元素：第 03 章的么半群（最大值、最小值、乘積、softmax 統計）都能以
相同程式碼掃描。

### 6.2 分段掃描

分段掃描會在區段邊界重新開始（每個區段第一個元素的旗標
$f_i = 1$）。它是對數對進行的一般掃描：

$$
(f_a, x_a) \oplus (f_b, x_b) = \bigl(f_a \lor f_b,\ f_b\ ?\ x_b : x_a + x_b\bigr)
$$

| 符號 | 意義 |
|---|---|
| $f$ | 區段起始旗標（1 = 新區段從此處開始） |
| $x$ | 累積值 |

此運算子具結合性，因此只要把數對當成值型別，warp、區塊與回看掃描
都不必修改。

### 6.3 線性遞迴

一階遞迴 $h_t = a_t h_{t-1} + b_t$ 看似循序，但映射
$h \mapsto a h + b$ 能以結合方式組合：

$$
(a_1, b_1) \oplus (a_2, b_2) = (a_1 a_2,\ a_2 b_1 + b_2), \qquad
h_t = A_t h_{-1} + B_t, \quad (A_t, B_t) = \bigoplus_{s=0}^{t} (a_s, b_s)
$$

| 符號 | 意義 |
|---|---|
| $a_t, b_t$ | 步驟 $t$ 的係數 |
| $(A_t, B_t)$ | 從初始狀態到 $h_t$ 的組合映射 |
| $h_{-1}$ | 初始狀態 |

選擇性狀態空間模型（Mamba）和指數移動平均就是透過這種方式在序列上
平行執行
（[LeetGPU – 線性遞迴](../leetgpu/082-linear-recurrence/)、
[LeetGPU – SSM 選擇性掃描](../leetgpu/094-ssm-selective-scan/)）。

## 7. 函式庫

- **CUB**：`cub::WarpScan`、`cub::BlockScan` 和 `cub::DeviceScan`
  （使用解耦回看的單次走訪）；參考實作。
- **Thrust**：`thrust::inclusive_scan`、
  `thrust::exclusive_scan_by_key`（分段）。
- **PyTorch**：`torch.cumsum`、`torch.cumprod`。

當掃描與其他操作（壓縮、排序走訪、SSM）融合時，可自行實作，讓資料
只需讀取一次。

## 重點整理

1. 掃描和歸約一樣能夠平行化：在執行緒內循序執行、在 warp 間使用
   Kogge-Stone、在區塊間掃描 warp 總和。
2. 透過共享記憶體暫存 tile，把合併載入與每執行緒連續項目結合；使用
   填補避免步幅衝突。
3. 先歸約再掃描很簡單，但會讀取輸入兩次；解耦回看只讀取一次，並可
   達到複製頻寬。
4. 跨區塊協定需要單一字狀態（旗標 + 值）、依啟動順序取得的 tile
   編號，以及可避開過時快取的載入（`volatile` 或原子操作）。
5. 任何結合性運算子都能進行掃描：分段掃描與線性遞迴都是對數對掃描。

## 練習

1. 不使用減法，把 `warpInclusiveScan` 改成互斥式掃描。

    <details markdown="1"><summary>答案</summary>

    完成包含式掃描後，執行
    `int e = __shfl_up_sync(kFullMask, v, 1);`，並在 lane 0 將
    `e = 0`。

    </details>

2. 對 256 個執行緒而言，`blockInclusiveScan` 會執行多少次加法？
   與對一個 tile 的全部 2048 個項目進行 Kogge-Stone 掃描相比如何？

    <details markdown="1"><summary>答案</summary>

    每個 warp 會執行
    $\sum_{d} (32 - d) = 31 + 30 + 28 + 24 + 16 = 129$ 次；8 個 warp
    加上 warp 0 的第二次掃描，約為 $9 \times 129 \approx 1161$ 次，
    再加上 224 次 warp 進位加法。Tile 掃描會讓每個執行緒循序執行
    7 次加法（共 1792 次），再讓每個執行緒執行 8 次進位加法。
    對 2048 個項目進行 Kogge-Stone 掃描則會執行
    $2048\cdot11 - 2047 \approx 20\,500$ 次。

    </details>

3. 撰寫分段*最大值*掃描的數對運算子，並以三組範例數對檢查其結合性。

4. 使用掃描程式實作穩定的串流壓縮：掃描保留旗標，再把每個保留元素
   寫入其互斥式前綴所指的位置。

    <details markdown="1"><summary>提示</summary>

    將它融合進 `scanTileInShared`：掃描旗標而不是值，把值保留在暫存器
    中，並在 `storeTile` 中只於旗標為 1 時把 `x` 寫入
    `out[prefix]`。保留元素總數就是最後一個元素的包含式前綴。

    </details>

## 實作練習

- [LeetGPU – 前綴和](../leetgpu/016-prefix-sum/)、[Tensara – Cumsum](../tensara/cumsum/)、
  [Tensara – 一維累加和](../tensara/running-sum-1d/)
- [LeetGPU – 分段前綴和](../leetgpu/070-segmented-prefix-sum/)
- [LeetGPU – 串流壓縮](../leetgpu/072-stream-compaction/)、[LeetGPU – 基數排序](../leetgpu/036-radix-sort/)
- [Tensara – Cumprod](../tensara/cumprod/)、[LeetGPU – 線性遞迴](../leetgpu/082-linear-recurrence/)、
  [LeetGPU – GAE 反向掃描](../leetgpu/110-gae-reverse-scan/)
