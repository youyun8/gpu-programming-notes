# 矩陣乘法 7 – Split-K 與 Stream-K

> **第三部分 · 矩陣乘法** ·
> 程式：[`06-split-k.cu`](06-split-k.cu)、[`07-stream-k.cu`](07-stream-k.cu) · 先備知識：[矩陣乘法 2 – 向量化載入](01-vectorized-loads.md) ·
> 下一篇：[矩陣乘法 8 – Tensor Core](07-tensor-cores.md)

目前為止，每個 kernel 都為每個輸出分塊啟動一個 block。只有分塊數遠多於
SM 數時才有效率。對 $M = N = 512$、$K = 16384$ 這類「窄長」GEMM，
一台有 108–132 個 SM 的 GPU 上只有 16 個 $128\times128$ 分塊，大部分 GPU
都會閒置。即使分塊充足，最後一個 wave 也常只有一部分有工作。沿著 $K$
方向拆分工作，可同時解決這兩個問題。

**你將學到**

- 分塊量化如何浪費 SM，以及如何計算填滿效率；
- 使用 atomic 或確定性 workspace 歸約的 split-K，以及如何選擇 split factor；
- Stream-K：分配 MAC 迴圈迭代而非分塊，以及 contributor 與 owner 的角色；
- 讓跨 block 修正保持正確的記憶體順序規則，以及它為何不會 deadlock；
- 如何在模擬器上測試跨 block 通訊協定。

## 1. 量化：分塊與 SM

$$
T = \left\lceil \frac{M}{B_M} \right\rceil \left\lceil \frac{N}{B_N} \right\rceil, \qquad
\eta_{\text{tile}} = \frac{T}{P\,\lceil T / P \rceil}
$$

| 符號 | 意義 |
|---|---|
| $T$ | 輸出分塊（不拆分時即為 block 數） |
| $P$ | 同時執行的 block 數：SM 數 × 每個 SM 的 block 數 |
| $\eta_{\text{tile}}$ | 若每個分塊耗時相同，SM 時間中實際進行有效工作的比例 |

當 $P = 132$：$T = 16$ 時 $\eta = 12\%$；$T = 140$ 時要執行兩個 wave，
第二個 wave 只有 6 % 填滿，因此 $\eta = 53\%$。

![四個 SM 上的九個分塊：每個分塊一個 block，與 Stream-K 的比較](../figures/gemm-wave-quantization.svg)

## 2. Split-K

### 2.1 拆分歸約範圍

把歸約拆成 $S$ 個範圍；block $(x, y, z)$ 只會在範圍 $z$ 上計算分塊
$(y, x)$：

$$
C = \sum_{z=0}^{S-1} A_{:,\,\mathcal{K}_z}\,B_{\mathcal{K}_z,\,:}, \qquad
\mathcal{K}_z = \bigl[z\,c,\ \min(K, (z+1)\,c)\bigr), \qquad
c = B_K\left\lceil \frac{\lceil K / S \rceil}{B_K} \right\rceil
$$

| 符號 | 意義 |
|---|---|
| $S$ | Split factor（`gridDim.z`） |
| $\mathcal{K}_z$ | Split $z$ 負責的 $K$ 範圍 |
| $c$ | Chunk 長度，向上取整為完整切片，讓 `float4` 載入保持對齊 |

![Split-K：把 S 個部分分塊加總成 C](../figures/gemm-split-k.svg)

### 2.2 合併部分分塊

向量化載入的主迴圈只需改成從 `k_begin` 執行到 `k_end`。部分分塊可用兩種方式
合併（`./split_k --atomic` 選擇第一種）：

| 模式 | 做法 | 額外流量 | 確定性 |
|---|---|---|---|
| Atomic | `cudaMemset(C, 0)`，接著每個 split 對每個元素執行 `atomicAdd` | $S\cdot MN\cdot 4$ 位元組的 atomic（在 L2 中處理） | 否：FP32 加法順序會變動 |
| Workspace | Split $z$ 把部分結果寫入 `workspace[z]`；第二個 kernel 加總 $z = 0 \dots S-1$ | $2\,S\cdot MN\cdot 4$ 位元組 | 是 |

### 2.3 選擇切分數

程式選擇 $S$ 時，會讓 $T\cdot S \approx 2P$，同時讓每個 split 至少保留
4 個 $K$ 切片：

```cpp
int chooseSplits(int m, int n, int k, int num_sms) {
    const int tiles = gemm::ceilDiv(m, kBlockM) * gemm::ceilDiv(n, kBlockN);
    const int by_fill = gemm::ceilDiv(2 * num_sms, tiles);
    const int by_depth = std::max<int>(1, k / (4 * kBlockK));
    return std::max<int>(1, std::min<int>(by_fill, by_depth));
}
```

當歸約流量相對 GEMM 本身很小時，Split-K 才有優勢；粗略而言需滿足
$K \gg S\cdot$（數百）。它是 LLM 推論 decode 階段 GEMM（$M$ = batch，
且很小）的標準選擇，也是 AITER `splitk` kernel 與 TensileLite
`GlobalSplitU` 的做法
（[第 06 章第 5 節](../06-aiter-asm-gemm.md#5-epilogue-split-k-and-bf16-rounding)）。

## 3. Stream-K

### 3.1 分配迭代，而非分塊

Split-K 仍有量化問題：$P$ 個 SM 執行 $T\cdot S$ 個相同大小的 block。
Stream-K 不再分配分塊，而是分配 *MAC 迴圈迭代*，徹底消除量化：

$$
L = T\left\lceil \frac{K}{B_K} \right\rceil, \qquad
\text{block } g \text{ runs iterations } \left[\left\lfloor \frac{gL}{G} \right\rfloor,\ \left\lfloor \frac{(g+1)L}{G} \right\rfloor\right), \qquad
\eta_{\text{SK}} = \frac{L}{G\,\lceil L/G \rceil} \approx 1
$$

| 符號 | 意義 |
|---|---|
| $L$ | MAC 迴圈總迭代數（一個分塊的每個 $B_K$ 切片各一次） |
| $G$ | Persistent block 數，正好等於 GPU 可同時容納的數量 |
| $\eta_{\text{SK}}$ | 填滿效率；不同 block 的迭代數最多相差一 |

一個 block 的範圍會跨越分塊邊界，因此同一分塊可能由多個 block 計算：

![Stream-K 範圍：contributor 發布部分分塊，owner 負責加總](../figures/gemm-stream-k-ranges.svg)

### 3.2 貢獻者與擁有者

對其範圍中的每個分塊區段，block $g$ 會採取以下三種動作之一：

1. 範圍包含**完整分塊**：像一般 GEMM 一樣計算並儲存。
2. **貢獻者（contributor）**（範圍結束於分塊內）：把部分分塊存入 `workspace[g]`，
   執行 `__threadfence()`，再設定 `flags[g]`。
3. **擁有者（owner）**（範圍結束於分塊末尾，但分塊開頭在較早 block 的範圍內）：
   等待那些較早 block 的 flag，依固定順序加入其部分結果，再儲存。

```cpp
if (seg_end < tile_end) {                        // contributor
    for (...) slot[(8 * i + j) * kThreads + tid] = acc[i][j];
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicExch(&flags[g], 1);
} else {
    if (it > tile_begin) {                       // owner of a shared tile
        for (int p = g - 1; p >= 0 && rangeStart(p + 1, total, gridDim.x) > tile_begin; --p) {
            if (tid == 0) {
                while (atomicAdd(&flags[p], 0) == 0) {}
                __threadfence();
            }
            __syncthreads();
            const volatile float* slot = workspace + p * kTileElems;   // not from a stale L1 line
            for (...) acc[i][j] += slot[(8 * i + j) * kThreads + tid];
        }
    }
    // store the finished tile
}
```

### 3.3 為何這套協定是正確的

以下五個細節保證了正確性：

- **每個範圍只有最後一個區段可能成為 contributor**，所以每個 block
  只需一個 workspace slot（$G\cdot B_MB_N\cdot4$ 位元組：$G = 132$
  時約 8 MiB）。
- **Owner 只等待編號較小的 block，而且所有 $G$ 個 block 會同時駐留。**
  將 $G$ 設為 SM 數 ×
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 可保證後者。否則（例如
  $G$ 大於可容納量），owner 可能永遠等待一個仍在等候 SM 的 contributor。
- **記憶體順序。** Contributor 的 `__threadfence()` 會讓 workspace 儲存
  排在 flag 之前；owner 觀察到 flag 後執行 fence，並以 `volatile` 載入
  workspace（不會命中 SM 非一致性 L1 中的過期 cache line），使部分結果
  可見。
- **確定性。** 部分結果會依固定順序加入，因此可逐位元重現，不像 atomic。
- **每次啟動都會重設 flag**（`cudaMemsetAsync`）。

### 3.4 混合排程

論文也說明了*混合*排程：先執行完整 wave 的完整分塊（沒有修正成本），
只有剩餘部分使用 Stream-K。hipBLASLt 也附帶自己的 Stream-K kernel 函式庫
（[第 07 章](../07-hipblaslt-tensilelite.md#15-work-decomposition-gsu-and-stream-k)）。

## 4. 測試跨 Block 通訊協定

模擬器會依索引順序一次執行一個 block，這是有效排程之一：每個 contributor
都會在 owner 開始前完成。因此 `--test` 能針對許多 grid 大小，檢查拆分方式
（範圍、owner、workspace 索引）的算術：

```bash
for g in 1 3 7 13; do python3 tools/cuemu/cuemu.py run tutorials/gemm/07-stream-k.cu -- --blocks=$g --test; done
```

它無法找出並行 block 間的記憶體順序錯誤；在 GPU 上應搭配
`compute-sanitizer --tool racecheck`，並以不同 grid 大小重複執行。

## 重點整理

1. 分塊很少時，每個分塊一個 block 會讓多數 SM 閒置；分塊很多時，最後一個 wave 也常未填滿。
2. Split-K 以 $S$ 倍增加平行度，代價是合併 $S$ 個部分分塊（atomic：快速但不確定；workspace：具確定性）。
3. Stream-K 讓每個 persistent block 分到相同數量的迭代；共用分塊透過 workspace 與 flag 修正。
4. 跨 block 通訊需要 writer 執行 fence → flag，reader 執行 flag → fence，使用繞過 L1 的載入，且所有 block 必須同時駐留。

## 練習

1. 在自己的 GPU 上，對 $M = N = 1024$、$K = 8192$ 計時向量化載入、split-K
   （兩種模式）與 Stream-K。各自在什麼情況勝出？
2. 實作混合排程：前 $\lfloor T / G \rfloor\cdot G$ 個分塊各用一個 block，
   其餘部分使用 Stream-K。
3. 不使用第二個 kernel，讓 atomic split-K 具有確定性：採用第 03 章第 5 節
   的「最後抵達的 block 負責歸約」做法。

    <details markdown="1"><summary>提示</summary>

    為每個分塊建立一個 counter。每個 split 把部分分塊寫入
    `workspace[z]`，執行 fence，再遞增該分塊的 counter；看到 $S - 1$ 的
    split 依 $z = 0, \dots, S-1$ 順序加總 $S$ 個部分結果，寫入 $C$，並
    重設 counter。AITER 的 semaphore 就是這樣運作（第 06 章第 5 節）。

    </details>
