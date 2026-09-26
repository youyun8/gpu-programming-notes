# 04.3 – 非同步複製：`cp.async` Pipeline 與 TMA

> **第三部分 · 矩陣乘法 · 04.x GEMM 深入解析** ·
> 程式：[`03-cp-async.cu`](03-cp-async.cu) · 延續：[04.2](02-double-buffering.md) ·
> 下一篇：[04.4 – Warp 分塊](04-warp-tiling.md)

經由暫存器進行雙緩衝有兩項成本：傳輸中的切片會占用暫存器，而且每個元素
仍需要兩條指令（一次全域載入、一次共享記憶體儲存）。Ampere（sm_80）
加入了 `cp.async`，可將資料從全域記憶體直接複製到共享記憶體，繞過
暫存器檔案且不阻塞 thread。Hopper（sm_90）則加入 Tensor Memory
Accelerator（TMA），能用一條指令複製完整分塊。

![從全域記憶體前往共享記憶體的三種方式](../figures/gemm-copy-paths.svg)

**你將學到**

- `cp.async` 模型：每個 thread 的非同步複製、commit group 與 wait；
- 如何建立多階段 pipeline 並推導 wait 數量；
- 為何空 group 也必須 commit，以及模擬器如何找出錯誤計數；
- 需要多少 stage，以及它們占用多少共享記憶體；
- Hopper 的 TMA 與 mbarrier 如何改變整體做法。

## 1. `cp.async` 模型

每個 thread 會發出 4、8 或 16 位元組的複製。複製會組成 group，而 thread
可等待到最多只剩最新的 $n$ 個 group 尚未完成：

| `<cuda_pipeline.h>` | PTX | 意義 |
|---|---|---|
| `__pipeline_memcpy_async(dst, src, 16, zfill)` | `cp.async.{ca,cg}.shared.global [dst], [src], 16, src_size` | 開始複製；最後 `zfill` 個位元組填零 |
| `__pipeline_commit()` | `cp.async.commit_group` | 結束目前的 group |
| `__pipeline_wait_prior(n)` | `cp.async.wait_group n` | 等待到最多剩下 $n$ 個 group |

設計由三項特性決定：

1. **Wait 是每個 thread 各自處理。** `wait_group` 只涵蓋*目前 thread*
   發出的複製。同一分塊中其他 thread 的複製，必須在 wait 後再用
   `__syncthreads()` 等待。
2. **Group 會依序完成**，所以「最多 $n$ 個尚未完成」等於「除了最新的
   $n$ 個以外都已寫入」。這與[第 06 章](../06-aiter-asm-gemm.md)的 AMD
   `s_waitcnt vmcnt(n)` 完全相同。
3. **越界元素不需額外成本。** 當 `src_size = 0`（此處 `zfill = 16`）時，
   不會讀取來源，目的地會填零，因此邊界分塊不需要另一條程式路徑。

## 2. 多階段 Pipeline

使用 $P$ 個 stage（緩衝區）時，會有 $P - 1$ 個切片正在傳輸，另有一個
切片正在運算：

![3 階段 cp.async pipeline](../figures/gemm-pipeline.svg)

```cpp
// Prologue: slices 0 .. kStages-2, one group each (committed even when empty).
for (int s = 0; s < kStages - 1; ++s) {
    if (s < num_slices) issueSlice<kVec>(a_s[s], b_s[s], ..., s * kBlockK);
    __pipeline_commit();
}
for (int s = 0; s < num_slices; ++s) {
    __pipeline_wait_prior(kStages - 2);   // my copies of slice s have landed
    __syncthreads();                      // everyone's have; stage (s-1) % kStages is free
    const int next = s + kStages - 1;
    if (next < num_slices) issueSlice<kVec>(a_s[next % kStages], b_s[next % kStages], ..., next * kBlockK);
    __pipeline_commit();
    // ... compute on stage s % kStages ...
}
__pipeline_wait_prior(0);
```

Wait 數量可由 group 計數推得。步驟 $s$ 的 wait 之前，已 commit 的 group
為 $0, \dots, s + P - 2$（group $j$ 存放切片 $j$）：

$$
\text{committed} = s + P - 1, \qquad
\text{complete} \ \ge\ \text{committed} - (P - 2) = s + 1
\ \Rightarrow\ \text{slices } 0, \dots, s \text{ have landed}
$$

| 符號 | 意義 |
|---|---|
| $P$ | Stage 數（`kStages` = 3） |
| $s$ | 目前步驟（正在運算的切片） |

這就是接近結尾時仍須 commit **空 group** 的原因：否則「group $j$ =
切片 $j$」的對應會失效，最後幾個切片可能在抵達前就開始運算。模擬器能
抓出這項錯誤：把 wait 改成 `kStages - 1` 後，`--test` 會失敗，因為
cuemu 會延遲每次複製，直到對應的 wait 才完成。

需要多少 stage？要讓 $P - 1$ 個切片的運算時間足以涵蓋載入延遲：

$$
(P - 1)\,C \ \ge\ L \quad\Rightarrow\quad P \ \ge\ 1 + \left\lceil \frac{L}{C} \right\rceil, \qquad
\text{smem} = P\,(B_M + B_N)\,B_K \cdot 4\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $C$ | 一個切片的運算時間 |
| $L$ | 負載下的全域載入延遲（通常為 500–1000 個週期） |
| smem | 每個 block 的共享記憶體用量（不計 padding） |

此處 $P = 3$、$B_K = 8$，使用 30 KB。Tensor-core kernel 每個切片的
運算時間短得多，會使用 3–5 個較大切片的 stage，因此需要選用較高的
共享記憶體上限（`cudaFuncSetAttribute` 搭配
`cudaFuncAttributeMaxDynamicSharedMemorySize`）。

## 3. Kernel 有哪些變化

- **沒有暫存用暫存器。** 複製會直接進入共享記憶體。
- **不再轉置。** 複製只能搬移位元組，不能把 `float4` 分散至四列。因此
  $A$ 保持 row-major（`a_s[stage][m][k]`，每列 padding 至 12 個 float，
  讓列保持 16 位元組對齊），並以純量載入讀取 $A$ fragment。它們是
  broadcast（半個 warp 共用 $t_y$），所以會占用發射槽，但沒有 bank
  衝突。Tensor-core kernel 可完全避開此問題：`ldmatrix` 能以任一方向
  讀取 fragment（[04.7](07-tensor-cores.md)）。
- **未對齊的形狀。** 當 $K$ 或 $N$ 不是 4 的倍數時，列不會以 16 位元組
  對齊，因此 kernel 會具現化成使用 4 位元組複製（`kVec = 1`）：複製
  指令數是四倍，但使用相同 pipeline。

## 4. Hopper 上的 TMA { #4-tma-on-hopper }

TMA 每條指令可搬移一個完整的多維分塊：

1. Host 端使用 `cuTensorMapEncodeTiled` 建立 **tensor map**：矩陣的基底
   位址、大小與 stride、box（分塊）大小，以及選用的共享記憶體 swizzle
   （[04.7](07-tensor-cores.md#4-swizzled-shared-memory) 的 XOR 模式，由
   硬體套用）。
2. Kernel 中由**一個 thread** 以分塊座標發出
   `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`。
   Box 超出邊界的部分會填零。
3. 完成狀態由共享記憶體中的 **mbarrier** 追蹤：發出指令的 thread 會宣告
   預期的位元組數（`mbarrier.arrive.expect_tx`），TMA 單元逐步遞減計數，
   consumer 則等待 barrier 的 phase。

這會把位址運算與每個 thread 的複製指令完全移出主迴圈；每個切片只有幾條
`wgmma` 指令時尤其重要。Pipeline 接著會為每個 stage 使用兩個 mbarrier
（「full」與「empty」），而不是 `__syncthreads()`，且通常會採用
*warp specialization*：producer warp 發出 TMA 複製，consumer warpgroup
執行 MMA（[04.7 第 6 節](07-tensor-cores.md#6-hopper-wgmma-and-warp-specialization)）。
此儲存庫沒有 Hopper 範例程式；CUTLASS 的 `sm90` collective mainloop
是參考實作。

## 5. 常見陷阱

- **忘記在 wait 後加 barrier。** `wait_prior` 只涵蓋呼叫它的 thread 所
  發出的複製。
- **過早重新填入 stage。** 步驟 $s$ 會重新填入切片 $s - 1$ 所在的 stage；
  必須在步驟 $s$ 的 barrier *之後*執行，才能保證每個 warp 都已完成使用。
- **離開時仍有複製正在進行。** 在 epilogue 前（或把共享記憶體改作他用前）
  呼叫 `__pipeline_wait_prior(0)`。
- **對齊。** 16 位元組複製的來源與目的位址都必須以 16 位元組對齊；
  模擬器會檢查，GPU 則會發生錯誤。

## 重點整理

1. `cp.async` 不經暫存器，直接把全域記憶體複製到共享記憶體；wait 以 thread 與 group 為單位，因此之後仍要加 barrier。
2. 使用 $P$ 個 stage 時，等待到只剩 $P-2$ 個 group，再重新填入上一個切片釋放的 stage。
3. 每次迭代都要 commit 一個 group，即使是空 group，讓 group $j$ 永遠對應切片 $j$。
4. TMA 每條指令可搬移完整分塊，並透過 mbarrier 通知完成，進而支援 warp-specialized pipeline。

## 練習

1. 將 `kStages` 設為 `2` 與 `4`。當 $B_K = 16$ 時，哪一個需要 dynamic
   shared memory？

    <details markdown="1"><summary>答案</summary>

    一個 stage 是 $(128\times20 + 16\times128)\times4 = 18\,432$ 位元組。
    兩個 stage（36 KB）可放進 48 KB 的 static shared memory；三個（54 KB）
    與四個（72 KB）需要 dynamic shared memory 及 opt-in attribute。

    </details>
2. 比較 04.2 與此 kernel 主迴圈的指令數（`cuobjdump -sass`）。`STS`
   到哪裡去了？
3. 改用以 4 位元組 `cp.async` 複製寫入的轉置配置，取代純量 $A$ fragment
   載入（只有 $A$ 使用 `kVec = 1`）。速度有變快嗎？
