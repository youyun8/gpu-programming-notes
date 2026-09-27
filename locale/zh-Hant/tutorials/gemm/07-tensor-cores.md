# 04.7 – Tensor Core：WMMA、`mma.sync` 與 `wgmma`

> **第三部分 · 矩陣乘法 · 04.x GEMM 深入解析** ·
> 程式：[`08-wmma.cu`](08-wmma.cu)、[`09-mma-sync.cu`](09-mma-sync.cu) · 延續：[04.3](03-async-copies.md)、[04.4](04-warp-tiling.md) ·
> 下一篇：[05 – AMD CDNA3 與 MFMA](../05-amd-cdna3-mfma.md)

Tensor core 每條 warp 指令會執行一次小型矩陣乘加。使用 16 位元輸入時，
吞吐量約為同一 GPU FP32 FMA 的 8–16 倍（A100：dense FP16/BF16 為
312 TFLOP/s，FP32 為 19.5 TFLOP/s）。前幾頁的內容仍全部適用（block 分塊、
pipeline、warp 分塊）；改變的是最內層：lane 的 $8\times8$ outer product
改成 warp 的 $16\times8\times16$ MMA。

**你將學到**

- 三種 tensor-core 介面（WMMA、`mma.sync`、`wgmma`）及其適用時機；
- 完整的 WMMA kernel 及其對齊規則；
- `mma.sync.m16n8k16` 的文件化暫存器配置，以及利用該配置的 epilogue；
- 為何 tensor-core 分塊會在共享記憶體產生衝突，以及 XOR swizzle 如何在不加 padding 的情況下解決；
- `ldmatrix` 如何載入完整 fragment，包括轉置後的 B operand；
- Hopper kernel 的結構：TMA、`wgmma`、mbarrier 與 warp specialization。

## 1. 三種介面

| 介面 | 架構 | 單位 | Fragment 配置 | 使用者 |
|---|---|---|---|---|
| WMMA (`nvcuda::wmma`) | sm_70+ | Warp，$16\times16\times16$（FP16） | 不透明 | 可攜式 CUDA C++ |
| `mma.sync`（PTX） | sm_80+（m16n8k16） | Warp，$16\times8\times16$ | 有文件說明 | CUTLASS 2.x、FlashAttention 2 |
| `wgmma.mma_async`（PTX） | sm_90a | Warpgroup（4 個 warp），$64\times N\times16$，$N \le 256$ | Operand 位於共享記憶體 | CUTLASS 3.x（Hopper） |

第 05 章介紹過 AMD 的對應指令 MFMA，其暫存器配置也有文件說明。

## 2. WMMA：簡單的入門方式

[`08-wmma.cu`](08-wmma.cu) 保留 block 分塊（$128\times128$、$B_K = 32$）、
04.4 的 8 個 warp（每個負責 $64\times32$ warp 分塊），以及雙緩衝
`cp.async` pipeline。每個 warp 會保存 $4\times2$ 個 $16\times16$
累加器 fragment：

```cpp
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kFragsM][kFragsN];   // 4 x 2
...
for (int kk = 0; kk < kBlockK; kk += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[kFragsM];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[kFragsN];
    for (int i = 0; i < kFragsM; ++i)
        wmma::load_matrix_sync(a_frag[i], &a_s[buf][warp_m * kWarpTileM + 16 * i][kk], kStrideA);
    for (int j = 0; j < kFragsN; ++j)
        wmma::load_matrix_sync(b_frag[j], &b_s[buf][kk][warp_n * kWarpTileN + 16 * j], kStrideB);
    for (int i = 0; i < kFragsM; ++i)
        for (int j = 0; j < kFragsN; ++j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
}
```

WMMA 的限制在記憶體，而非運算：

- `load_matrix_sync` 與 `store_matrix_sync` 需要**以 32 位元組對齊的指標**，
  leading dimension 也必須是 **16 位元組的倍數**。共享分塊會 padding
  8 個 half（每列 80 與 272 位元組），同時符合兩項要求並錯開 bank。
- Fragment 的元素至 lane 映射未明確規定，因此 epilogue 必須先經過共享
  記憶體中的每 warp 暫存分塊，再對 $C$ 進行邊界檢查後儲存。

## 3. `mma.sync.m16n8k16`：文件化的 Fragment

PTX 指令會依固定配置從暫存器取得 operand。對 lane $\ell$，令
$g = \ell / 4$、$t = \ell \bmod 4$：

![mma.m16n8k16 中每個 lane 保存的 A、B、C 元素](../figures/gemm-mma-layout.svg)

$$
a_0 = A[g][2t{:}2t{+}1],\ \ a_1 = A[g{+}8][2t{:}2t{+}1],\ \ a_2 = A[g][2t{+}8{:}2t{+}9],\ \ a_3 = A[g{+}8][2t{+}8{:}2t{+}9]
$$

$$
b_0 = B[2t{:}2t{+}1][g],\ \ b_1 = B[2t{+}8{:}2t{+}9][g], \qquad
(d_0, d_1) = C[g][2t{:}2t{+}1],\ \ (d_2, d_3) = C[g{+}8][2t{:}2t{+}1]
$$

| 符號 | 意義 |
|---|---|
| $\ell$ | Lane，0–31 |
| $g, t$ | 列群組（$\ell / 4$，0–7）及群組內 thread（$\ell \bmod 4$，0–3） |
| $a_0 \dots a_3$ | 四個 32 位元暫存器，每個包含 $A$（16 × 16）的兩個 FP16 值 |
| $b_0, b_1$ | 兩個 32 位元暫存器，每個包含 $B$（16 × 8）的兩個 FP16 值 |
| $d_0 \dots d_3$ | $C$（16 × 8）的四個 FP32 累加器 |

因配置已知，[`09-mma-sync.cu`](09-mma-sync.cu) 的 epilogue 可直接儲存累加器：

```cpp
const int g = lane / 4, t = lane % 4;
const int col = col0 + warp_col + 8 * j + 2 * t;
for (int h = 0; h < 2; ++h) {
    const int row = row0 + warp_row + 16 * i + g + 8 * h;
    if (row < m)
        *reinterpret_cast<float2*>(&c[row * n + col]) = make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
}
```

這項資訊也能支援 WMMA 難以完成的融合：逐列縮放、bias、activation，以及
FlashAttention 的 softmax 統計，全都可在暫存器中套用。

## 4. Swizzle 共享記憶體 { #4-swizzled-shared-memory }

Fragment 從共享記憶體讀取時，以 8 列 × 16 位元組為單位（見下方
`ldmatrix`）。在一般 row-major 分塊中，這 8 列之間相隔整數個 128 位元組
cache line，因此會落在相同 bank：

$$
\operatorname{group}(r, c) = \left(\frac{r\cdot R}{16} + c\right) \bmod 8
$$

| 符號 | 意義 |
|---|---|
| $r$ | 分塊中的列 |
| $c$ | 列內的 16 位元組 chunk |
| $R$ | 每列的位元組長度（$A$ 切片為 64，$B$ 為 256） |
| group | Chunk 位於 128 位元組 cache line 的哪一個 4-bank 群組 |

對 $B$（$R = 256$），$rR/16 = 16r$ 是 8 的倍數，因此 fragment 的 8 列
都落在同一 bank 群組，形成 8-way 衝突。Padding（如 WMMA kernel）可以
解決，但會耗用記憶體，也讓對齊更麻煩；標準替代方案是使用 **XOR swizzle**
排列每列的 chunk：

![XOR swizzle：以列索引對 chunk 欄進行 XOR](../figures/gemm-smem-swizzle.svg)

```cpp
// A slice: 4 chunks per row, two rows per 128-byte line.
__device__ int offsetA(int row, int col) { return row * 32 + (((col / 8) ^ ((row >> 1) & 3)) * 8) + col % 8; }
// B slice: 16 chunks per row.
__device__ int offsetB(int row, int col) { return row * 128 + (((col / 8) ^ (row & 7)) * 8) + col % 8; }
```

以列的位元對 chunk 做 XOR，會形成每列 chunk 的排列，因此分塊每列仍正好
占用 $R$ 位元組，每個 chunk 也保持 16 位元組對齊（`cp.async` 與
`ldmatrix` 不需更動），而相同邏輯 chunk 的連續 8 列會落入 8 個不同群組。
Writer（`issueSlice`）與 reader（`ldmatrix` 位址）只需使用相同函式。
CuTe 稱之為 `Swizzle<3, 3, 3>` 類配置；Hopper TMA 會由硬體套用相同模式
（`CU_TENSOR_MAP_SWIZZLE_128B`）。

## 5. `ldmatrix`：一條指令載入 Fragment

若用一般載入取得 $a_0 \dots a_3$，每個 lane 需要 4 個 `LDS.32`，且位址
計算複雜。`ldmatrix.sync.aligned.m8n8.x4.shared.b16` 可為整個 warp 載入
四個由 16 位元值組成的 $8\times8$ 矩陣：

![ldmatrix.x4：lane 提供列位址，暫存器接收 fragment](../figures/gemm-ldmatrix.svg)

對 $A$ fragment，lane $\ell$ 會指向 $16\times16$ 分塊的列
$(\ell \bmod 8) + 8\,(\lfloor \ell/8 \rfloor \bmod 2)$ 與 chunk
$\lfloor \ell / 16 \rfloor$；四個結果暫存器正好就是 $a_0 \dots a_3$。
對以 $k$ 為主序儲存的 $B$，`.trans` 變體會提供轉置，一個 `x4` 則可涵蓋
兩個 $n8$ 分塊：

```cpp
// A: one ldmatrix.x4 per m16 tile
const int row = warp_row + 16 * i + lane % 8 + 8 * ((lane / 8) % 2);
const int col = kk + 8 * (lane / 16);
ldmatrixX4<false>(a_frag[i], as + offsetA(row, col));
// B: one ldmatrix.x4.trans per pair of n8 tiles
const int q = lane / 8;
ldmatrixX4<true>(r, bs + offsetB(kk + lane % 8 + 8 * (q % 2), warp_col + 16 * p + 8 * (q / 2)));
```

每個 $k16$ 步驟中，一個 warp 會發出 4 + 2 個 `ldmatrix` 及 16 個
`mma.sync`（$4\times4$ 個 $16\times8$ 分塊，組成其 $64\times32$ warp
分塊）。使用 $128\times128\times32$ 切片的 3-stage `cp.async` pipeline
時，kernel 會使用 48 KiB 共享記憶體，以及每個 thread 125 個暫存器。

**沒有 GPU 也能檢查。** Inline PTX 無法在 CPU 上執行，因此
`09-mma-sync.cu` 在 `#ifdef __CUEMU__` 下會透過小型 wrapper 呼叫
`cuemuLdmatrix` / `cuemuMmaM16N8K16`。這些函式實作上述 PTX 文件配置
（32 個 lane 全部會合、交換暫存器並計算），所以 lane 映射錯誤、漏掉
`.trans`，或 writer 與 reader 的 swizzle 不一致，都會讓 `--test` 失敗。

## 6. Hopper：`wgmma` 與 Warp Specialization { #6-hopper-wgmma-and-warp-specialization }

在 sm_90 上，最佳 kernel 的形態再次改變：

![Hopper pipeline：TMA producer、wgmma consumer、mbarrier](../figures/gemm-hopper.svg)

- **`wgmma.mma_async`** 由 *warpgroup*（連續 4 個 warp，共 128 個 thread）
  發出，用於 $64\times N\times16$ 分塊，其中 $N$ 最大為 256。$B$
  （也可包括 $A$）會透過 *matrix descriptor*（位址、leading/stride
  位元組偏移、swizzle 模式）直接從共享記憶體讀取；累加器留在暫存器中。
  它是非同步指令：由 `wgmma.fence`、`wgmma.commit_group` 與
  `wgmma.wait_group` 包圍，概念類似 `cp.async` group。
- **TMA**（[04.3](03-async-copies.md#4-tma-on-hopper)）負責填滿 stage；
  tensor map 的共享記憶體 swizzle 必須與 descriptor 相符。
- **Warp specialization。** Producer warp（通常透過 `setmaxnreg` 使用較少
  暫存器）只發出 TMA 複製；一至兩個 consumer warpgroup 只發出 `wgmma`。
  它們以每 stage 的「full」/「empty」mbarrier 同步，而非 block-wide barrier。
- **Thread block cluster** 可將一次 TMA 載入 multicast 至數個需要相同
  $A$ 或 $B$ 分塊的 block 共享記憶體。

雖然可依 PTX ISA 文件手寫，程式會很長；CUTLASS 3 的 sm90
`CollectiveMma` 與 ThunderKittens 是較易讀的參考。此儲存庫不含 Hopper
程式，因為目前的測試基礎設施都無法執行或模擬它。

## 7. 常見陷阱

- **以 FP32 累加。** FP16 累加在 $K$ 很長時會溢位並損失精度；此處每個
  kernel 都使用 FP32 累加器。
- **對齊與形狀限制。** 程式要求 $K$ 與 $N$ 是 8 的倍數（`cp.async`
  需要 16 位元組列）。函式庫會用 padding 複製或較慢的備援 kernel
  處理其他形狀。
- **一對元素在暫存器內的順序。** 每個 32 位元暫存器中，欄（或 $k$）
  索引較小的元素位於低半部。
- **`ldmatrix` 位址**必須以 16 位元組對齊，且位於共享記憶體
  （`__cvta_generic_to_shared`）。

## 重點整理

1. Tensor core 以 warp-wide MMA 取代 lane 的 outer product；block 與 warp 層級維持不變。
2. WMMA 可攜但不透明；`mma.sync` fragment 配置有文件說明，因此 epilogue 可在暫存器中運作。
3. Tensor core 每次從共享記憶體讀取 8 列 × 16 位元組：用列位元對 16 位元組 chunk 進行 swizzle。
4. `ldmatrix` 把 32 個列位址轉成可直接使用的 fragment；`.trans` 可處理以 $k$ 為主序的 B。
5. Hopper 把 operand 移至共享記憶體（`wgmma`），把複製交給 TMA，並以 mbarrier 同步。

## 練習

1. 將 `09-mma-sync.cu` 改為 BF16（`mma.sync...bf16.bf16.f32`、
   `__nv_bfloat16`）。還需要改什麼？

    <details markdown="1"><summary>答案</summary>

    Harness 型別與轉換（`__float2bfloat16`），以及模擬器：
    `cuemuMmaM16N8K16` 會解碼 FP16，因此 BF16 版本需要 BF16 解碼。
    Fragment 配置、`ldmatrix` 與 swizzle 不變；不論格式為何，它們搬移的
    都是 16 位元值。

    </details>
2. 移除 swizzle（改用 `row * 32 + col` 與 `row * 128 + col`），並以
   Nsight Compute 測量 bank 衝突。接著改用 padding 修正；3 個 stage
   會增加多少共享記憶體？

    <details markdown="1"><summary>答案</summary>

    每列 padding 8 個 half（16 位元組）可保持 16 位元組對齊：每個 stage
    中，A 變成 $128\times40\times2 = 10\,240$ 位元組，B 變成
    $32\times136\times2 = 8\,704$ 位元組，三個 stage 共
    $3\times18\,944 = 56\,832$ 位元組。這超過 48 KB static shared memory，
    所以 kernel 需要 dynamic shared memory；swizzle 版本則正好放進 48 KB。

    </details>
3. 依配置公式，使用每個 lane 四次 32 位元載入，取代 $A$ 的 `ldmatrix`。
   用 `--test` 驗證，再比較指令數。
