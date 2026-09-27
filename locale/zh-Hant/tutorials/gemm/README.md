# 矩陣乘法：1–9 路線圖

> **第三部分 · 矩陣乘法** · 起點：[矩陣乘法 1 – 基礎](../04-tiled-matmul.md)

這條獨立路線先介紹算術與記憶體模型，再逐項加入技巧來建構快速的分塊
kernel，最後說明在實際系統中部署 GEMM 所需的決策。請依序閱讀各章：
每個實作章節都以前一個程式為起點，並只做一項主要變更。

| 步驟 | 章節 | 核心概念 | 程式 |
|---|---|---|---|
| 1 | [基礎](../04-tiled-matmul.md) | 算術強度、共享記憶體分塊、暫存器分塊與融合 epilogue | 行內 kernel |
| 2 | [向量化載入](01-vectorized-loads.md) | 128 位元全域與共享記憶體存取，以及無衝突的 fragment 配置 | [`01-vectorized.cu`](01-vectorized.cu) |
| 3 | [雙緩衝](02-double-buffering.md) | 讓下一個切片的載入與目前的運算重疊 | [`02-double-buffering.cu`](02-double-buffering.cu) |
| 4 | [非同步複製](03-async-copies.md) | `cp.async` 多階段 pipeline，以及 Hopper TMA | [`03-cp-async.cu`](03-cp-async.cu) |
| 5 | [Warp 分塊](04-warp-tiling.md) | 對應 block → warp → lane 的硬體階層 | [`04-warp-tiling.cu`](04-warp-tiling.cu) |
| 6 | [分塊 Swizzle](05-tile-swizzling.md) | 將分塊啟動分組，以提高 L2 重用 | [`05-tile-swizzle.cu`](05-tile-swizzle.cu) |
| 7 | [Split-K 與 Stream-K](06-split-k-stream-k.md) | 輸出分塊太少時增加平行度 | [`06-split-k.cu`](06-split-k.cu)、[`07-stream-k.cu`](07-stream-k.cu) |
| 8 | [Tensor Core](07-tensor-cores.md) | WMMA、`ldmatrix`、`mma.sync`、共享記憶體 swizzle 與 `wgmma` | [`08-wmma.cu`](08-wmma.cu)、[`09-mma-sync.cu`](09-mma-sync.cu) |
| 9 | [Production GEMM](08-production-gemm.md) | Persistent 與 grouped kernel、融合、精度、調校、dispatch 與量測 | 設計指南 |

步驟 2–8 都包含經完整測試的程式。比較相鄰程式的差異，即可看出每項
最佳化增加了哪些程式碼。步驟 9 會整合這些技巧，並說明何時應在
production 環境改用函式庫。

## 每一頁都會改進的階層

![GEMM 分塊階層：每一層都為下一層準備資料](../figures/gemm-hierarchy.svg)

快速 GEMM 就是把同一個概念套用到記憶體階層的每一層：將 $A$ 與 $B$ 的
分塊放入更快的記憶體，並依分塊允許的程度，盡可能重複使用每個元素。
若每個工作單位（block、warp 或 lane）負責一個 $T_M\times T_N$ 輸出分塊，
而歸約步長為 $T_K$：

$$
\frac{\text{FMAs}}{\text{elements loaded}} = \frac{T_M T_N T_K}{(T_M + T_N)\,T_K} = \frac{T_M T_N}{T_M + T_N}
$$

| 符號 | 意義 |
|---|---|
| $T_M, T_N$ | 該層級一個工作單位負責的輸出列數與欄數 |
| $T_K$ | 一次準備的歸約深度（會相消） |

| 層級 | 暫存位置 | 分塊 | 重用率 $T_MT_N/(T_M+T_N)$ |
|---|---|---|---|
| Block | 共享記憶體 | $128\times128$ | 從 L2 載入每個元素可進行 64 次 FMA |
| Warp | （warp 所見的共享記憶體） | $64\times32$ | 從共享記憶體讀取每個元素可進行 21.3 次 FMA |
| Lane | 暫存器 | $8\times8$ | 讀入暫存器的每個元素可進行 4 次 FMA |

[矩陣乘法 1 – 基礎](../04-tiled-matmul.md)會建立 block 與 lane 兩個層級。
步驟 2 會加寬載入；步驟 3 與 4 隱藏載入延遲；步驟 5 加入 warp 層級；
步驟 6 讓 block 透過 L2 合作；步驟 7 在分塊很少時仍讓所有 SM 保持忙碌；
步驟 8 以 tensor-core 指令取代 lane 層級的 FMA。步驟 9 則把 kernel
知識轉化為 production 環境的 dispatch 與驗證策略。

## 執行程式

每支程式都有相同的命令列介面，由 [`harness.cuh`](harness.cuh) 提供：

```bash
cd tutorials/gemm
nvcc -O3 -arch=sm_80 -std=c++17 04-warp-tiling.cu -o warp_tiling
./warp_tiling                # M = N = K = 4096: time, TFLOP/s, spot check of 256 entries
./warp_tiling 2048 512 8192  # any shape
./warp_tiling --test         # awkward shapes, every entry checked against a CPU reference
```

沒有 GPU？[cuemu](../../tools/cuemu/README.md) 模擬器可在 CPU 上執行每支程式的
`--test` 模式，包括 `cp.async` pipeline、`ldmatrix` 與 `mma.sync`。它實作了
這些指令的文件語意，因此 fragment 索引錯誤或漏掉 wait 也會測試失敗：

```bash
python3 tools/cuemu/cuemu.py run tutorials/gemm/09-mma-sync.cu -- --test
make gemm-test               # all of them, as CI does
```

模擬器只檢查正確性。計時需要 GPU；各頁引用的數字是文獻中的典型範圍，
並非此儲存庫的實測結果。

## 延伸閱讀

- Simon Boehm，*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*（2022）。
- NVIDIA，[CUTLASS](https://github.com/NVIDIA/cutlass) 及其
  `media/docs`（高效率 GEMM、CuTe 配置、swizzle）。
- NVIDIA，*PTX ISA* 中關於 `cp.async`、`ldmatrix`、`mma` 與 `wgmma` 的章節。
- Osama 等人，*Stream-K: Work-centric Parallel Decomposition for Dense
  Matrix-Matrix Multiplication on the GPU*（PPoPP 2023）。
- Triton，*Matrix Multiplication* 教學（分組排序）。
