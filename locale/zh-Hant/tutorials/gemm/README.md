# 04.x – GEMM 深入解析：最佳化階梯的其餘部分

> **第三部分 · 矩陣乘法** · 先備知識：[04 – 分塊矩陣乘法](../04-tiled-matmul.md) ·
> 下一篇：[04.1 – 向量化載入](01-vectorized-loads.md)

[第 04 章](../04-tiled-matmul.md)最後列出一系列技巧，能讓使用暫存器分塊的
SGEMM 從約 cuBLAS 一半的效能，提升到僅差幾個百分點，接著再進入 tensor core。
本節各頁會說明每一項技巧，並將它實作成完整且經過測試的程式：

| 頁面 | 技巧 | 程式 |
|---|---|---|
| [04.1](01-vectorized-loads.md) | 128 位元全域與共享記憶體存取，以及無衝突的 fragment 配置 | [`01-vectorized.cu`](01-vectorized.cu) |
| [04.2](02-double-buffering.md) | 雙緩衝：載入下一個切片時，同時進行運算 | [`02-double-buffering.cu`](02-double-buffering.cu) |
| [04.3](03-async-copies.md) | `cp.async` 多階段 pipeline，以及 Hopper 上的 TMA | [`03-cp-async.cu`](03-cp-async.cu) |
| [04.4](04-warp-tiling.md) | Warp 分塊：block → warp → lane | [`04-warp-tiling.cu`](04-warp-tiling.cu) |
| [04.5](05-tile-swizzling.md) | 為 L2 重用安排 swizzle（「分組」）分塊順序 | [`05-tile-swizzle.cu`](05-tile-swizzle.cu) |
| [04.6](06-split-k-stream-k.md) | 輸出分塊太少時使用 Split-K 與 Stream-K | [`06-split-k.cu`](06-split-k.cu)、[`07-stream-k.cu`](07-stream-k.cu) |
| [04.7](07-tensor-cores.md) | Tensor core：先用 WMMA，再以 `ldmatrix` + `mma.sync` 搭配 swizzle 共享記憶體；以及 Hopper 的 `wgmma` | [`08-wmma.cu`](08-wmma.cu)、[`09-mma-sync.cu`](09-mma-sync.cu) |

請依序閱讀。每支程式都以前一支為起點，且只改一件事，因此比較相鄰檔案的
差異，就能清楚看出該技巧增加了哪些程式碼。

每一頁的結構都相同：學習目標、搭配圖解的概念、成本模型（公式與符號表）、
關鍵程式碼、常見陷阱、重點整理，以及附答案的練習。

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

第 04 章建立了 block 與 lane 兩個層級。接下來各頁會加寬載入（04.1）、
隱藏載入延遲（04.2、04.3）、加入 warp 層級（04.4）、讓 block 透過 L2
合作（04.5）、在分塊很少時仍讓所有 SM 保持忙碌（04.6），最後以
tensor-core 指令取代 lane 層級的 FMA（04.7）。

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
