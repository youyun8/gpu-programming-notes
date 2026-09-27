# GPU 程式設計學習路徑

這些教學分成七條清楚的學習路徑。請從第一部分開始，接著可選擇
平行運算模式、矩陣乘法 1–9，或獨立的 Triton 章節。AMD 路徑以
矩陣乘法為基礎；Kimi K3 案例則把 Triton 或 FlyDSL 與 serving
framework 連接起來。發布部分可獨立閱讀。實作章節都在
[`examples/`](examples/check.cuh) 中附有經過測試的程式。

![從 GPU 基礎到 Triton、AMD 與 Kimi K3 框架案例的七條學習路徑](figures/overview-learning-path.svg)

## 第一部分 · 基礎

了解 GPU 如何執行程式，以及如何提供資料給它。這些知識足以針對各種逐元素、
歸約與正規化問題，撰寫正確且受記憶體頻寬限制的 kernel。

| # | 章節 | 主要主題 | 練習 |
|---|---|---|---|
| 00 | [開始使用](00-getting-started.md) | 工具鏈（PTX、SASS）、程式生命週期、錯誤檢查、計時、屋頂線模型 | 向量加法、ReLU |
| 01 | [執行模型](01-execution-model.md) | Grid／block／warp、索引、分歧、隱藏延遲、佔用率、stream | 向量加法、矩陣加法 |
| 02 | [記憶體階層與合併存取](02-memory-hierarchy.md) | 記憶體空間、合併存取、layout、`float4`、bank、swizzle、轉置 | 矩陣轉置、矩陣複製 |
| 03 | [平行歸約](03-parallel-reduction.md) | 工作量／深度、shuffle、跨 block 模式、逐列歸約、準確度、么半群 | 歸約、Softmax、點積 |
| 09 | [效能分析與剖析](09-profiling.md) | 計時、Nsight Systems 與 Compute、指標、停滯原因、佔用率、sanitizer、rocprof | 矩陣轉置、矩陣複製 |

## 第二部分 · 平行模式

歸約以外的基礎元件：warp 層級協作、scan、
鄰域運算，以及融合式 attention 背後的線上演算法。
每章都附有經過測試的範例程式。

| # | 章節 | 主要主題 | 程式 |
|---|---|---|---|
| 10 | [Warp 層級原語與 Cooperative Groups](10-warp-primitives.md) | Shuffle、vote、`match_any`、warp 聚合 atomic、壓縮、cooperative groups | [`10-warp-primitives.cu`](examples/10-warp-primitives.cu) |
| 11 | [Scan](11-scan.md) | Kogge-Stone 與 Brent-Kung、block scan、先歸約再 scan、decoupled look-back | [`11-scan.cu`](examples/11-scan.cu) |
| 12 | [卷積與 Stencil](12-convolution-stencils.md) | Halo、常數記憶體 filter、二維分塊、2.5-D stencil 分塊 | [`12-convolution-stencil.cu`](examples/12-convolution-stencil.cu) |
| 13 | [Softmax、LayerNorm 與 FlashAttention](13-softmax-attention.md) | 線上 softmax、Welford、以 register 內 tile 實作的融合式 attention | [`13-softmax-attention.cu`](examples/13-softmax-attention.cu) |

## 第三部分 · 矩陣乘法

這是一條由簡單迴圈一路通往生產環境 dispatcher，且內容完備、可獨立學習的九步驟路徑。
請依序閱讀。步驟 2–8 各自加入一項最佳化，並附有完整且經過測試的程式；
步驟 9 則將這些 kernel 連結至實際工作負載。

| 步驟 | 章節 | 主要主題 | 程式 |
|---|---|---|---|
| 1 | [基礎](04-tiled-matmul.md) | 成本模型、樸素 GEMM、共享記憶體與 register 分塊、epilogue | 內嵌 kernel |
| 2 | [向量化載入](gemm/01-vectorized-loads.md) | `LDG/LDS.128`、對齊、無衝突的 fragment layout | [`01-vectorized.cu`](gemm/01-vectorized.cu) |
| 3 | [雙緩衝](gemm/02-double-buffering.md) | 讓載入與數學運算重疊、每個 slice 一次 barrier | [`02-double-buffering.cu`](gemm/02-double-buffering.cu) |
| 4 | [非同步複製](gemm/03-async-copies.md) | `cp.async` pipeline 與 TMA | [`03-cp-async.cu`](gemm/03-cp-async.cu) |
| 5 | [Warp 分塊](gemm/04-warp-tiling.md) | Block → warp → lane 階層 | [`04-warp-tiling.cu`](gemm/04-warp-tiling.cu) |
| 6 | [Tile Swizzling](gemm/05-tile-swizzling.md) | 分組啟動順序與 L2 重複使用 | [`05-tile-swizzle.cu`](gemm/05-tile-swizzle.cu) |
| 7 | [Split-K 與 Stream-K](gemm/06-split-k-stream-k.md) | 提高平行度、部分 tile 與跨 block 修正 | [`06-split-k.cu`](gemm/06-split-k.cu)、[`07-stream-k.cu`](gemm/07-stream-k.cu) |
| 8 | [Tensor Core](gemm/07-tensor-cores.md) | WMMA、`ldmatrix`、`mma.sync`、共享記憶體 swizzle、`wgmma` | [`08-wmma.cu`](gemm/08-wmma.cu)、[`09-mma-sync.cu`](gemm/09-mma-sync.cu) |
| 9 | [生產環境 GEMM](gemm/08-production-gemm.md) | Persistent／grouped／batched GEMM、CUTLASS／CuTe、準確度、調校與分派 | 設計指南 |

[開啟完整的矩陣乘法 1–9 路線圖。](gemm/README.md)

## 第四部分 · Triton

這個獨立章節從第一個加上 mask 的向量運算開始，逐步介紹 Triton，
一路延伸至生產環境 kernel，並附有六個經過測試的範例與循序漸進的練習。

| # | 章節 | 主要主題 | 程式 |
|---|---|---|---|
| 14 | [Triton – 從第一個 Kernel 到生產環境](14-triton.md) | Program、mask、pointer／descriptor、fusion、歸約、LayerNorm、matmul、attention、atomic、scan、persistence、編譯器 layout 與剖析 | [`14-triton/`](examples/14-triton/test_kernels.py) |

## 第五部分 · AMD 架構與函式庫

在 AMD 的 CDNA3（MI300）上運用相同觀念：先介紹硬體與指令，
接著手寫 kernel，再介紹能產生數千個這類 kernel 的產生器。
這條路徑預設你已學過矩陣乘法步驟 1，最好也完成步驟 8。

| # | 章節 | 主要主題 | 練習 |
|---|---|---|---|
| 05 | [CDNA3 與 MFMA](05-amd-cdna3-mfma.md) | 詞彙對照、wave64、MFMA 運算元 layout、教學用 kernel 及其 ISA | [`amd/mfma_gemm.hip`](amd/mfma_gemm.hip) |
| 06 | [深入手寫 AMD GEMM](06-aiter-asm-gemm.md) | AITER 分派、每個 SIMD 一個 wave、direct-to-LDS、交錯執行、split-K | 反組譯 AITER `.co` 檔案 |
| 07 | [hipBLASLt 與 TensileLite](07-hipblaslt-tensilelite.md) | Solution 參數、kernel 名稱、選擇、離線調校 | `hipblaslt-bench` |

## 第六部分 · Kimi K3 框架案例研究

這些章節追蹤真實的框架路徑，同時清楚區分各後端的界線。

| # | 章節 | 主要主題 | 程式 |
|---|---|---|---|
| 15 | [SGLang 中的 Triton – 以 Kimi K3 服務為例](15-triton-model-systems.md) | SiTU、state cache、KDA decode／prefill／speculation、MLA、MoE、量化、sampling 與後端分派 | [`15-triton-k3/`](examples/15-triton-k3/test_model_kernels.py) |
| 16 | [AITER 中的 FlyDSL – 在 AMD GPU 上執行 Kimi K3](16-aiter-flydsl-kimi-k3.md) | Layout／copy／MMA 模型、AITER 分派、量化兩階段 MoE、MLA、通訊與完整配方 | AITER 與 FlyDSL 標記測試 |

## 第七部分 · 發布

| # | 章節 | 主要主題 | 程式 |
|---|---|---|---|
| 08 | [部署這個網站](08-deploying-this-site.md) | 網站建置器、GitHub Pages、靜態主機、EPUB/PDF、CI、圖表 | |

## 每一章的編排方式

1. **頁首**：所屬部分、先備知識和下一章。
2. **你將學到**：條列學習目標。
3. **編號章節與小節**（1、1.1、1.2……）：各自推導公式
   （每個獨立公式後都附有符號表），並展示完整 kernel。
4. **重點整理**：需要記住的少數要點。
5. **練習**：多數附有可展開的提示或答案。
6. **實作練習**：運用本章內容的題目頁面。

每個題目頁面都有相同小節：*問題*、*公式化*、
*方法*、*成本分析*、*常見問題*、*驗證*、*相關內容*，最後是
完整解答原始碼。

## 請牢記這個公式

每一章與每個題目頁面都會回到屋頂線界限
（第 00 章）：

$$
T_{\min} = \max\left(\frac{W}{F},\ \frac{Q}{\beta}\right), \qquad I = \frac{W}{Q}, \qquad I^{\star} = \frac{F}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $W$ | kernel 的有效浮點運算次數 |
| $Q$ | 必須在 DRAM 之間搬移的位元組數 |
| $F$ | 峰值計算吞吐量 |
| $\beta$ | DRAM 峰值頻寬 |
| $T_{\min}$ | 理論上可達到的最佳時間 |
| $I, I^{\star}$ | kernel 的算術強度，以及 GPU 的轉折點 |

第一和第二部分著重於讓 $I < I^{\star}$ 的 kernel 達到 $Q/\beta$
（幾乎所有逐元素、歸約、scan、stencil 與正規化問題）。
第三和第四部分則著重於在記憶體階層的每一層提高矩陣乘法的*有效* $I$，
直到 $W/F$ 成為限制。

## 執行範例程式

```bash
make examples-test                     # every examples/*.cu on the CPU emulator
nvcc -O3 -arch=sm_80 -std=c++17 -lineinfo tutorials/examples/11-scan.cu -o scan && ./scan --bench
make triton-test                     # chapters 14-15 (interpreter without a GPU)
```

## 延伸閱讀

- *Programming Massively Parallel Processors*（Hwu、Kirk、El Hajj），第 4 版
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Mark Harris，*Optimizing Parallel Reduction in CUDA*
- Simon Boehm，*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*
- AMD，*AMD Instinct MI300 ISA Reference Guide*（CDNA3）
- AMD，[Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- AITER，`docs/isa_kernel_optimization.md`
- Osama 等人，*Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU*（2023）
