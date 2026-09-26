# 07 – hipBLASLt 與 TensileLite：由程式撰寫的 GEMM Kernel

> **第五部 · AMD Production Kernel** · 先備知識：[05](05-amd-cdna3-mfma.md)、[06](06-aiter-asm-gemm.md) ·
> 下一章：[14 – Triton](14-triton.md)（第四部）

AITER（第 06 章）手寫數十個 GEMM kernel。[hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/) 則發布了**數千個**：ROCm 的 `libhipblaslt` 對每種 GPU architecture 都包含一組 code object，而 PyTorch 預設用它在 MI300 上執行 `torch.matmul`。它們不是人工撰寫，而是由 **TensileLite** 產生。這個 Python 程式接收 parameter list，為每種組合輸出完整 assembly kernel，再 benchmark 各組合以決定發布哪些。

**你將學會**

- TensileLite *solution* 是什麼，以及主要 parameter 的意義：tile hierarchy（`MatrixInstruction`、`DepthU`）、global/local read、LDS layout、instruction scheduling、工作切分與 tile order；
- 各 parameter 如何對應第 05–06 章（及 NVIDIA 04.1–04.7 頁面）中手工完成的技術；
- 如何從 profile 解讀 `Cijk_…` kernel 名稱；
- hipBLASLt 如何在 runtime 選擇 kernel，以及如何針對自己的 shape 調校；
- TensileLite 如何產生、benchmark 與發布 kernel。

> **程式碼位置。** hipBLASLt 原為獨立的 `ROCm/hipBLASLt` 儲存庫，現已退役到 `develop_deprecated` branch。開發移至 [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) monorepo 的 `projects/hipblaslt`。Generator 位於 `tensilelite/Tensile/`：
> - `KernelWriterAssembly.py`、`KernelWriter.py`：generator。
> - `Components/`：可插拔組件，如 `SIA.py`、`StreamK.py`、`GSU.py`、`LocalRead.py`、`MAC_*.py`。
> - `Common/ValidParameters.py`：每個 parameter 與註解。
> - `SolutionStructs/`：validation 與 naming。
>
> 以下檔案參照皆以這些路徑為準。

## 1. 從「一個 GEMM」到「一個 Solution」

TensileLite **solution** 是大型 design space 中的一點：

| 群組 | Parameter | 第 05/06 章中的對應概念 |
|-------|------------|--------------------------|
| Tile shape | `MatrixInstruction`、`DepthU` | MFMA shape、wave tile、每 workgroup wave 數、K step |
| Global → LDS | `PrefetchGlobalRead`（PGR）、`DirectToLds`（DTL）、`GlobalReadVectorWidth`、`BufferLoad` | Prefetch depth、direct-to-LDS load |
| LDS | `1LDSBuffer`、`LdsPadA/B`、`LdsBlockSizePerPad`、`TransposeLDS` | Double buffering、bank-conflict padding |
| LDS → register | `PrefetchLocalRead`（PLR）、`ClusterLocalRead` | Register double buffering（`a[0:63]` / `a[64:127]`） |
| Scheduling | `ScheduleIterAlg`（SIA）、`GlobalReadPerMfma`、`LocalWritePerMfma` | 在 MFMA 間 interleave load |
| 工作切分 | `GlobalSplitU`（GSU）、`GlobalSplitUAlgorithm`、`StreamK` | Split-K、Stream-K |
| Cache 行為 | `WorkGroupMapping`（WGM）、`WorkGroupMappingXCC`（WGMXCC）、`StaggerU*` | Tile order、XCD placement、DRAM channel 分散 |
| Epilogue | `StoreRemapVectorWidth`、`StoreVectorWidth`、activation / bias / scaling fusion | Coalesced store |

### 1.1 `MatrixInstruction`：以 9 個數字表達 Tile Hierarchy

`ValidParameters.py` 中的註解解釋 9-number 格式：

```
[32, 32, 1, 2,   1,   4, 1,   2, 2]
 ^^^^^^^^^^^^    ^    ^^^^    ^^^^
 MFMA MxNxKxB  BlkM  WaveTile  Waves
```

- **MFMA** `32x32x1x2` 是 2-block MFMA variant。`MIBlockM = 1` 時，每個 instruction 涵蓋 32×64。
- **WaveTile** `4×1`：每個 wave 發出 4×1 個上述 instruction，涵蓋 128×64。
- **Waves** `2×2`：每 workgroup 四個 wave，所以 **macro tile** 是 (32·4·2) × (64·1·2) = **256×128**。

一般而言，將 9 個數寫成 $[m, n, k, b,\ \beta_M,\ w_M, w_N,\ W_M, W_N]$：

$$
\text{MT}_0 = m\,\beta_M\,w_M\,W_M, \qquad
\text{MT}_1 = n\,\frac{b}{\beta_M}\,w_N\,W_N, \qquad
\text{threads} = 64\,W_M W_N
$$

| 符號 | 意義 |
|---|---|
| $m, n, k$ | MFMA shape（如 32、32、1） |
| $b$ | MFMA 一次計算的 block 數（multi-block variant）；大多數為 1 |
| $\beta_M$ | `MIBlockM`：沿 M 疊放的 $b$ block 數（其餘沿 N） |
| $w_M, w_N$ | WaveTile：每 wave 沿 M、N 的 MFMA tile 數 |
| $W_M, W_N$ | 每 workgroup 沿 M、N 的 wave 數 |
| $\text{MT}_0, \text{MT}_1$ | 沿 M、N 的 macro tile（workgroup tile） |

此例中 $\text{MT}_0 = 32\cdot1\cdot4\cdot2 = 256$、$\text{MT}_1 = 32\cdot2\cdot1\cdot2 = 128$，共 256 threads。

![MatrixInstruction [32, 32, 1, 2, 1, 4, 1, 2, 2]：MFMA tile、wave tile 與 macro tile](figures/ch07-macro-tile.svg)

gfx942 bf16 kernel 多半使用 `16x16x16` 或 `32x32x8` MFMA，寫成 `[16,16,16,1, 1, …]`。

較大的 WaveTile 能提高第 05 章教學 kernel 所欠缺的 MFMA-per-byte ratio，但需要更多 accumulator register：
- 128×64 fp32 wave tile 有 8192 個 value 分到 64 lane，即每 lane 128 個 AGPR。
- 因此快速 kernel 每 SIMD 只跑一或兩個 wave。

Accumulator 成本與 reuse 都可由 wave tile $T_M\times T_N$（MFMA tile 乘以 WaveTile）推出：

$$
r_{\text{acc}} = \frac{T_M T_N}{64}, \qquad
\frac{\text{MFMAs}}{\text{operand fetches}} = \frac{w_M w_N}{w_M + w_N}
$$

| 符號 | 意義 |
|---|---|
| $T_M, T_N$ | 每 wave 沿 M、N 的 output element 數 |
| $r_{\text{acc}}$ | 每 lane 的 fp32 accumulator register 數 |
| $w_M, w_N$ | 每 wave 的 MFMA tile；每個 A fragment reuse $w_N$ 次，每個 B fragment reuse $w_M$ 次 |

16x16 MFMA 的 $4\times4$ WaveTile（$T_M = T_N = 64$）需要每 lane 64 個 accumulator register，並 reuse 每個 operand fragment 4 次；$128\times64$ 需要 128 個。

**`DepthU`** 是一次 main-loop iteration 的 K extent：AITER kernel 中為 64；16-bit type 通常為 32–128。

### 1.2 Global Read：PGR、DirectToLds

`ValidParameters.py` 對 `PrefetchGlobalRead` 的說明：

- **PGR=0：** 不 prefetch。Load、wait、寫入 LDS，再 compute。
- **PGR=1：** double-buffer *global → VGPR → LDS* 路徑。需要兩倍 LDS 與 staging VGPR。
- **PGR=2：** 在 staged data 寫入 LDS 時再發出一次 global prefetch，因此同時有兩個 tile in flight。

這正是 AITER `pf3` kernel 寫死的行為，也是第 06 章 `vmcnt(N)` 所表達的工作。

`DirectToLds=1` 使用 `buffer_load … lds`（第 06 章第 4 節）：
- 移除 staging VGPR 與 `ds_write`。註解指出某設定可「省下 33 個 VGPR」。
- 限制：
  - 每 lane 搬 4 byte（`GlobalReadVectorWidth · bpe = 4`）；
  - `M0` 必須保存 LDS address；
  - 某些 layout 需要 `TransposeLDS=1`。

### 1.3 LDS 與 Local Read

- **`1LDSBuffer`** 只用一個而非兩個 LDS buffer，以 overlap 換 capacity：可用較大 tile 或較高 occupancy。搭配 SIA3 時只能與 PGR 一起使用。
- **`LdsPadA/B`** 與 **`LdsBlockSizePerPad`** 插入 padding 以打散 bank conflict。與 `mfma_gemm.hip` 的 `+8` row padding 相同，只是改用搜尋而非猜測。
- **`PrefetchLocalRead=n`** 讓 register 預存 MFMA 前 *n* 個 iteration 的 `ds_read` 結果，是第 06 章 `a[0:63]` / `a[64:127]` 交換的一般化。

### 1.4 Scheduling：`ScheduleIterAlg`

Generator 先分別產生一個 loop iteration 的四條 instruction stream：
1. global read 與 pointer increment；
2. local write；
3. local read；
4. MFMA。

`KernelWriter.makeSchedule` 再請 `SIA` component 合併：

| SIA | 策略 |
|-----|----------|
| 0 | 不 interleave：global read、local read、local write，最後全部 MAC |
| 1 / 2 | 依 local-read iteration interleave 的舊 heuristic |
| **3** | **以 MFMA 為中心：** 依可控制密度將 memory instruction 放在 MFMA 間 |

SIA=3 時，`GlobalReadPerMfma` 與 `LocalWritePerMfma`（0.01–32）控制密度。`0.1` 表示每 10 個 MFMA 一次 global read。

將 global read 聚在一起能提高 memory efficiency，但全滿的 vector-memory FIFO 會阻擋**所有** issue，包括 MFMA，因此密度需要調校。結果與手寫 AITER loop 相同：MFMA、一兩個 load、MFMA，依此類推。

Generator 也會自行計算每個 `s_waitcnt`。它知道每個 producer 與 consumer 間放了多少 load，並以硬體 `MaxVmcnt` 為上限，因此能發出最緊但安全的 count。

### 1.5 工作切分：GSU 與 Stream-K

假設 `M·N / (MT0·MT1)` 個 output tile 遠少於 304 個 CU，例如 decode 時 M=128。可用兩種方法使用 idle CU。

**GlobalSplitU（GSU）。** 將 K 切成 GSU slice。Partial result 有三種合併方式：

| `GlobalSplitUAlgorithm` | 合併方式 |
|-------------------------|---------------------------|
| `SingleBuffer` | Atomic accumulate 到單一 buffer，如 AITER `global_atomic_add_f32` |
| `MultipleBuffer` | 每個 slice 寫自己的 buffer；第二個 kernel 做 reduction |
| `MultipleBufferSingleKernel` | 分開 buffer，但最後抵達的 workgroup 在同一 kernel 中用 synchronizer/semaphore 做 reduction，如 AITER |

`GSU=-1` 讓 runtime 選擇。

**Stream-K**（[Osama et al., 2023](https://arxiv.org/abs/2301.03598)）：
- 約每 CU launch 一個 workgroup。
- 將所有 tile 的 **MAC-loop iteration 總數平均分配**給每個 workgroup；workgroup 可能完成一個 tile，再從下一個 tile 中間開始。
- 共享 tile 的 partial 可透過 workspace（deterministic）或 atomic fix up。

其平衡效果為：

$$
L = T\left\lceil \frac{K}{\text{DepthU}} \right\rceil, \qquad
L_g \in \left\{ \left\lfloor \frac{L}{G} \right\rfloor,\ \left\lceil \frac{L}{G} \right\rceil \right\}, \qquad
\eta_{\text{SK}} = \frac{L}{G\,\lceil L/G \rceil}
\quad\text{vs.}\quad
\eta_{\text{tile}} = \frac{T}{G\,\lceil T/G \rceil}
$$

| 符號 | 意義 |
|---|---|
| $T$ | Output（macro）tile 數 |
| DepthU | 每 main-loop iteration 的 K |
| $L$ | 整個 GEMM 的 MAC-loop iteration 總數 |
| $G$ | Stream-K workgroup 數（約為 CU 數） |
| $L_g$ | 分配給 workgroup $g$ 的 iteration |
| $\eta_{\text{SK}}, \eta_{\text{tile}}$ | Stream-K 與 one-workgroup-per-tile 的 fill efficiency |

因為 $L \gg G$，$\eta_{\text{SK}}$ 幾乎是 1；$T$ 略高於 $G$ 的倍數時，$\eta_{\text{tile}}$ 可低至約 50%。代價是修復兩個 workgroup 共享的 tile。

這消除了「最後一 wave 只有 10% 滿」的量化問題，也讓一個 kernel 能良好涵蓋多種 shape，縮小 library。hipBLASLt 透過環境變數提供：

```bash
export TENSILE_SOLUTION_SELECTION_METHOD=2   # 0 = standard tuned library (default), 2 = Stream-K library
export TENSILE_STREAMK_DYNAMIC_GRID=3        # 0 = all CUs; 3 = analytical model picks the grid (default)
export TENSILE_STREAMK_FIXED_GRID=64         # force 64 workgroups (leave CUs for concurrent kernels)
export TENSILE_STREAMK_MAX_CUS=128           # cap CUs used
```

優先順序為 `FIXED_GRID > DYNAMIC_GRID > MAX_CUS > GRID_MULTIPLIER`。

### 1.6 Cache-Aware Tile Order：WGM、WGMXCC、StaggerU

- **`WorkGroupMapping`（WGM）** 重新排列 workgroup ID，使同時 in flight 的 tile 在 C 中形成高度 WGM 的 box。Box 中 tile 會在 L2 共用 A-row 與 B-column panel。公式是 `wgSerial = wg0 + (wg1 % WGM) · nwg0`。
- **`WorkGroupMappingXCC`（WGMXCC）** 抵銷 MI300 將 workgroup *i* round-robin 放到 XCD *i % 8* 的行為。它 remap ID，讓**連續 logical tile 在同一 XCD 執行**，共享其 4 MiB L2。`WorkGroupMappingXCCGroup` 設定 group size，`-1` 代表「CU count」。
- **`StaggerU`** / `StaggerUStride` / `StaggerUMapping` 讓每個 workgroup 從不同 K offset 開始，循環走過 K。
  - K 是很大的二次方時很重要：否則每個 tile 都從同一 DRAM channel 開始。
  - `StaggerUMapping` 選擇由 wg0、wg1、wg2 或 serial ID 驅動 offset。

![預設 round-robin XCD placement，與將相鄰 tile 保持在同一 XCD 的 remapping](figures/ch07-xcd-remap.svg)

WGM 概念與 Triton、CUTLASS 的 grouped launch order 相同。以常見形式表示，serial launch index $s$ 對應到 workgroup 計算的 tile $(w_0', w_1')$：

$$
s = w_0 + w_1\,n_0, \qquad
w_1' = g\left\lfloor \frac{s}{g\,n_0} \right\rfloor + (s \bmod g), \qquad
w_0' = \left\lfloor \frac{s \bmod g\,n_0}{g} \right\rfloor
$$

| 符號 | 意義 |
|---|---|
| $w_0, w_1$ | Launch-order workgroup index |
| $n_0$ | Dimension 0 的 tile 數 |
| $g$ | Box 高度（WGM） |
| $s$ | Serial launch order |
| $w_0', w_1'$ | 實際計算的 tile |

連續 workgroup 先沿 $g$ 個 tile row 向下，再向右移一個 tile column，因此 in-flight workgroup 涵蓋高度 $g$ 的 box，並在 L2 共用 $g$ 個 A row panel 與少數 B column panel。TensileLite 自己的公式細節不同（WGMXCC 還會疊加 XCD remap），但 reuse 理由相同。

WGM、WGMXCC、StaggerU、GSU 都能低成本在 runtime 改變，因為它們打包進 kernel argument，而非編譯寫死：

```
internalArgs  (32 bit): input type | StaggerU (3-bit mapping, 5-bit shift, 8-bit value)
                        | GSU control (GSUC, GSUWGMRR, 14-bit GSU)
internalArgs1 (32 bit): WGMXCCG (10) | WGMXCC (6) | WGM (16, signed)
```

`Components/README.md` 將此 layout 記為 kernel-argument「Version 2」。因此一個 code object 可服務許多 tuned variant。

## 2. 解讀 Kernel 名稱

Kernel 名稱由 `SolutionStructs/Naming.py` 產生：

1. 從 problem type 開始。
2. 加上 `MT<MT0>x<MT1>x<DepthU>` 與 `MI<M>x<N>x<B>`。
3. 對每個必要 parameter，附加其*大寫字母*與 value。

例如：

```
…_MT256x256x64_MI16x16x1_SN_…_DTL1_…_PGR2_PLR1_…_SIA3_…
```

可解讀為：

| Fragment | 意義 |
|----------|---------|
| `MT256x256x64` | Macro tile 256×256，DepthU 64 |
| `MI16x16x1` | 16×16 MFMA，1 block |
| `DTL1` | DirectToLds |
| `PGR2` | PrefetchGlobalRead=2 |
| `PLR1` | PrefetchLocalRead=1 |
| `SIA3` | ScheduleIterAlg=3 |

其他 parameter 也同樣縮寫：
- `1LDSBuffer` → `LDSB`
- `WorkGroupMapping` → `WGM`
- `StaggerU` → `SU`
- `GlobalSplitU` → `GSU`

在 MI300 上 profile PyTorch、看到 `Cijk_…` kernel 時，可依此了解其行為。

## 3. hipBLASLt 如何在 Runtime 選擇

1. **Library logic file。** Benchmark 會產生每 architecture、data type 一份 YAML，列出 solution 及其勝出的 problem size。
2. **Heuristic。** `hipblasLtMatmulAlgoGetHeuristic` 從「standard grid」（確切 tuned size）、「free-size」library 或 Stream-K library（見 `TENSILE_SOLUTION_SELECTION_METHOD`）回傳 problem 的 ranked list。
3. **Solution index。** 每個 solution 在單一 library build 中有穩定 index。可在 `hipblaslt-bench` 用 `--algo_method index` 明確指定，或透過 extension API。

## 4. 為自己的 Shape 調校 hipBLASLt

以下是 `docs/how-to/how-to-use-hipblaslt-offline-tuning.rst` 的 offline tuning 流程：

```bash
# 1. Log the GEMMs your application issues, as ready-to-run bench commands
export HIPBLASLT_LOG_MASK=32
python my_model.py 2> gemms.log        # prints: hipblaslt-bench --api_method c -m … -n … -k … --algo_method index --solution_index …

# 2. Tune: benchmark every applicable solution, record the winner
export HIPBLASLT_TUNING_FILE=tuning.txt
hipblaslt-bench <one logged line>       # repeat for each unique line; iters/cold_iters default to 1000

# 3. Use: override default selection with the tuned winners
unset HIPBLASLT_TUNING_FILE
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
python my_model.py
```

![Offline tuning 的三個步驟](figures/ch07-tuning-flow.svg)

兩項警告：
- Solution index 只對**相同 library build 與相同 architecture**有效。升級 ROCm 後要重新調校。
- `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` 將候選 solution 限制在 application 實際提供的 workspace 內，對 GSU、Stream-K 很重要。

Framework-level 替代方式：
- **PyTorch TunableOp：** `PYTORCH_TUNABLEOP_ENABLED=1` 在 runtime 對每個 shape 嘗試 hipBLASLt 與 rocBLAS candidate，並將結果快取於 CSV。
- **AITER `gemm_a16w16_tune.py --with-hipblaslt`：** 讓 hipBLASLt solution 與 asm、triton 等 backend 競爭（第 06 章）。勝出的 `solidx` 以 `libtype=hipblaslt` 寫入 `bf16_tuned_gemm.csv`。

## 5. 用 TensileLite 產生自己的 Kernel

TensileLite 也可直接執行列出 fork parameter 與 problem size 的 YAML config。範例位於 `tensilelite/HostLibraryTests/configs/`（例如 `mixed_configs/aquavanjaram_*.yaml`；「aquavanjaram」即 gfx942）。格式會隨 release 改變，請從自己的 checkout 中現有 config 開始。TensileLite 接著：

1. 列舉所有有效組合：`SolutionStructs/Validators` 做 validity check，`TensileLogic` program 檢查 `MatrixInstruction`。
2. 產生並組譯每個 kernel。
3. 在 GPU 上 benchmark。
4. 寫出 hipBLASLt 能載入的 library logic。

這是 AMD 方法中「以搜尋手工打造」的一半；AITER `.co` kernel 則是「親手打造」的一半。兩者最終都有第 06 章所追蹤的 loop structure。

## 6. 摘要：AMD GEMM Playbook

| 技術 | AITER asm（第 06 章） | TensileLite parameter |
|-----------|-------------------|-----------------------|
| 大型 per-wave tile、每 SIMD 1 wave | 每 wave 16×128，512 register | `MatrixInstruction` WaveTile、`MaxOccupancy` |
| Direct-to-LDS | `buffer_load_dword … lds` | `DirectToLds` |
| 預先排列 weight | B preshuffle 到 AGPR | gfx94x 上供 A 使用的 `HIPBLASLT_ORDER_COL16_4R8`（bf16/fp16）/ `COL16_4R16`（fp8）matrix order |
| Multi-stage prefetch | `vmcnt(18)`、LDS ping-pong | `PrefetchGlobalRead`、`1LDSBuffer` |
| Register double buffering | `a[0:63]` ↔ `a[64:127]` | `PrefetchLocalRead` |
| MFMA/memory interleaving | 手工 | `ScheduleIterAlg=3`、`GlobalReadPerMfma` |
| Split-K / Stream-K | z-grid + atomic + semaphore | `GlobalSplitU*`、`StreamK` |
| L2/XCD-aware tile order | – | `WorkGroupMapping`、`WorkGroupMappingXCC`、`StaggerU` |
| Per-shape selection | Tuned CSV + heuristic | Library logic + heuristic + offline tuning |

## 重點整理

1. hipBLASLt kernel 是 parameter space 中的一點；generator 將 parameter 轉成 assembly，benchmark 決定發布哪些點。
2. `MatrixInstruction` 編碼整個 tile hierarchy（MFMA → wave tile → macro tile）；`DepthU` 是 K step。Wave tile 越大，reuse 與 accumulator register 越多。
3. PGR、DirectToLds、PLR、LDS padding 與 `ScheduleIterAlg=3`，分別是 prefetch、direct-to-LDS load、register double buffering、避免 bank conflict 與 MFMA interleaving 的 generated 版本。
4. GSU（split-K）與 Stream-K 修正 tile quantization；WGM、WGMXCC、StaggerU 讓 tile order 對 cache 與 channel 友善，且都是低成本 runtime argument。
5. Selection 是 lookup 加 heuristic；offline tuning 可針對確切 shape override，但只適用一個 library build 與 architecture。

## 練習

1. 取 4096×4096×4096 bf16 GEMM 的 `hipblaslt-bench` command：
   - 以 `--algo_method heuristic --requested_solution 10 --print_kernel_info` 執行。
   - 用第 2 節解讀前三個 kernel 名稱。
   - 哪些 parameter 不同？

    <details markdown="1"><summary>提示</summary>

    將名稱依 fragment 對齊（`MT…`、`MI…`，再來是大寫字母縮寫）。大型 square GEMM 的 candidate 通常使用相同 MFMA，而在 macro tile、`DepthU`、`PGR`/`PLR`、`WGM` 或 GSU 上不同。

    </details>

2. 在 M ∈ {1, 16, 128, 1000}、N = K = 8192 時比較 `TENSILE_SOLUTION_SELECTION_METHOD=0` 與 `=2`。用第 1.5 節的 wave-quantization 理由解釋差異。

    <details markdown="1"><summary>提示</summary>

    $M = 1$ 或 16 時，$T = \lceil N/\text{MT}_1 \rceil$ 遠少於 304 個 CU，因此 standard library 必須依賴 GSU，而 Stream-K 會將少數 tile 的 $K$ loop 分散到所有 CU。$M = 1000$ 時 tile 數較多，差異縮小。

    </details>

3. 在 MI300X 上，以只在 `WorkGroupMappingXCC` 為 1 或 8 方面不同的 solution（若能找到），執行相同 GEMM。用 `rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum` 測量 L2 hit rate。
