# 07 – hipBLASLt 與 TensileLite：由程式產生的 GEMM kernel

> **第五部 · AMD 架構與函式庫** · 先備知識：[05](05-amd-cdna3-mfma.md)、[06](06-aiter-asm-gemm.md) ·
> 下一章：[15 – SGLang 中的 Triton：服務 Kimi K3](15-triton-model-systems.md)

AITER（第 06 章）手寫了幾十個 GEMM kernel，而 [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/) 發布的卻是**數以千計**：ROCm 的 `libhipblaslt` 為每種 GPU 架構各附一組程式碼物件，PyTorch 在 MI300 上預設就用它來執行 `torch.matmul`。這些 kernel 沒有一個是手寫的，它們都來自 **TensileLite**——一個 Python 程式：它接收一份參數清單，為每一種參數組合產生完整的組合語言 kernel，再逐一做基準測試，決定哪些要隨函式庫發布。

**你將學到**

- 什麼是 TensileLite 的 *solution*，以及主要參數的意義：分塊階層（`MatrixInstruction`、`DepthU`）、全域與區域讀取、LDS 配置、指令排程、工作切分與分塊順序；
- 每個參數對應到第 05–06 章中手工做過的哪一項技巧（NVIDIA 方面則對應矩陣乘法 2–8 各頁）；
- 如何從效能分析結果中解讀 `Cijk_…` 這類 kernel 名稱；
- hipBLASLt 在執行期如何選擇 kernel，以及如何針對自己的矩陣形狀調校這個選擇；
- TensileLite 如何產生、測試並發布 kernel。

> **程式碼在哪裡。** hipBLASLt 原本是獨立的儲存庫 `ROCm/hipBLASLt`，現已封存到 `develop_deprecated` 分支；後續開發在 [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) 單一儲存庫的 `projects/hipblaslt` 下進行。產生器位於 `tensilelite/Tensile/`：
> - `KernelWriterAssembly.py`、`KernelWriter.py`：產生器本體。
> - `Components/`：可抽換的元件，例如 `SIA.py`、`StreamK.py`、`GSU.py`、`LocalRead.py` 與 `MAC_*.py`。
> - `Common/ValidParameters.py`：所有參數及其註解。
> - `SolutionStructs/`：合法性檢查與命名。
>
> 以下提到的檔案都以這些路徑為準。

## 1. 從「一個 GEMM」到「一個 solution」

一個 TensileLite **solution** 是龐大設計空間中的一個點：

| 類別 | 參數 | 在第 05/06 章中的對應 |
|-------|------------|--------------------------|
| 分塊形狀 | `MatrixInstruction`、`DepthU` | MFMA 形狀、wave 分塊、每個 workgroup 的 wave 數、每步 K |
| 全域記憶體 → LDS | `PrefetchGlobalRead`（PGR）、`DirectToLds`（DTL）、`GlobalReadVectorWidth`、`BufferLoad` | 預先載入深度、直接載入 LDS |
| LDS | `1LDSBuffer`、`LdsPadA/B`、`LdsBlockSizePerPad`、`TransposeLDS` | 雙緩衝、避免 bank 衝突的填補 |
| LDS → 暫存器 | `PrefetchLocalRead`（PLR）、`ClusterLocalRead` | 暫存器雙緩衝（`a[0:63]` / `a[64:127]`） |
| 排程 | `ScheduleIterAlg`（SIA）、`GlobalReadPerMfma`、`LocalWritePerMfma` | 在 MFMA 之間穿插載入 |
| 工作切分 | `GlobalSplitU`（GSU）、`GlobalSplitUAlgorithm`、`StreamK` | Split-K、Stream-K |
| 快取行為 | `WorkGroupMapping`（WGM）、`WorkGroupMappingXCC`（WGMXCC）、`StaggerU*` | 分塊順序、XCD 分派、分散 DRAM 通道 |
| Epilogue | `StoreRemapVectorWidth`、`StoreVectorWidth`、激活 / 偏差 / 縮放的融合 | 合併存取的寫出 |

### 1.1 `MatrixInstruction`：用 9 個數字描述分塊階層

#### 解讀這 9 個數字

`ValidParameters.py` 中的註解說明了這 9 個數字的格式：

```
[32, 32, 1, 2,   1,   4, 1,   2, 2]
 ^^^^^^^^^^^^    ^    ^^^^    ^^^^
 MFMA MxNxKxB  BlkM  WaveTile  Waves
```

- **MFMA** `32x32x1x2` 是一次計算 2 個區塊的 MFMA 變體。`MIBlockM = 1` 時，每道指令涵蓋 32×64。
- **WaveTile** `4×1`：每個 wave 發出 4×1 道這樣的指令，涵蓋 128×64。
- **Waves** `2×2`：每個 workgroup 有四個 wave，因此**巨分塊**（macro tile）為 (32·4·2) × (64·1·2) = **256×128**。

一般而言，把 9 個數字寫成 $[m, n, k, b,\ \beta_M,\ w_M, w_N,\ W_M, W_N]$：

$$
\text{MT}_0 = m\,\beta_M\,w_M\,W_M, \qquad
\text{MT}_1 = n\,\frac{b}{\beta_M}\,w_N\,W_N, \qquad
\text{threads} = 64\,W_M W_N
$$

| 符號 | 意義 |
|---|---|
| $m, n, k$ | MFMA 形狀（例如 32、32、1） |
| $b$ | MFMA 一次計算的區塊數（多區塊變體），大多數為 1 |
| $\beta_M$ | `MIBlockM`：這 $b$ 個區塊中沿 M 方向堆疊的數量（其餘沿 N 方向） |
| $w_M, w_N$ | WaveTile：每個 wave 沿 M 與 N 方向的 MFMA 分塊數 |
| $W_M, W_N$ | 每個 workgroup 沿 M 與 N 方向的 wave 數 |
| $\text{MT}_0, \text{MT}_1$ | 沿 M 與 N 方向的巨分塊（workgroup 分塊）大小 |

以上例而言：$\text{MT}_0 = 32\cdot1\cdot4\cdot2 = 256$，$\text{MT}_1 = 32\cdot2\cdot1\cdot2 = 128$，共 256 個執行緒。

![MatrixInstruction [32, 32, 1, 2, 1, 4, 1, 2, 2]：MFMA 分塊、wave 分塊與巨分塊](figures/ch07-macro-tile.svg)

gfx942 的 bf16 kernel 大多使用 `16x16x16` 或 `32x32x8` 的 MFMA，寫成 `[16,16,16,1, 1, …]`。

#### 為什麼較大的 WaveTile 有幫助

選擇較大的 WaveTile，是 TensileLite 提高「每位元組 MFMA 數」的方法——這正是第 05 章教學 kernel 所缺乏的。代價是累加器暫存器：
- fp32 的 128×64 wave 分塊有 8192 個值，分給 64 個 lane，就是每個 lane 128 個 AGPR。
- 這就是快速 kernel 每個 SIMD 只跑一到兩個 wave 的原因。

累加器的成本與重用率，都可以由 wave 分塊 $T_M\times T_N$（MFMA 分塊乘上 WaveTile）推得：

$$
r_{\text{acc}} = \frac{T_M T_N}{64}, \qquad
\frac{\text{MFMAs}}{\text{operand fetches}} = \frac{w_M w_N}{w_M + w_N}
$$

| 符號 | 意義 |
|---|---|
| $T_M, T_N$ | 每個 wave 沿 M 與 N 方向的輸出元素數 |
| $r_{\text{acc}}$ | 每個 lane 的 fp32 累加器暫存器數 |
| $w_M, w_N$ | 每個 wave 的 MFMA 分塊數；每個 A 片段重用 $w_N$ 次，每個 B 片段重用 $w_M$ 次 |

由 16x16 MFMA 組成的 $4\times4$ WaveTile（$T_M = T_N = 64$）需要每個 lane 64 個累加器暫存器，且每個運算元片段重用 4 次；$128\times64$ 則需要 128 個。

#### DepthU

**`DepthU`** 是主迴圈一次迭代的 K 長度：在 AITER 的 kernel 中為 64，16 位元型別通常為 32–128。

### 1.2 全域讀取：PGR、DirectToLds

`ValidParameters.py` 對 `PrefetchGlobalRead` 的描述：

- **PGR=0：** 不預先載入。載入、等待、寫入 LDS、計算。
- **PGR=1：** 對 *全域記憶體 → VGPR → LDS* 這條路徑做雙緩衝。需要兩倍的 LDS，以及中轉用的 VGPR。
- **PGR=2：** 在中轉資料寫入 LDS 的同時，*再*發出一次全域預先載入，讓兩個分塊同時在傳輸中。

這正是 AITER 的 `pf3` kernel 寫死的做法，也就是第 06 章中 `vmcnt(N)` 所表達的內容。

`DirectToLds=1` 使用 `buffer_load … lds`（第 06 章第 4 節）：
- 它省去中轉用的 VGPR 與 `ds_write`；註解指出在某個設定下「可省下 33 個 VGPR」。
- 限制條件：
  - 每個 lane 只搬 4 位元組（`GlobalReadVectorWidth · bpe = 4`）；
  - `M0` 必須存放 LDS 位址；
  - 某些資料配置需要 `TransposeLDS=1`。

### 1.3 LDS 與區域讀取

- **`1LDSBuffer`** 選擇只用一個 LDS 緩衝區而不是兩個，以犧牲重疊換取容量：可用更大的分塊，或更高的佔用率。搭配 SIA3 時只能與 PGR 一起使用。
- **`LdsPadA/B`** 與 **`LdsBlockSizePerPad`** 插入填補以避免 bank 衝突。概念與 `mfma_gemm.hip` 中每列 `+8` 的填補相同，只是改用搜尋而不是猜測。
- **`PrefetchLocalRead=n`** 讓暫存器中保有提前 *n* 次迭代的 `ds_read` 結果。這就是第 06 章 `a[0:63]` / `a[64:127]` 交替使用的一般化。

### 1.4 排程：`ScheduleIterAlg`

產生器會先把一次迴圈迭代的四條指令流分別產生出來：
1. 全域讀取及其指標遞增；
2. 區域寫入；
3. 區域讀取；
4. MFMA。

接著 `KernelWriter.makeSchedule` 詢問 `SIA` 元件要如何合併它們：

| SIA | 策略 |
|-----|----------|
| 0 | 不穿插：先全域讀取、區域讀取、區域寫入，最後才是所有 MAC |
| 1 / 2 | 較舊的啟發式規則，以每次區域讀取迭代為單位穿插 |
| **3** | **以 MFMA 為中心：** 以受控的密度，把記憶體指令放在 MFMA *之間* |

在 SIA=3 下，`GlobalReadPerMfma` 與 `LocalWritePerMfma`（0.01–32）控制這個密度，`0.1` 表示每 10 道 MFMA 穿插一次全域讀取。

把全域讀取接連發出能提高記憶體效率，但向量記憶體的 FIFO 一旦滿了，就會擋住*所有*指令的發出（包括 MFMA），所以密度必須調校。最後產生的程式碼，形狀與手寫的 AITER 迴圈相同：一道 MFMA、一兩道載入、一道 MFMA，依此類推。

產生器也會自行計算每一個 `s_waitcnt`。它知道在每個生產者與消費者之間放了多少道載入（並以硬體的 `MaxVmcnt` 為上限），因此能產生最緊但仍安全的計數。

### 1.5 工作切分：GSU 與 Stream-K { #15-work-decomposition-gsu-and-stream-k }

假設輸出分塊數 `M·N / (MT0·MT1)` 遠少於 304 個 CU，例如解碼時 M = 128。要利用閒置的 CU，有兩種方法。

#### GlobalSplitU（GSU）

把 K 切成 GSU 份，部分結果以下列三種方式之一合併：

| `GlobalSplitUAlgorithm` | 部分結果如何合併 |
|-------------------------|---------------------------|
| `SingleBuffer` | 以原子操作累加到同一個緩衝區，類似 AITER 的 `global_atomic_add_f32` |
| `MultipleBuffer` | 每份各自寫入自己的緩衝區，再由第二個 kernel 歸約 |
| `MultipleBufferSingleKernel` | 使用各自的緩衝區，但由最後抵達的 workgroup 在同一個 kernel 中歸約，用的是類似 AITER 的同步器 / 號誌 |

`GSU=-1` 讓執行期自行決定。

#### Stream-K

Stream-K（[Osama et al., 2023](https://arxiv.org/abs/2301.03598)）的做法如下：
- 大約每個 CU 啟動一個 workgroup。
- 把所有分塊的 MAC 迴圈迭代*平均分給*每個 workgroup，因此一個 workgroup 可能做完一個分塊之後，從下一個分塊的中間開始。
- 部分分塊的修正可以透過工作區（結果具決定性），或使用原子操作。

Stream-K 達到的負載平衡，可以寫成：

$$
L = T\left\lceil \frac{K}{\text{DepthU}} \right\rceil, \qquad
L_g \in \left\{ \left\lfloor \frac{L}{G} \right\rfloor,\ \left\lceil \frac{L}{G} \right\rceil \right\}, \qquad
\eta_{\text{SK}} = \frac{L}{G\,\lceil L/G \rceil}
\quad\text{vs.}\quad
\eta_{\text{tile}} = \frac{T}{G\,\lceil T/G \rceil}
$$

| 符號 | 意義 |
|---|---|
| $T$ | 輸出（巨）分塊數 |
| DepthU | 主迴圈每次迭代的 K 長度 |
| $L$ | 整個 GEMM 的 MAC 迴圈迭代總數 |
| $G$ | Stream-K 的 workgroup 數（約等於 CU 數） |
| $L_g$ | 分配給 workgroup $g$ 的迭代數 |
| $\eta_{\text{SK}}, \eta_{\text{tile}}$ | Stream-K 與「一個分塊一個 workgroup」的填滿效率 |

由於 $L \gg G$，$\eta_{\text{SK}}$ 幾乎等於 1；而當 $T$ 稍大於 $G$ 的倍數時，$\eta_{\text{tile}}$ 可能低至約 50%。代價是必須修正由兩個 workgroup 共同負責的分塊。

#### 在 hipBLASLt 中使用 Stream-K

這消除了「最後一輪只填滿 10%」的量化問題；而且一個 kernel 就能良好涵蓋許多形狀，也讓函式庫變小。hipBLASLt 透過環境變數提供這項功能：

```bash
export TENSILE_SOLUTION_SELECTION_METHOD=2   # 0 = standard tuned library (default), 2 = Stream-K library
export TENSILE_STREAMK_DYNAMIC_GRID=3        # 0 = all CUs; 3 = analytical model picks the grid (default)
export TENSILE_STREAMK_FIXED_GRID=64         # force 64 workgroups (leave CUs for concurrent kernels)
export TENSILE_STREAMK_MAX_CUS=128           # cap CUs used
```

優先順序為 `FIXED_GRID > DYNAMIC_GRID > MAX_CUS > GRID_MULTIPLIER`。

### 1.6 考慮快取的分塊順序：WGM、WGMXCC、StaggerU { #16-cache-aware-tile-order-wgm-wgmxcc-staggeru }

#### 三個調整參數

- **`WorkGroupMapping`（WGM）** 重新排列 workgroup 編號，讓同時執行的分塊在 C 中形成高度為 WGM 的矩形區域。同一區域內的分塊在 L2 中共用 A 的列面板與 B 的欄面板。公式為 `wgSerial = wg0 + (wg1 % WGM) · nwg0`。
- **`WorkGroupMappingXCC`（WGMXCC）** 抵銷 MI300 把第 *i* 個 workgroup 輪流分派到第 *i % 8* 個 XCD 的效應。它重新對應編號，讓*邏輯上連續的分塊在同一個 XCD 上執行*，共用該 XCD 的 4 MiB L2。`WorkGroupMappingXCCGroup` 設定分組大小，`-1` 表示「等於 CU 數」。
- **`StaggerU`** / `StaggerUStride` / `StaggerUMapping` 讓每個 workgroup 的 K 迴圈從不同的位移開始，沿 K 輪轉。
  - 當 K 是很大的 2 的冪次時，這一點很重要：否則每個分塊都會從同一個 DRAM 通道開始讀取。
  - `StaggerUMapping` 選擇由哪個 workgroup 索引決定位移：wg0、wg1、wg2 或序號。

![預設的輪流 XCD 分派，與讓相鄰分塊留在同一個 XCD 的重新對應](figures/ch07-xcd-remap.svg)

#### 分組的啟動順序

WGM 背後的概念，與 Triton 和 CUTLASS 使用的「分組」（grouped）啟動順序相同。以那種常見寫法表示，啟動序號 $s$ 會對應到 workgroup 實際計算的分塊 $(w_0', w_1')$：

$$
s = w_0 + w_1\,n_0, \qquad
w_1' = g\left\lfloor \frac{s}{g\,n_0} \right\rfloor + (s \bmod g), \qquad
w_0' = \left\lfloor \frac{s \bmod g\,n_0}{g} \right\rfloor
$$

| 符號 | 意義 |
|---|---|
| $w_0, w_1$ | 依啟動順序的 workgroup 索引 |
| $n_0$ | 沿維度 0 的分塊數 |
| $g$ | 矩形區域的高度（WGM） |
| $s$ | 啟動序號 |
| $w_0', w_1'$ | 實際計算的分塊 |

連續的 workgroup 會先沿著 $g$ 個分塊列往下走，再往右移一個分塊欄，因此同時執行的 workgroup 涵蓋一塊高度為 $g$ 的區域，在 L2 中共用 A 的 $g$ 個列面板與 B 的少數幾個欄面板。TensileLite 自己的公式在細節上不同（WGMXCC 還會在其上加一層 XCD 重新對應），但重用的道理相同。

#### 執行期的 kernel 參數

WGM、WGMXCC、StaggerU 與 GSU 都能在執行期低成本地改變：它們被打包進 kernel 參數，而不是編譯進程式碼：

```
internalArgs  (32 bit): input type | StaggerU (3-bit mapping, 5-bit shift, 8-bit value)
                        | GSU control (GSUC, GSUWGMRR, 14-bit GSU)
internalArgs1 (32 bit): WGMXCCG (10) | WGMXCC (6) | WGM (16, signed)
```

這個配置在 `Components/README.md` 中記載為 kernel 參數的「Version 2」。因此，一個程式碼物件就能服務許多調校過的變體。

## 2. 解讀 kernel 名稱

kernel 名稱由 `SolutionStructs/Naming.py` 產生：

1. 以問題型別開頭。
2. 加上 `MT<MT0>x<MT1>x<DepthU>` 與 `MI<M>x<N>x<B>`。
3. 對每個必要參數，附加其名稱中的*大寫字母*，再接上它的值。

所以像這樣的名稱片段

```
…_MT256x256x64_MI16x16x1_SN_…_DTL1_…_PGR2_PLR1_…_SIA3_…
```

可以解讀為：

| 片段 | 意義 |
|----------|---------|
| `MT256x256x64` | 巨分塊 256×256，DepthU 64 |
| `MI16x16x1` | 16×16 的 MFMA，1 個區塊 |
| `DTL1` | DirectToLds |
| `PGR2` | PrefetchGlobalRead=2 |
| `PLR1` | PrefetchLocalRead=1 |
| `SIA3` | ScheduleIterAlg=3 |

其他參數也以同樣的方式縮寫：
- `1LDSBuffer` → `LDSB`
- `WorkGroupMapping` → `WGM`
- `StaggerU` → `SU`
- `GlobalSplitU` → `GSU`

在 MI300 上分析 PyTorch 的效能時，若看到 `Cijk_…` 這樣的 kernel，就可以用這個方法讀出它做了什麼。

## 3. hipBLASLt 如何在執行期做選擇

1. **函式庫邏輯檔。** 依架構與資料型別各有一份 YAML 檔，由基準測試產生，列出各 solution 以及它們勝出的問題大小。
2. **啟發式規則。** `hipblasLtMatmulAlgoGetHeuristic` 針對問題回傳一份排序過的清單，來源可能是「標準網格」（精確調校過的大小）、「任意大小」函式庫，或 Stream-K 函式庫（見 `TENSILE_SOLUTION_SELECTION_METHOD`）。
3. **Solution 索引。** 每個 solution 在*同一次函式庫建置中*都有固定的索引。可以在 `hipblaslt-bench` 中以 `--algo_method index` 明確指定，或透過擴充 API 指定。

## 4. 針對自己的矩陣形狀調校 hipBLASLt

以下是 `docs/how-to/how-to-use-hipblaslt-offline-tuning.rst` 中的離線調校流程：

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

![離線調校的三個步驟](figures/ch07-tuning-flow.svg)

有兩點需要注意：
- Solution 索引只在**同一次函式庫建置、同一種架構**下有效。升級 ROCm 後要重新調校。
- `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` 會把可選的 solution 限制在應用程式實際提供的工作區大小之內，這對 GSU 與 Stream-K 很重要。

框架層級的替代方案：
- **PyTorch TunableOp：** `PYTORCH_TUNABLEOP_ENABLED=1` 會在執行期針對每種形狀嘗試 hipBLASLt 與 rocBLAS 的候選 kernel，並把結果快取在 CSV 中。
- **AITER 的 `gemm_a16w16_tune.py --with-hipblaslt`** 讓 hipBLASLt 的 solution 與它的組合語言、Triton 等後端一起競賽（第 06 章）。勝出者的 `solidx` 會以 `libtype=hipblaslt` 寫入 `bf16_tuned_gemm.csv`。

## 5. 用 TensileLite 產生自己的 kernel

TensileLite 也可以直接以一份 YAML 設定檔執行，檔中列出要展開的參數與問題大小。範例設定位於 `tensilelite/HostLibraryTests/configs/`（例如 `mixed_configs/aquavanjaram_*.yaml`；「aquavanjaram」就是 gfx942）。格式會隨版本變動，所以請以自己 checkout 中的設定檔為起點。TensileLite 接著會：

1. 列舉每一種合法組合：合法性由 `SolutionStructs/Validators` 檢查，`MatrixInstruction` 則由 `TensileLogic` 程式檢查。
2. 產生並組譯每個 kernel。
3. 在你的 GPU 上逐一做基準測試。
4. 寫出 hipBLASLt 能載入的函式庫邏輯。

這是 AMD 做法中「以搜尋代替手工」的那一半；AITER 的 `.co` kernel 則是「純手工」的那一半。兩者最後得到的，都是第 06 章中追蹤過的同一種迴圈結構。

## 6. 總結：AMD GEMM 的招式表

| 技巧 | AITER 組合語言（第 06 章） | TensileLite 參數 |
|-----------|-------------------|-----------------------|
| 每個 wave 負責大分塊，每個 SIMD 一個 wave | 每個 wave 16×128，512 個暫存器 | `MatrixInstruction` 的 WaveTile、`MaxOccupancy` |
| 直接載入 LDS | `buffer_load_dword … lds` | `DirectToLds` |
| 預先排好的權重 | B 預先重排後直接載入 AGPR | gfx94x 上 A 的矩陣順序 `HIPBLASLT_ORDER_COL16_4R8`（bf16/fp16）/ `COL16_4R16`（fp8） |
| 多階段預先載入 | `vmcnt(18)`、LDS 乒乓緩衝 | `PrefetchGlobalRead`、`1LDSBuffer` |
| 暫存器雙緩衝 | `a[0:63]` ↔ `a[64:127]` | `PrefetchLocalRead` |
| MFMA 與記憶體指令穿插 | 手工 | `ScheduleIterAlg=3`、`GlobalReadPerMfma` |
| Split-K / Stream-K | z 網格 + 原子操作 + 號誌 | `GlobalSplitU*`、`StreamK` |
| 考慮 L2/XCD 的分塊順序 | – | `WorkGroupMapping`、`WorkGroupMappingXCC`、`StaggerU` |
| 依形狀選擇 kernel | 調校表 CSV + 啟發式規則 | 函式庫邏輯 + 啟發式規則 + 離線調校 |

## 重點整理

1. hipBLASLt 的每個 kernel 都是參數空間中的一個點；產生器把參數轉成組合語言，再由基準測試決定哪些點要發布。
2. `MatrixInstruction` 描述整個分塊階層（MFMA → wave 分塊 → 巨分塊），`DepthU` 則是每步 K。wave 分塊越大，重用越多，累加器暫存器也越多。
3. PGR、DirectToLds、PLR、LDS 填補與 `ScheduleIterAlg=3`，分別是預先載入、直接載入 LDS、暫存器雙緩衝、避免 bank 衝突與 MFMA 穿插的自動產生版本。
4. GSU（split-K）與 Stream-K 解決分塊量化問題；WGM、WGMXCC 與 StaggerU 讓分塊順序對快取與記憶體通道友善，而且都是低成本的執行期參數。
5. 選擇 kernel 是查表加上啟發式規則；離線調校可以針對你的確切形狀覆寫這個選擇，但只在同一次函式庫建置與同一種架構上有效。

## 練習

1. 取一個 4096×4096×4096 bf16 GEMM 的 `hipblaslt-bench` 指令：
   - 加上 `--algo_method heuristic --requested_solution 10 --print_kernel_info` 執行。
   - 用第 2 節的方法解讀排名前三的 kernel 名稱。
   - 它們的哪些參數不同？

    <details markdown="1"><summary>提示</summary>

    把名稱逐段對齊（先是 `MT…`、`MI…`，再是各個大寫字母縮寫）。對大型方陣 GEMM，候選 kernel 通常使用相同的 MFMA，差別在巨分塊、`DepthU`、`PGR`/`PLR`、`WGM` 或 GSU。

    </details>
2. 在 N = K = 8192、M ∈ {1, 16, 128, 1000} 的情況下，比較 `TENSILE_SOLUTION_SELECTION_METHOD=0` 與 `=2`，並用第 1.5 節的 wave 量化論點解釋差異。

    <details markdown="1"><summary>提示</summary>

    當 $M = 1$ 或 16 時，$T = \lceil N/\text{MT}_1 \rceil$ 相對於 304 個 CU 非常小，標準函式庫只能依賴 GSU，而 Stream-K 會把這少數幾個分塊的 $K$ 迴圈分散到所有 CU。$M = 1000$ 時分塊數很多，兩者的差距就會縮小。

    </details>
3. 在 MI300X 上，若能找到只有 `WorkGroupMappingXCC` 不同的兩個 solution，分別以 1 和 8 執行同一個 GEMM，並用 `rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum` 量測 L2 命中率。
