# 09 – 分析與效能解析

> **第一部分 · CUDA 基礎** · 先備知識：[00](00-getting-started.md)（屋頂線）、
> [01](01-execution-model.md)、[02](02-memory-hierarchy.md) ·
> 程式：[`examples/09-profile-targets.cu`](examples/09-profile-targets.cu) ·
> 下一章：[10 – Warp 層級原語](10-warp-primitives.md)（第二部分）

第 00–03 章預測 kernel *應該*多快。本章要找出它為何*沒有*那麼快。分析器會把單一數字（kernel 時間）轉成解釋：哪個單元已飽和、哪個單元閒置、warp 在等待什麼，以及哪一行原始碼造成等待。這項技能的重點不在工具本身，而在於依正確順序向工具提出正確問題。

**你將學到**

- 如何正確計算 GPU 程式碼時間，以及應拿時間與什麼比較；
- 分析流程：先看完整程式（Nsight Systems），再看單一 kernel（Nsight Compute），每次只改一項；
- 如何解讀 speed-of-light、記憶體、佔用率和 warp 狀態小節，以及背後的指標名稱；
- 如何將停滯原因對應到修正方法；
- 四組小型 kernel，每組配對的分析結果只會有一個指標不同；
- 正確性工具（`compute-sanitizer`）和 AMD 對應工具（`rocprofv3`、`rocprof-compute`）。

## 1. 分析前先測量

### 1.1 計算 GPU 工作時間

Kernel 啟動是非同步的：`kernel<<<...>>>()` 會在啟動排入佇列後立刻返回。以主機端計時器包住啟動，測到的是啟動開銷而非 kernel。請先暖機一次（第一次啟動還需載入模組和快取），再使用 event 在 GPU 自己的 stream 上計時：

```cpp
cudaEventRecord(start_event);
for (int r = 0; r < reps; ++r) kernel<<<grid, block>>>(args...);
cudaEventRecord(stop_event);
cudaEventSynchronize(stop_event);
cudaEventElapsedTime(&ms, start_event, stop_event);   // total for `reps` launches
```

這就是 [`examples/check.cuh`](examples/check.cuh) 中的 `ex::timeMs`，所有範例程式的 `--bench` 模式都會使用。以下三項規則能讓數字可信：

1. **重複並暖機。** 短 kernel 的單次啟動會受雜訊主導；先暖機一次，再取數十次啟動的平均值。
2. **留意快取。** 在 16 MB 輸入上連續啟動時，資料會留在 L2（A100/H100 有 40–50 MB）；應測量 DRAM 的 benchmark 必須使用比 L2 大數倍的輸入。
3. **固定或回報時脈。** GPU 會隨溫度和功率升降頻。Nsight Compute 預設將時脈鎖定在基準值（`--clock-control base`），所以它回報的時間通常比 benchmark 長。

### 1.2 應與什麼比較

沒有界限的時間毫無意義。對受記憶體限制的 kernel，請換算成有效頻寬；對受計算限制的 kernel，則換算成 FLOP/s：

$$
\beta_{\text{eff}} = \frac{Q}{t}, \qquad F_{\text{eff}} = \frac{W}{t}, \qquad
\eta = \frac{T_{\min}}{t} = \frac{\max(W/F,\ Q/\beta)}{t}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | 往返 DRAM 的必要位元組數（每個輸入讀一次、每個輸出寫一次） |
| $W$ | 浮點運算次數（一次 FMA 計為 2） |
| $t$ | 實測 kernel 時間 |
| $\beta,\ F$ | DRAM 峰值頻寬和峰值計算吞吐量（第 00 章第 7 節） |
| $T_{\min}$ | 時間的屋頂線界限 |
| $\eta$ | 達到屋頂線界限的比例 |

當 $\eta$ 高於約 80 % 時即可停止：分析器只會告訴你已碰到屋頂。若低於此值，分析器才會告訴你原因。

![本章範例 kernel 相對於屋頂線的位置。複製和轉置不做算術，因此以頻寬判斷；遠低於兩種屋頂的點受延遲限制](figures/ch09-roofline.svg)

### 1.3 範例程式

[`examples/09-profile-targets.cu`](examples/09-profile-targets.cu) 包含四組 kernel。同組 kernel 執行相同工作，只有一項特性不同，因此只有一組指標能解釋差異：

| 組別 | Kernel | 特性 | 應查看的位置 |
|---|---|---|---|
| 合併存取 | `copyCoalesced`、`copyStrided` | warp 載入的位址 | 每次要求的 sector 數、DRAM 位元組數 |
| 轉置 | `transposeNaive`、`transposeShared`、`transposePadded` | 儲存模式；共享記憶體 bank | 每次要求的 sector 數、bank 衝突、MIO 停滯 |
| 分歧 | `scaleDivergent`、`scaleUniform` | 分支是否拆分 warp | 每條指令的有效執行緒數 |
| 計算 | `polynomial` | 每元素 256 次相依 FMA | FMA pipeline 使用率、「wait」停滯、佔用率 |

`./profile_targets` 會執行正確性檢查（CI 會在 CPU 模擬器上執行這個命令）；`./profile_targets --bench` 接著會在 256 MB 陣列上為每個 kernel 計時，這才是要分析的執行方式。

## 2. 工具箱

| 工具 | 回答的問題 | 常用命令 |
|---|---|---|
| Nsight Systems（`nsys`） | 程式時間花在哪裡？CPU、API 呼叫、複製、kernel、空檔 | `nsys profile --trace=cuda,nvtx ./app` |
| Nsight Compute（`ncu`） | 為何這個 kernel 只有這樣的速度？ | `ncu --set full -k regex:name -o report ./app` |
| NVTX | 在時間軸上命名程式區段 | `nvtxRangePushA("name")` / `nvtxRangePop()` |
| `compute-sanitizer` | Kernel 是否正確？越界、競爭、未初始化讀取 | `compute-sanitizer --tool racecheck ./app` |
| `rocprofv3`（AMD） | ROCm 上的追蹤和計數器 | `rocprofv3 --kernel-trace --stats -- ./app` |
| `rocprof-compute`（AMD） | Instinct GPU 上的逐 kernel 分析和屋頂線 | `rocprof-compute profile -n run -- ./app` |

兩個建置旗標很重要：

- **`-lineinfo`** 會記錄每條指令的原始碼行，使 Nsight Compute 能把停滯歸因到 `.cu` 檔案中的特定行。它不會改變產生的程式碼（`-G` 則會停用最佳化，絕不可用於效能工作）。
- 範例程式中的 **`-DUSE_NVTX`** 會啟用 NVTX 範圍（NVTX 3 只有標頭，隨 toolkit 提供）。

## 3. 分析流程

![分析流程：計算程式時間、在時間軸上找出高成本部分、分析單一 kernel、只改一項，再次測量](figures/ch09-workflow.svg)

由上而下分析可避免最常見的時間浪費：最佳化只占執行時間 5 % 的 kernel，卻忽略 GPU 正閒置等待主機。流程有五個步驟：

1. **計時。** 記錄端到端時間和各 kernel 時間；這是比較每次變更的基準。
2. **時間軸。** Nsight Systems 會顯示 GPU 是否有在工作，以及哪些 kernel 占主導。
3. **單一 kernel。** 使用 Nsight Compute 分析占主導的 kernel：它離哪個屋頂有多近？
4. **診斷。** 逐層追蹤限制來源：先判斷記憶體或計算，再看停滯原因，最後找到原始碼行。
5. **只改一項**，再重新測量。一次修改兩項，就無法知道哪一項有幫助（或哪一項造成傷害）。

## 4. Nsight Systems：完整程式

### 4.1 解讀時間軸

```bash
nsys profile --trace=cuda,nvtx -o timeline ./profile_targets --bench
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum timeline.nsys-rep
```

第一個命令會記錄可在 Nsight Systems GUI 中開啟的報告；第二個命令會在終端機印出逐 kernel 和逐複製摘要（總時間、執行次數、平均、最小與最大值），通常這些資訊就足夠。

![時間軸示意圖：CPU 執行緒、CUDA API 呼叫、NVTX 範圍、複製和 kernel 各自一列。設定和第一次複製期間 GPU 閒置](figures/ch09-timeline.svg)

由上而下解讀：

| 時間軸模式 | 意義 | 常見修正 |
|---|---|---|
| CPU 忙碌時 GPU 列有空檔 | 主機是瓶頸 | 將工作移到 GPU，或以 stream 重疊 |
| 許多短 kernel，彼此間有空檔 | 啟動開銷（每次數 µs）占主導 | 融合 kernel；CUDA Graphs |
| 從 pageable 記憶體執行 `cudaMemcpy` | 透過 bounce buffer 暫存，無法重疊 | 釘選記憶體（`cudaMallocHost`）和 `cudaMemcpyAsync` |
| 複製與 kernel 從不重疊 | 所有工作都在同一 stream | 將資料分塊放到多個 stream |
| 長時間 `cudaDeviceSynchronize` 或 `cudaMalloc` | 迴圈中有同步或配置 | 只配置一次；只在需要結果時同步 |
| 一個 kernel 占據 GPU 列大部分 | Kernel 是瓶頸 | 改用 Nsight Compute |

### 4.2 NVTX 範圍

NVTX 範圍會在時間軸上標示程式階段，將「第二次複製後的第三個 kernel」變成「`transposes` 範圍」。範例程式使用 RAII 型別包裝；未使用 `-DUSE_NVTX` 時，它不會編譯出任何內容：

```cpp
struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};

{
    NvtxRange range("transposes");
    // ... launches ...
}
```

`nsys profile --capture-range=nvtx --nvtx-capture=transposes` 只會記錄該範圍，讓長時間執行的報告維持小巧。

## 5. Nsight Compute：單一 Kernel

### 5.1 收集報告

```bash
ncu --set full -k regex:transpose -c 3 -o transpose ./profile_targets --bench
ncu --import transpose.ncu-rep --page details     # or open it in the GUI
```

| 選項 | 效果 |
|---|---|
| `-k regex:NAME` | 只分析名稱相符的 kernel |
| `-c N`、`--launch-skip N` | 略過 N 次後，最多分析 N 次啟動 |
| `--set full` | 收集所有小節（很慢：每個 kernel 需 replay 數十次） |
| `--section NAME` | 只收集部分小節，例如 `SpeedOfLight` |
| `--metrics a,b,c` | 只收集指定指標 |
| `--replay-mode application` | 每個 pass 都重新執行完整程式，而非逐 kernel 儲存並還原記憶體 |
| `--clock-control none` | 不鎖定時脈（時間會更接近 benchmark） |

Nsight Compute 會多次 **replay** 每個 kernel，因為硬體每個 pass 只能計算少數指標。它會在 pass 間儲存並還原 kernel 寫入的記憶體，使結果保持一致，但一次分析可能需要數秒到數分鐘。請只分析幾次啟動，不要分析整個 benchmark。

### 5.2 各小節

| 小節 | 顯示內容 |
|---|---|
| GPU Speed Of Light | 實際計算和記憶體吞吐量占峰值的比例；一句話結論 |
| Memory Workload Analysis | 各層（L1、L2、DRAM）的流量、命中率、每次要求的 sector 數、bank 衝突 |
| Compute Workload Analysis | 各 pipeline（FMA、ALU、tensor、LSU……）的使用率 |
| Launch Statistics | 網格與區塊大小、每執行緒暫存器、每區塊共享記憶體、波數 |
| Occupancy | 理論和實際佔用率，以及限制來源 |
| Scheduler Statistics | 每個排程器每週期可發出與實際發出的 warp 數 |
| Warp State Statistics | 每發出一條指令，warp 在各停滯原因上平均花費的週期數 |
| Source Counters | 逐行和逐指令樣本、分支效率、未合併存取 |

### 5.3 Speed of Light：第一個結論

Speed Of Light 小節會回報兩個數字：**SM 吞吐量**（最忙計算單元占其峰值的比例）和**記憶體吞吐量**（最忙記憶體單元：DRAM、L2 或 L1）。它們可用來分類 kernel：

| SM % | 記憶體 % | 結論 | 下一步 |
|---|---|---|---|
| 低 | 高（> 60 %） | 受記憶體限制 | Memory Workload Analysis：這些位元組是否必要？ |
| 高 | 低 | 受計算限制 | Compute Workload Analysis：哪個 pipeline？是否為正確 pipeline？ |
| 高 | 高 | 平衡良好 | 只有演算法變更能改善 |
| 低 | 低 | 受延遲限制 | 先看佔用率，再看 Warp State：warp 為何無法發出？ |

「記憶體吞吐量」指最忙的*單元*，不一定是 DRAM：90 % L1 吞吐量但 20 % DRAM 吞吐量的 kernel 受共享記憶體或 L1 限制；分塊不佳的 GEMM 通常就是如此。

### 5.4 指標名稱

小節中的每個數字都是具有結構化名稱 `unit__counter.rollup.submetric` 的指標。記住少數幾個，就能用 `--metrics` 精確收集所需資料：

| 指標 | 意義 |
|---|---|
| `gpu__time_duration.sum` | Kernel 時間 |
| `dram__bytes_read.sum`、`dram__bytes_write.sum` | DRAM 流量，可與 $Q$ 比較 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 頻寬占峰值比例 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 吞吐量占峰值比例 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 全域載入要求的 32 位元組 sector |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 全域載入要求（每條 warp 指令一筆） |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | 共享記憶體 bank 衝突造成的額外 wavefront（載入；儲存為 `op_st`） |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 實際佔用率 |
| `smsp__thread_inst_executed_per_inst_executed.ratio` | 每條已執行指令的平均有效執行緒數（32 代表無分歧） |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA pipeline 使用率 |

Rollup `.sum`、`.avg`、`.min`、`.max` 會跨單元（SM、記憶體分割區）彙整；`.pct_of_peak_sustained_elapsed` 等 submetric 會以峰值正規化。不同架構的名稱可能略有不同；`ncu --query-metrics` 會列出你的 GPU 支援的名稱。

有兩個衍生比例值得手動計算：

$$
\text{sectors per request} = \frac{N_{\text{sectors}}}{N_{\text{requests}}},
\qquad
\text{traffic ratio} = \frac{Q_{\text{read}} + Q_{\text{write}}}{Q}
$$

| 符號 | 意義 |
|---|---|
| $N_{\text{sectors}},\ N_{\text{requests}}$ | 上述兩個 `l1tex__t_…_op_ld.sum` 指標 |
| $Q_{\text{read}},\ Q_{\text{write}}$ | `dram__bytes_read.sum` 和 `dram__bytes_write.sum` |
| 每次要求的 sector 數 | warp 讀取 32 個連續 float（128 B）時為 4；每個 lane 命中不同 sector 時最高為 32 |
| 流量比例 | 1 代表只有必要流量；大於 1 表示位元組被抓取多次（合併存取或快取重複使用不佳） |

### 5.5 Warp 狀態：Warp 為何等待

當排程器在某週期沒有可執行的 warp，該週期就浪費了。Warp State 小節會將每個 warp 的週期歸因到一項**停滯原因**。比例最大的原因就是應優先處理的對象。

![三類 kernel 的停滯原因分布（比例為示意）：串流複製等待記憶體；發生 bank 衝突的轉置等待共享記憶體 pipeline；相依 FMA 鏈等待固定指令延遲](figures/ch09-stalls.svg)

| 停滯原因 | Warp 正在等待 | 常見修正 |
|---|---|---|
| Long scoreboard | 全域、區域或 texture 載入（L1TEX） | 增加進行中的位元組（向量載入、更多 warp、ILP）；改善重複使用；檢查暫存器 spill（區域記憶體） |
| Short scoreboard | 共享記憶體載入或特殊函式（MUFU：`exp`、`rsqrt`） | 移除 bank 衝突；減少 MUFU 運算 |
| MIO throttle | 記憶體輸入／輸出佇列已滿（共享記憶體、shuffle、特殊函式） | 使用更少且更寬的共享記憶體存取；移除衝突 |
| LG throttle | 區域／全域佇列已滿 | 使用較寬載入（`float4`）；降低每位元組所需指令數 |
| Wait | 固定延遲相依性（例如約 4 週期的 FMA 結果） | 增加逐執行緒獨立指令（ILP）或 warp 數 |
| Math pipe throttle | 所需 pipeline 忙碌 | 不需修正（pipeline 已飽和），或改用其他 pipeline（tensor core） |
| Barrier | 區塊中的其他 warp 尚未抵達 `__syncthreads()` | 平衡 warp 間工作；減少 barrier |
| Membar | 記憶體 fence | 減少 fence；縮小範圍 |
| Branch resolving | 分支目標 | 減少分支，或使分支更一致 |
| Not selected | 無；它可執行，但另一個 warp 已發出 | 不需修正（代表平行度充足） |
| Selected | 無；它在該週期已發出 | 不需修正 |

停滯原因是症狀，不是病因。達到 90 % DRAM 吞吐量的 kernel 出現「Long scoreboard」，只是受記憶體限制的正常樣貌；若只有 30 %，才代表進行中的位元組太少，可用 Little 定律（第 01 章第 5.2 節）量化。

### 5.6 佔用率與波數

佔用率是 SM 上常駐 warp 數除以最大值（A100 和 H100 為 64）。Occupancy 小節會回報*理論*值（受暫存器、共享記憶體和區塊大小限制）與*實際*值（SM 隨時間平均實際容納的數量）。

$$
\text{warps per SM} = \min\left(
\left\lfloor \frac{R_{\text{SM}}}{r \cdot 32} \right\rfloor,\
\left\lfloor \frac{S_{\text{SM}}}{s} \right\rfloor \cdot w,\
b_{\max} \cdot w,\ 64 \right), \qquad
\text{waves} = \frac{\text{blocks}}{N_{\text{SM}} \cdot \text{blocks per SM}}
$$

| 符號 | 意義 |
|---|---|
| $R_{\text{SM}}$ | 每個 SM 的暫存器數（65 536） |
| $r$ | 每執行緒暫存器數（來自 `ptxas -v` 或 Launch Statistics）；每個 warp 以每執行緒 8 個為單位配置 |
| $S_{\text{SM}},\ s$ | 每個 SM 和每個區塊的共享記憶體 |
| $w$ | 每區塊 warp 數 |
| $b_{\max}$ | 每個 SM 的最大常駐區塊數（近期 GPU 為 32） |
| $N_{\text{SM}}$ | SM 數量（A100 為 108，H100 SXM 為 132） |

暫存器項是簡化公式：暫存器以 warp 為單位分塊配置，因此先將 $r$ 向上取整到 8 的倍數。Runtime 可用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 計算精確值。

高佔用率是手段，不是目標。它透過切換其他 warp 來隱藏延遲；具有足夠 ILP 的 kernel（第 04 章的暫存器 tile）即使只有 25 % 佔用率也能全速執行。以下兩種佔用率問題才重要：

- **實際值遠低於理論值**代表 SM 缺乏工作：區塊太少（網格小於一波），或各區塊完成時間差異很大。
- **尾端效應。** 1.1 波的網格在最後 0.1 波時，幾乎整個 GPU 都是空的。若 kernel 執行時間長，請讓網格波數接近整數，或使用 persistent grid（第 04.6 章）。

### 5.7 Source Counters

使用 `-lineinfo` 時，Source 頁面會並排顯示 CUDA 原始碼和 SASS，並針對每一行顯示：已執行指令、依原因分類的停滯樣本、各記憶體指令每次要求的 sector 數，以及各分支的分歧。它回答流程中的最後一個問題：*是哪一行*。其中兩項檢查永遠值得進行：

- 在 SASS 中搜尋 `LDL`/`STL`（區域記憶體）：這代表暫存器 spill，或以執行時索引存取而無法放入暫存器的陣列；
- 確認預期的寬載入是 `LDG.E.128`（float4），且熱迴圈中的共享載入沒有因 bank 衝突而加倍。

## 6. 案例研究

以下是範例程式的四組 kernel，以及可區分它們的指標。比例由存取模式決定；實際時間則取決於 GPU。

### 6.1 合併存取

`copyCoalesced` 會讓每個 warp 讀取 32 個連續 float：**每次要求 4 個 sector**，DRAM 流量等於 $Q$。`copyStrided` 讀取元素 $33i \bmod n$：warp 的每個 lane 都命中不同 sector，所以一次要求需要 **32 個 sector**，而每個 32 位元組只使用其中 4 個。其他 28 個位元組要到很久之後才由其他 warp 使用，屆時 256 MB 輸入早已把它們逐出 L2，因此 DRAM 讀取流量最多成長 8 倍。

應查看：每次要求的 sector 數為 4 與約 32；`dram__bytes_read.sum` 約為 268 MB 與最高約 2 GB；兩者的記憶體吞吐量都很高（跨步複製確實很*忙*，只是搬了無用的位元組）。第 02 章第 2 節會說明 sector 機制。

### 6.2 轉置

`transposeNaive` 沿列讀取（合併），沿欄寫入：**儲存**要求各需要 32 個 sector。`transposeShared` 會在共享記憶體中暫存 32 × 32 tile，因此兩端的全域存取都會合併；但讀取 tile 的一欄 `tile[threadIdx.x][c]` 時，32 個 lane 全部命中同一 bank，造成 **32 路 bank 衝突**。`transposePadded` 多補一欄，使各列錯開一個 bank，消除衝突。

| Kernel | 每次全域儲存要求的 sector 數 | 共享載入 bank 衝突 | 主要停滯 |
|---|---|---|---|
| `transposeNaive` | ~32 | 無（未使用共享記憶體） | Long scoreboard / LG throttle |
| `transposeShared` | 4 | 每條載入指令多 31 個 wavefront | MIO throttle、short scoreboard |
| `transposePadded` | 4 | 0 | Long scoreboard（複製應有的狀態） |

Padding 版本應達到約等於 `copyCoalesced` 的頻寬；正確評估基準是該複製，而非簡單版本。第 02 章第 4、5 節和第 04.5 章（swizzling）會深入討論 bank。

### 6.3 分歧

`scaleDivergent` 會讓偶數和奇數 lane 進入不同的 32 次迴圈。warp 會依序執行兩條路徑，每次遮罩一半 lane，因此同樣結果需要發出兩倍指令。`scaleUniform` 改以 warp 索引分支：每個 warp 只會以所有 lane 執行一條路徑。

應查看：`smsp__thread_inst_executed_per_inst_executed.ratio` 約為 16 與 32，且已執行指令數約為兩倍。*時間*是否加倍取決於 kernel 是否受發出速率限制：一致版本每個元素有 64 次浮點運算和 8 位元組流量，即 8 flop/B，低於 A100 的轉折點，因此部分分歧成本會隱藏在記憶體時間後。分歧位於受計算限制 kernel 的熱迴圈內時，影響才大。

### 6.4 受計算限制的 Kernel

`polynomial` 使用 Horner 法則計算 256 次多項式：每個元素有 256 次 FMA（512 次浮點運算）和 8 位元組流量，即 64 flop/B，位於轉折點右側很遠。其 SM 吞吐量應該很高，記憶體吞吐量則很低。

每次 FMA 都相依於上一次，所以單一 warp 每條指令會停滯約 4 個週期（「wait」）。此 kernel 仍可達到大部分 FMA 峰值，因為各排程器有許多 warp 可交錯執行：使用 4 個排程器且延遲為 4 週期時，每個排程器有 4 個可執行 warp 即可讓 pipeline 滿載。若強制配置大量共享記憶體，讓佔用率減半，FMA 使用率就會下降；若每個執行緒處理兩個獨立元素（ILP），使用率便會恢復。這仍是 Little 定律，只是套用到 FMA pipeline 而非 DRAM。

## 7. 正確性工具

快速但錯誤的 kernel 不是成果。`compute-sanitizer` 可捕捉一般檢查可能漏掉的錯誤：

| 工具 | 可找出的問題 |
|---|---|
| `--tool memcheck`（預設） | 越界和未對齊存取、無效釋放 |
| `--tool racecheck` | 共享記憶體資料競爭（例如缺少 `__syncthreads()`） |
| `--tool initcheck` | 讀取未初始化的全域記憶體 |
| `--tool synccheck` | 錯誤使用 barrier 和 warp 同步（例如錯誤 mask） |

請嘗試刪除 `transposeTiled` 中的 `__syncthreads()`，再執行 `compute-sanitizer --tool racecheck ./profile_targets`。一般檢查可能仍會通過（競爭取決於時序），但 racecheck 會回報 `tile` 上的先寫後讀風險，以及涉及的兩行原始碼。沒有 GPU 時，CPU 模擬器會讓相同錯誤穩定重現：它會讓每個執行緒執行到下一個 barrier，因此缺少 barrier 時，執行緒會讀到尚未寫入的 tile 項目，使轉置檢查失敗（參閱 [tools/cuemu](../tools/cuemu/README.md)）。

## 8. 在 AMD GPU 上進行分析

ROCm 有相同的兩層工具（第 05 章介紹硬體）：

| NVIDIA | AMD（ROCm） | 說明 |
|---|---|---|
| `nsys` | `rocprofv3 --kernel-trace --memory-copy-trace`、`rocprof-sys` | 追蹤與時間軸（Perfetto UI） |
| `ncu` | `rocprof-compute`（原名 Omniperf） | 每個 kernel 的 speed of light、記憶體圖表、屋頂線 |
| `ncu --metrics` | `rocprofv3 --pmc SQ_WAVES FETCH_SIZE ...` | 原始硬體計數器 |
| NVTX | ROCTx（`roctxRangePush`） | 時間軸標記 |
| `compute-sanitizer` | HIP 的 AddressSanitizer（`-fsanitize=address`） | 記憶體錯誤 |

CDNA 上實用的計數器和衍生指標包括：`FETCH_SIZE` 和 `WRITE_SIZE`（往返記憶體的 KB 數）、`SQ_WAVES`（啟動的 wavefront）、VALU 使用率，以及 `SQ_LDS_BANK_CONFLICT`（LDS bank 衝突，對應共享記憶體衝突）。`rocprofv3 --stats` 會像 `nsys stats` 一樣印出逐 kernel 總計。

## 9. 檢查清單

1. GPU 有在工作嗎？（時間軸）若沒有，先修正主機端。
2. 哪個 kernel 占主導？分析該 kernel。
3. 它離屋頂線界限有多遠（$\eta$）？高於約 80 %：停止。
4. Speed of light：受記憶體、計算或延遲限制？
5. 受記憶體限制：DRAM 流量是否接近 $Q$？每次要求是否有 4 個 sector（對 4 位元組元素）？是否有 bank 衝突？
6. 受計算限制：忙碌的 pipeline 是否正確（應為 FMA 或 tensor，而非用來做索引運算的 ALU 或用來做除法的 MUFU）？
7. 受延遲限制：查看實際佔用率、波數和主要停滯原因。
8. 找到原始碼行（Source 頁面），只改一項，再回到步驟 1。

## 重點整理

1. 暖機後使用 event 計算 GPU 工作時間，並一律把時間換算成屋頂線界限的比例。
2. 由上而下分析：使用 Nsight Systems 分析程式、使用 Nsight Compute 分析單一 kernel，再找到原始碼行。
3. Speed of light 會給出結論：受記憶體、計算或延遲限制；「記憶體」可能指 L1 或共享記憶體，不一定是 DRAM。
4. 每次要求的 sector 數、DRAM 位元組數與 $Q$ 的比較、bank 衝突，以及每條指令的有效執行緒數，可以解釋大多數記憶體和分歧問題。
5. 主要停滯原因會指出修正方向，但必須搭配吞吐量數字解讀。
6. 佔用率是隱藏延遲的一種手段；ILP 是另一種。
7. 只要 kernel 使用共享記憶體或 warp 原語，就應執行 `compute-sanitizer`。

## 練習

1. 某 kernel 總共讀寫 1 GiB，在 A100（1.55 TB/s）上耗時 0.9 ms。$\eta$ 是多少？值得分析嗎？

    <details markdown="1"><summary>答案</summary>

    $Q = 2^{30}$ B，$T_{\min} = 2^{30} / 1.55\times10^{12} = 0.69$ ms，因此
    $\eta = 0.69 / 0.9 \approx 77$ %。這已接近實務上限（約為規格表頻寬的 90 %）；分析可能找出最後 10–15 %，例如不完整的最後一波或少數未合併存取，但更大的改善機會可能在其他地方。

    </details>

2. 某 kernel 以 `in[threadIdx.x * 4]` 讀取 `float`，Nsight Compute 回報每次載入要求有 16 個 sector。請解釋此數字。若改為 `in[threadIdx.x * 8]`，結果是多少？

    <details markdown="1"><summary>答案</summary>

    Lane 間距為 16 位元組，所以一個 warp 橫跨 512 位元組：16 個 32 位元組 sector，每個只使用 4 位元組。若跨步為 8 個 float（32 位元組），每個 lane 都命中自己的 sector：每次要求 32 個 sector。

    </details>

3. 某 kernel 每執行緒使用 96 個暫存器、每區塊 256 個執行緒，且不使用共享記憶體。在 A100 上的理論佔用率是多少？改為 128 個暫存器後會如何？

    <details markdown="1"><summary>答案</summary>

    一個區塊需要 $96 \times 256 = 24\,576$ 個暫存器，所以
    $\lfloor 65\,536 / 24\,576 \rfloor = 2$ 個區塊可容納：64 個 warp 中有 16 個，佔 25 %。
    使用 128 個暫存器時，一個區塊需要 32 768 個：仍可容納 2 個區塊，佔 25 %。使用 64 個暫存器時則可容納 4 個區塊，佔 50 %。

    </details>

4. 某歸約 kernel 的主要停滯原因是「barrier」。這代表什麼？第 03 章的哪項變更可改善？

    <details markdown="1"><summary>答案</summary>

    Warp 會在 `__syncthreads()` 等待區塊中的其他 warp；這通常發生在共享記憶體樹的最後幾步，因為只剩少數 warp 仍在工作。第 03 章的 warp shuffle 版本可在單一 warp 內完成最後 32 個值，不需 barrier；雙層版本每區塊只需一次 barrier。

    </details>

5. 為何 Nsight Compute 有時會回報比 `--bench` 更長的 kernel 時間？如何讓兩者一致？

    <details markdown="1"><summary>答案</summary>

    它會將時脈鎖定在基準頻率（`--clock-control base`）以確保執行結果可重現，而 benchmark 會以加速時脈執行。使用 `--clock-control none` 即可比較；也可以比較 kernel 間的比例，而非絕對時間。Replay 間的快取清除（`--cache-control all`，預設值）也會讓短 kernel 比在迴圈中從 L2 取得資料時更慢。

    </details>

6. 在 GPU 上分析範例程式，並以自己的數字填寫第 6.2 節表格。哪個 kernel 的頻寬最接近 `copyCoalesced`？

## 實作練習

- [LeetGPU – 矩陣轉置](../leetgpu/003-matrix-transpose/)（第 6.2 節的轉置）
- [LeetGPU – 矩陣複製](../leetgpu/031-matrix-copy/)、[LeetGPU – 向量加法](../leetgpu/001-vector-add/)（受頻寬限制的基準）
- [LeetGPU – ReLU](../leetgpu/021-relu/)（與第 00 章的屋頂線界限比較）
