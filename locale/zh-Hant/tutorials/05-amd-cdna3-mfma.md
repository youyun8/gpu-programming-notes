# 05 – AMD CDNA3 與 MFMA：從 CUDA 到 wave64 矩陣核心

> **第五部 · AMD 架構與函式庫** · 先備知識：[矩陣乘法 1](04-tiled-matmul.md)（最好也讀過 [矩陣乘法 8](gemm/07-tensor-cores.md)） ·
> 下一章：[06 – 拆解手寫的 AMD GEMM](06-aiter-asm-gemm.md)

第 01–03 章與矩陣乘法 1 都使用 CUDA 的術語。本章把這些概念對應到 AMD 的資料中心 GPU（CDNA3：MI300X / MI300A / MI325X，ISA 目標為 `gfx942`），再用 MFMA 矩陣核心指令寫一個小型的 bf16 GEMM，並閱讀編譯器產生的 ISA。第 06、07 章會用這套術語，拆解 **AITER** 中手寫組合語言的 GEMM，以及 **hipBLASLt** 自動產生的 GEMM。

**你將學到**

- CUDA 的概念如何對應到 AMD（CU、wavefront、LDS、VGPR/AGPR/SGPR）；
- 哪些硬體差異會改變 kernel 的寫法：wave64、純量單元、需要明確等待的記憶體計數器、由暫存器決定的佔用率，以及 XCD 的分派方式；
- 一道 MFMA 指令計算什麼，以及每個 lane 究竟持有哪些運算元元素；
- 如何在沒有 GPU 的情況下為 gfx942 編譯 HIP kernel 並閱讀 ISA；
- 為什麼直截了當的 MFMA GEMM 離峰值很遠，以及縮小差距的各項技術（第 06–07 章）。

## 1. 術語對照

### 1.1 從 CUDA 到 HIP

| CUDA | HIP / AMD | 說明 |
|------|-----------|-------|
| SM | CU（compute unit，運算單元） | MI300X：304 個 CU，每個 XCD 38 個，共 8 個 XCD |
| Warp（32） | **Wavefront（64）** | CDNA 上 `warpSize == 64`；lane 遮罩為 64 位元 |
| 共享記憶體 | LDS（local data share） | gfx942 上每個 CU 有 64 KiB |
| 暫存器 | VGPR（每個 lane 各自一份）、**AGPR**（每個 lane 各自一份，放累加器）、SGPR（整個 wave 共用，純量） | 每個 wave 的每個 lane 最多 512 個 VGPR+AGPR |
| Tensor core / `mma.sync` | **MFMA**（`v_mfma_*`） | 整個 wave 執行一道指令，運算元分散在 64 個 lane 上 |
| `cp.async` / TMA | `buffer_load … lds`（直接載入 LDS） | 從全域記憶體直接寫入 LDS，不經過 VGPR |
| `__syncthreads()` | `s_barrier`（加上 fence） | |
| Scoreboard（硬體自動追蹤相依） | **明確的** `s_waitcnt vmcnt/lgkmcnt` | 由編譯器（或手寫組合語言的人）主動等待計數器 |
| 整顆 GPU 一份 L2 | **每個 XCD 一份** L2（4 MiB）+ 256 MiB Infinity Cache | workgroup 被分派到哪個 XCD 很重要 |

### 1.2 會改變 kernel 寫法的差異

1. **Wave64。** 對整個 wave 做歸約需要 `log2(64) = 6` 步；ballot 與遮罩都是 64 位元寬。使用寬度為 64 的 `__shfl_xor`，組合語言中則用 DPP 或 `ds_swizzle`。
2. **純量單元。** 在整個 wave 中都相同的值（指標、迴圈計數器、步幅）放在 SGPR 中。純量 ALU 的運算可以與向量運算同時進行；`s_load_dword` 經由純量快取讀取 kernel 參數。
3. **明確的記憶體計數器。** 每個 wave 都有幾個計數器，記錄尚未完成的操作：
   - `vmcnt`：向量記憶體（全域與 buffer）的載入。
   - `lgkmcnt`：LDS、GDS、常數與訊息操作。
   - `expcnt`：export 操作。

   `s_waitcnt vmcnt(N)` 會一直等到尚未完成的向量載入不超過 `N` 筆為止。載入依序返回，因此 `vmcnt(N)` 的意思是「除了最近發出的 N 筆之外，其餘都已抵達」。手寫 kernel 就是靠它把載入做成管線。
4. **佔用率幾乎完全由暫存器決定，比 NVIDIA 明顯得多。** 每個 SIMD 的每個 lane 有 512 個暫存器，由其上的所有 wave 分用：
   - 用滿 512 個（256 個 VGPR + 256 個 AGPR）的 kernel，**每個 SIMD 只能放一個 wave**，也就是每個 CU 四個 wave。
   - 最快的 AMD GEMM 正是這樣設計的：每個 SIMD 只跑一個「肥大」的 wave，用軟體管線化來隱藏延遲，而不是靠切換到其他 wave。
5. **workgroup 以輪流方式分派到各個 XCD。** 第 `i` 個 workgroup 會送到第 `i % 8` 個 XCD。
   - 因此，兩個共用同一段 `A` 的相鄰輸出分塊，會落在不同的 L2 上。
   - hipBLASLt 的 `WorkGroupMappingXCC`（第 07 章）就是為了抵銷這個效應。

![MI300X：8 個 XCD、每個 38 個 CU，各有自己的 L2；workgroup 以輪流方式分派到各 XCD](figures/ch05-mi300x.svg)


### 1.3 第一個範例：對一個 wave 做歸約

第 03 章的 wave 歸約在 wave64 上要多一步，而且寬度應取自 `warpSize`，而不是寫死的 32：

```cpp
__device__ float waveReduceSum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) v += __shfl_down(v, offset);  // 6 steps on wave64
    return v;   // valid in lane 0
}
```

把 32 寫死的程式碼（以 `unsigned` 存 lane 遮罩、`threadIdx.x % 32`、32 格的 per-warp 部分和陣列）是移植時最常見的錯誤。HIP 也能把同一份原始碼編譯給 NVIDIA 使用，那時 `warpSize` 就是 32。

## 2. MFMA：一道指令、一個 wave、一整塊分塊

### 2.1 一道指令計算什麼

`v_mfma_f32_16x16x16_bf16 D, A, B, C` 由一個 wave 的 64 個 lane 合作，對 16×16×16 的分塊計算 `D = A·B + C`。它和 FMA 不同：FMA 是每個 lane 各自處理自己的資料，MFMA 則是 *wave 層級* 的操作——每個 lane 提供幾個運算元、取回幾個結果，中間的 $16\times16\times16$ 乘積由矩陣核心完成。

### 2.2 每個 lane 持有什麼

| 運算元 | 每個 lane 的大小 | lane `l`（`l ∈ [0, 64)`）持有的元素 |
|---------|---------------|-----------------------------------------------|
| `A`（16×16，bf16） | 4 × bf16 = 2 個 VGPR | 第 `l % 16` 列，k = `4·(l/16) … 4·(l/16)+3` |
| `B`（16×16，bf16） | 4 × bf16 = 2 個 VGPR | 第 `l % 16` 欄，k = `4·(l/16) … 4·(l/16)+3` |
| `C`/`D`（16×16，fp32） | 4 × fp32 = 4 個暫存器 | 第 `l % 16` 欄，第 `4·(l/16) … 4·(l/16)+3` 列 |

以公式表示，對 lane $\ell$ 與暫存器位置 $t \in \{0, 1, 2, 3\}$：

$$
a_{\ell,t} = A\bigl[\ell \bmod 16,\ 4\lfloor \ell/16 \rfloor + t\bigr], \qquad
b_{\ell,t} = B^{\mathsf T}\bigl[\ell \bmod 16,\ 4\lfloor \ell/16 \rfloor + t\bigr], \qquad
d_{\ell,t} = D\bigl[4\lfloor \ell/16 \rfloor + t,\ \ell \bmod 16\bigr]
$$

$$
D_{ij} = \sum_{k=0}^{15} A_{ik}\,B_{kj} + C_{ij}, \qquad 0 \le i, j < 16
$$

| 符號 | 意義 |
|---|---|
| $\ell$ | lane 索引，$0 \dots 63$ |
| $t$ | 該 lane 的 4 個值中的第幾個（2 個 VGPR 中的 bf16 半字，或 4 個累加器暫存器） |
| $a_{\ell,t}, b_{\ell,t}$ | lane $\ell$ 持有的運算元值 |
| $d_{\ell,t}$ | lane $\ell$ 持有的累加值 |
| $B^{\mathsf T}[j, k]$ | 即 $B[k, j]$：B 運算元先以輸出欄、再以 $k$ 索引 |
| $A, B, C, D$ | $16\times16$ 的運算元、輸入累加值與結果分塊 |

![v_mfma_f32_16x16x16_bf16 中，每個 lane 持有 A、B、D 的哪些元素](figures/ch05-mfma-layout.svg)

### 2.3 吞吐量

一道指令執行 $2\cdot16^3 = 8192$ 次浮點運算。整顆晶片的峰值為

$$
F = n_{\text{CU}}\cdot f\cdot \phi, \qquad 304 \times 2.1\ \text{GHz} \times 2048 \approx 1.31\ \text{PFLOP/s (dense bf16, MI300X)}
$$

| 符號 | 意義 |
|---|---|
| $n_{\text{CU}}$ | 運算單元數（MI300X 為 304） |
| $f$ | 引擎的峰值時脈 |
| $\phi$ | 每個 CU 每個時脈的稠密 bf16 運算量（CDNA3 為 2048，分散在 4 個 SIMD 上） |

### 2.4 為什麼採用 TN 配置

和 [leetgpu/022-gemm](../leetgpu/022-gemm/solution.cu) 中不公開細節的 NVIDIA WMMA fragment 相比，AMD 的運算元配置有正式文件，手寫 kernel 也仰賴它。

最重要的結果是：**A 運算元是同一列中 4 個連續的 k 值；當 B 以 `[N][K]` 存放時，B 運算元也是如此。** 若兩個矩陣都沿 K 連續（即「TN」配置，`C = A · Bᵀ`），那麼每次取運算元都是每個 lane 一次對齊的 8 位元組讀取。這就是 AITER 的組合語言 GEMM 全部都是 `_tn_` 的原因，也說明了 PyTorch `nn.Linear` 的權重配置 `[out, in]` 恰好合適。

### 2.5 其他形狀

其他形狀的指令遵循相同的概念：
- `32x32x8`：每個 FLOP 需要的指令更少，累加器為 16 個暫存器。
- `16x16x32` 的 fp8 版本。
- `v_mfma_f32_16x16x16_f16`。
- `v_mfma_i32_16x16x32_i8` 等等。

權威的表格在 *CDNA3 ISA* 指南，以及 AMD 的 [Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)。這個計算工具會印出每道指令確切的「暫存器 ↔ 元素」對應，以及所需的週期數。

### 2.6 累加暫存器（AGPR）

累加器可以放在 **AGPR**（`a[0:3]`）或 VGPR 中：
- AGPR 大約讓一個 wave 可用的暫存器數量加倍。
- 在兩者之間搬移資料需要 `v_accvgpr_read/write`。
- 編譯器會把累加器放在 AGPR；手寫 kernel 甚至會把*運算元*也暫存在那裡（第 06 章）。

## 3. 教學用 kernel

### 3.1 結構

[`tutorials/amd/mfma_gemm.hip`](amd/mfma_gemm.hip) 是一個完整的 bf16 TN GEMM，約 120 行 device 程式碼，另附主機端的測試與計時：

- **區塊分塊為 64×64×32**，256 個執行緒 = 4 個 wave，排成 2×2。每個 wave 負責 32×32，也就是 2×2 個 MFMA 分塊。
- **全域記憶體 → 暫存器 → LDS**。每一步 K，每個執行緒搬運 A 與 B 各 16 位元組。LDS 每列補齊到 40 個 bf16（80 位元組），讓一次運算元讀取所涉及的 16 列分散到不同的 bank。
- **LDS 雙緩衝，每一步只需一次 barrier**：
  1. 發出分塊 `k+1` 的全域載入。
  2. 從 LDS 讀取分塊 `k`，執行 MFMA。
  3. 把分塊 `k+1` 寫入另一個緩衝區。
  4. Barrier。

  另一個緩衝區最後一次被讀取是在上一輪迭代，而上一輪以 barrier 結束，因此覆寫它是安全的。
- **取運算元**完全依照上表：

  ```cpp
  // One MFMA operand: lane l supplies row (l % 16), k = k0 + 4 * (l / 16) .. +3.
  const uint16_t* p = tile + (row0 + lane % 16) * kLdsStride + k0 + 4 * (lane / 16);
  return *reinterpret_cast<const Short4*>(p);
  ```
- **Epilogue** 把 `acc[i][j][r]` 寫到第 `4·(lane/16) + r` 列、第 `lane % 16` 欄。

### 3.2 不用 GPU 也能編譯

查看 ISA 不需要安裝 ROCm。版本 ≥ 17 的標準 clang 就內建 AMDGPU 後端，[`hip_compat.h`](amd/hip_compat.h) 則提供這個 kernel 用到的少數 HIP 巨集：

```bash
clang++ -x hip -nogpuinc -nogpulib --cuda-device-only --offload-arch=gfx942 \
        -O3 -S -o mfma_gemm.s tutorials/amd/mfma_gemm.hip
```

### 3.3 閱讀 ISA

以下是 clang 18 產生的內層迴圈（已刪減）：

```asm
.LBB0_6:                                   ; K loop
	global_load_dwordx4 v[0:3], v[24:25], off    ; next A tile (16 B / lane)
	global_load_dwordx4 v[4:7], v[22:23], off    ; next B tile
	...
	ds_read2_b64 v[22:25], v21 offset1:4         ; A operands for k0 = 0 and 16, fused
	ds_read2_b64 v[34:37], v21 offset0:160 offset1:164
	ds_read2_b64 v[26:29], v30 offset1:4         ; B operands
	ds_read2_b64 v[30:33], v30 offset0:160 offset1:164
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_bf16 a[12:15], v[22:23], v[26:27], a[12:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_bf16 a[8:11], v[22:23], v[30:31], a[8:11]
	... 6 more MFMAs ...
	s_waitcnt vmcnt(1)
	ds_write_b128 v22, v[0:3]                    ; stage next tile into the other buffer
	s_waitcnt vmcnt(0)
	ds_write_b128 v21, v[4:7]
	...
	s_barrier
```

請注意：

1. 累加器被放進 AGPR：`a[0:15]`，`.agpr_count: 16`。
2. `k0 = 0` 與 `k0 = 16` 的兩次 8 位元組讀取，被合併成一道 `ds_read2_b64`（`offset1:4` 表示再往後 4 × 8 位元組 = 16 個 bf16）。
3. `s_waitcnt lgkmcnt(1)` 讓第一道 MFMA 在最後一筆 LDS 讀取仍在進行時就開始。
4. 每一步 K 有 8 道 MFMA，但搭配 2 次全域載入、4 次 LDS 讀取與 2 次 LDS 寫入。**這個比例太低了。**
   - MFMA 管線會等不到資料。
   - 預期只能達到 MI300X 約 1.3 PFLOP/s 稠密 bf16 峰值的一小部分。

   本章接下來的內容以及後兩章，都在設法改善這個比例。

### 3.4 為什麼慢：分塊的運算強度

用區塊分塊的算術就能具體看出問題。一個負責 $B_M\times B_N$ 輸出分塊、以 $B_K$ 為單位走完 $K$ 的 workgroup，每一段載入 $(B_M + B_N)B_K$ 個 bf16 值，執行 $2B_MB_NB_K$ 次浮點運算：

$$
I_{\text{tile}} = \frac{2B_MB_NB_K}{2\,(B_M + B_N)\,B_K} = \frac{B_MB_N}{B_M + B_N}\ \frac{\text{flop}}{\text{byte}}, \qquad
I^{\star} = \frac{F}{\beta} \approx \frac{1.31\times10^{15}}{5.3\times10^{12}} \approx 250\ \frac{\text{flop}}{\text{byte}}
$$

| 符號 | 意義 |
|---|---|
| $B_M, B_N, B_K$ | workgroup 的分塊大小（此處為 64、64、32） |
| $I_{\text{tile}}$ | workgroup 從 L2/HBM 取得的每個位元組所對應的運算量 |
| $\beta$ | HBM 頻寬（MI300X 約 5.3 TB/s） |
| $I^{\star}$ | 晶片的轉折點（ridge point） |

教學 kernel 的 $64\times64$ 分塊得到 $I_{\text{tile}} = 32$，只有轉折點的八分之一：若沒有快取幫忙，最多只能達到峰值的約 13%。$256\times256$ 的分塊可達 128，其餘則靠相鄰分塊在 L2 與 Infinity Cache 中的重用補足。

> 本儲存庫會為 gfx942 編譯檢查這個 kernel，但它從未實際執行過：CI 中沒有 AMD GPU。在 ROCm 機器上，執行
> `hipcc -O3 --offload-arch=gfx942 tutorials/amd/mfma_gemm.hip -o mfma_gemm && ./mfma_gemm 4096 4096 4096`
> 會以 fp64 的 CPU 參考實作檢查結果，並印出 TFLOP/s。

## 4. 快速的 GEMM 有何不同

### 4.1 各個步驟

以下每個步驟都能在第 06 章的反組譯結果中看到：

| 步驟 | 改變 | 效果 |
|------|--------|--------|
| 1 | **每個 wave 負責更大的分塊**（例如每個 wave 16×128 或 64×64，每個 workgroup 128×128–256×256） | 每載入一個位元組能做更多 MFMA；累加器增加到 128–256 個暫存器 |
| 2 | **直接載入 LDS**（`buffer_load_dword … lds`） | 全域資料直接寫入 LDS，不經 VGPR 中轉、不需要 `ds_write`，指令與暫存器都更少 |
| 3 | **wave 之間不共用的運算元就不經過 LDS** | 若每個 wave 負責 B 中不同的欄，就可以把 B 直接載入暫存器。權重會事先離線重排成 MFMA 運算元的順序（`bpreshuffle`） |
| 4 | **以計數器做軟體管線化** | 讓 2–3 步 K 的載入同時進行，並以 `s_waitcnt vmcnt(N)` 等待（`N` 為之後又發出的載入數），而不是 `vmcnt(0)` |
| 5 | **把所有指令穿插在 MFMA 之間** | 一道 MFMA 會占用矩陣管線好幾個週期。在相鄰兩道 MFMA 之間發出一次載入、LDS 讀取或位址更新，就能完全隱藏它們的發出成本。編譯器在這方面做得不好，所以最好的 kernel 都是手寫組合語言或自動產生的（TensileLite 的 `ScheduleIterAlg`） |
| 6 | **針對小的 M·N 使用 Split-K / Stream-K** | 輸出分塊太少、填不滿 304 個 CU？把 K 迴圈拆給多個 workgroup，再用原子操作或修正步驟加總 |
| 7 | **考慮快取的分塊順序** | 讓同時在同一個 XCD 上執行的 workgroup，共用該 XCD L2 中的 A/B 面板 |

### 4.2 填滿整台機器

步驟 6 的重要性只要簡單計數就能看出。假設有 $T$ 個輸出分塊、$n_{\text{CU}}$ 個運算單元，每個一次執行一個 workgroup：

$$
T = \left\lceil \frac{M}{B_M} \right\rceil\left\lceil \frac{N}{B_N} \right\rceil, \qquad
\text{waves} = \left\lceil \frac{T}{n_{\text{CU}}} \right\rceil, \qquad
\eta_{\text{fill}} = \frac{T}{n_{\text{CU}}\cdot\text{waves}}
$$

| 符號 | 意義 |
|---|---|
| $T$ | 輸出分塊數（不使用 split-K 時即 workgroup 數） |
| Waves | 涵蓋所有分塊所需的 workgroup 輪數 |
| $\eta_{\text{fill}}$ | CU 時間中做有用工作的比例（不計各分塊之間的負載不均） |

以 $M = N = 2048$、$256\times256$ 分塊為例，$T = 64$：MI300X 的 304 個 CU 只有 21% 在工作。若 $T = 320$，則需要兩輪，而第二輪只填滿 5%，因此 $\eta_{\text{fill}} = 53\%$。Split-K 把 $T$ 乘上切分數；Stream-K 則讓每個 CU 平均分到 $T\cdot\lceil K/B_K\rceil$ 次迴圈迭代（第 07 章）。

## 5. 實用工具

| 工具 | 用途 |
|------|-----|
| `llvm-objdump -d --mcpu=gfx942 file.co` | 反組譯程式碼物件（`.co`、`.hsaco`） |
| `readelf --notes file.co` | kernel 的中繼資料：VGPR/AGPR/SGPR 數量、LDS 大小、參數配置 |
| `roc-obj-ls` / `roc-obj-extract` | 從 fat binary / `.so` 中取出程式碼物件 |
| `rocprofv3 --kernel-trace --stats` | kernel 執行時間 |
| `rocprofv3 --pmc SQ_INSTS_VALU_MFMA_MOPS_BF16 …` | 硬體計數器（MFMA 使用率、LDS bank 衝突 `SQ_LDS_BANK_CONFLICT`） |
| rocprofv3 ATT（thread trace）+ Radeon GPU Analyzer / ROCm Compute Viewer | 指令層級的時間軸：每個 wave 卡在哪裡 |
| `rocprof-compute`（Omniperf） | Roofline 與「speed of light」摘要 |

## 重點整理

1. CU 就是 SM，wavefront 是 64 個 lane 的 warp，LDS 就是共享記憶體；真正重要的差異是 wave64、純量單元、明確的 `s_waitcnt` 計數器、由暫存器決定的佔用率，以及每個 XCD 各自的 L2。
2. MFMA 是 wave 層級的指令，運算元配置有正式文件：lane $\ell$ 持有 A（以及 B 的欄）第 $\ell \bmod 16$ 列上 4 個連續的 $k$，並持有 D 同一欄中的 4 列。
3. 兩個運算元都沿 K 連續（TN）時，每次取運算元都是每個 lane 一次對齊的 8 位元組讀取。
4. 寫出正確的 MFMA GEMM 很容易；要寫出快的，需要大的 wave 分塊、直接載入 LDS、以計數器做軟體管線化、指令穿插、工作切分，以及考慮快取的分塊順序——這正是第 06–07 章的主題。

## 練習

### 修改 kernel

1. 把教學 kernel 改用 32×32×8 的 MFMA（`__builtin_amdgcn_mfma_f32_32x32x8bf16_1k`），並用 Matrix Instruction Calculator 推出新的運算元與累加器配置。

    <details markdown="1"><summary>提示</summary>

    計算工具的 `--detail-instruction` 與 `--register-layout` 選項（見其 `--help`）會印出 CDNA3 指令的「暫存器 ↔ 元素」對應表。累加器會變成每個 lane 16 個暫存器，而每個 lane 持有 A 某一列上 4 個連續的 $k$（32 列 × 2 組、每組 4 個 $k$，分散在 64 個 lane 上）。

    </details>

2. 讓每個 wave 計算 32×64，而不是 32×32。編譯器現在回報幾個 AGPR？佔用率會如何變化？

    <details markdown="1"><summary>提示</summary>

    累加器從每個 lane 16 個暫存器加倍為 32 個（2×4 個分塊，每個 4 個暫存器）。請查看 `.s` 中繼資料裡的 `.agpr_count` 與 `.vgpr_count`，並記得每個 SIMD 的每個 lane 只有 512 個暫存器，由其上所有 wave 分用。

    </details>

3. 把經由暫存器中轉的載入改成直接載入 LDS：使用 `__builtin_amdgcn_global_load_lds`（clang 19 或更新版本）或內嵌組合語言，並比較迴圈中的指令數。

### 估算

4. 在 MI300X 上，以 $256\times256$ 分塊計算 $M = 4096$、$N = 1024$ 的 $\eta_{\text{fill}}$。改用 $S = 4$ 的 split-K 後會如何？

    <details markdown="1"><summary>答案</summary>

    $T = 16\cdot4 = 64$ 個分塊，只需一輪，$\eta = 64/304 = 21\%$。使用 $S = 4$ 時有 256 個 workgroup，$\eta = 256/304 = 84\%$（尚未計入合併部分分塊的成本）。

    </details>
