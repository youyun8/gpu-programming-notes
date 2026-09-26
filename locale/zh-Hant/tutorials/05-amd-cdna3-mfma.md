# 05 – AMD CDNA3 與 MFMA：從 CUDA 到 wave64 Matrix Core

> **第五部 · AMD Production Kernel** · 先備知識：[04](04-tiled-matmul.md)（最好也讀過 [04.7](gemm/07-tensor-cores.md)） ·
> 下一章：[06 – 手寫 AMD GEMM 內部](06-aiter-asm-gemm.md)

第 01–04 章使用 CUDA 詞彙。本章將其對應到 AMD 的資料中心 GPU（CDNA3：MI300X / MI300A / MI325X，ISA target `gfx942`），接著用 MFMA matrix-core instruction 建立小型 bf16 GEMM，並閱讀 compiler 產生的 ISA。第 06、07 章會用這些詞彙拆解 **AITER** 的手寫 assembly GEMM，以及 **hipBLASLt** 產生的 GEMM。

**你將學會**

- CUDA 概念如何對應到 AMD（CU、wavefront、LDS、VGPR/AGPR/SGPR）；
- 會改變 kernel 寫法的硬體差異：wave64、scalar unit、明確 memory counter、register 限制的 occupancy，以及 XCD placement；
- 一個 MFMA instruction 計算什麼，以及每個 lane 究竟持有哪些 operand element；
- 如何在沒有 GPU 時為 gfx942 編譯 HIP kernel 並閱讀 ISA；
- 為何直接的 MFMA GEMM 遠低於峰值，以及縮小差距的技術清單（第 06–07 章）。

## 1. 詞彙對照

### 1.1 從 CUDA 到 HIP

| CUDA | HIP / AMD | 備註 |
|------|-----------|-------|
| SM | CU（compute unit） | MI300X：304 個 CU，每個 XCD 38 個，共 8 個 XCD |
| Warp（32） | **Wavefront（64）** | CDNA 上 `warpSize == 64`；lane mask 是 64-bit |
| Shared memory | LDS（local data share） | gfx942 每個 CU 64 KiB |
| Register | VGPR（每 lane）、**AGPR**（每 lane，accumulator）、SGPR（每 wave，scalar） | 每 lane、每 wave 最多 512 個 VGPR+AGPR |
| Tensor core / `mma.sync` | **MFMA**（`v_mfma_*`） | 每個 wave 一條 instruction，operand 分散在 64 個 lane |
| `cp.async` / TMA | `buffer_load … lds`（direct-to-LDS） | Global → LDS，不經 VGPR |
| `__syncthreads()` | `s_barrier`（加 fence） | |
| Scoreboard | 明確的 `s_waitcnt vmcnt/lgkmcnt` | Compiler（或 asm 作者）等待 counter |
| 每 GPU 一份 L2 | 每個 XCD 一份 **L2**（4 MiB）+ 256 MiB Infinity Cache | Workgroup→XCD placement 很重要 |

### 1.2 會改變 Kernel 寫法的差異

1. **Wave64。** Wave reduction 需要 `log2(64) = 6` 步。Ballot 與 mask 是 64-bit。使用寬度 64 的 `__shfl_xor`，或 asm 中的 DPP / `ds_swizzle`。
2. **Scalar unit。** 整個 wave 相同的值（pointer、loop counter、stride）存於 SGPR。Scalar ALU 工作可與 vector 工作同時執行。`s_load_dword` 透過 scalar cache 讀取 kernel argument。
3. **明確 memory counter。** 每個 wave 都有 outstanding operation counter：
   - `vmcnt`：vector memory（global 與 buffer）load。
   - `lgkmcnt`：LDS、GDS、constant 與 message operation。
   - `expcnt`：export。

   `s_waitcnt vmcnt(N)` 會阻塞，直到最多只剩 `N` 個 vector load 尚未完成。Load 依序回傳，所以 `vmcnt(N)` 表示「除了最新的 N 個，其他都已到位」。手寫 kernel 以此 pipeline load。
4. **Occupancy 遠比 NVIDIA 更受 register 數量左右。** 每個 SIMD 有 512 個 register per lane，供其 wave 分享：
   - 使用全部 512 個（256 VGPR + 256 AGPR）的 kernel，每個 SIMD 只能有**一個 wave**，因此每個 CU 四個 wave。
   - 最快的 AMD GEMM 正是如此：每個 SIMD 一個肥大的 wave，以 software pipeline 而非其他 wave 隱藏 latency。
5. **Workgroup 以 round-robin 分派到 XCD。** Workgroup `i` 會到 XCD `i % 8`。
   - 共享一個 `A` panel 的相鄰 output tile 因此落在不同 L2。
   - hipBLASLt 的 `WorkGroupMappingXCC`（第 07 章）就是為了抵銷這點。

![MI300X：8 個 XCD，每個有 38 個 CU 與自己的 L2；workgroup 以 round-robin 分派到 XCD](figures/ch05-mi300x.svg)

### 1.3 第一個範例：對 Wave 做 Reduction

第 03 章的 wave reduction 在 wave64 上多一步，寬度取自 `warpSize`，而不是寫死 32：

```cpp
__device__ float waveReduceSum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) v += __shfl_down(v, offset);  // 6 steps on wave64
    return v;   // valid in lane 0
}
```

寫死 32（將 lane mask 宣告為 `unsigned`、使用 `threadIdx.x % 32`，或建立 32-entry 的 per-warp partial array）是最常見的 porting bug。HIP 也會為 NVIDIA 編譯同一份原始碼，該處的 `warpSize` 是 32。

## 2. MFMA：一條 Instruction、一個 Wave、一整個 Tile

### 2.1 一條 Instruction 計算什麼

`v_mfma_f32_16x16x16_bf16 D, A, B, C` 由 wave 的 64 個 lane 合作，為 16×16×16 tile 計算 `D = A·B + C`。FMA 由每個 lane 對自己的資料執行；MFMA 則是 *wave-level* 操作：每個 lane 貢獻幾個 operand value、接收幾個結果，matrix core 在中間完成 $16\times16\times16$ 乘積。

### 2.2 每個 Lane 持有什麼

| Operand | 每 lane 大小 | Lane `l` 持有的 element（`l ∈ [0, 64)`） |
|---------|---------------|-----------------------------------------------|
| `A`（16×16, bf16） | 4 × bf16 = 2 VGPR | Row `l % 16`，k = `4·(l/16) … 4·(l/16)+3` |
| `B`（16×16, bf16） | 4 × bf16 = 2 VGPR | Col `l % 16`，k = `4·(l/16) … 4·(l/16)+3` |
| `C`/`D`（16×16, fp32） | 4 × fp32 = 4 regs | Col `l % 16`，row `4·(l/16) … 4·(l/16)+3` |

以 lane $\ell$ 與 register slot $t \in \{0, 1, 2, 3\}$ 表示：

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
| $\ell$ | Lane index，$0 \dots 63$ |
| $t$ | 該 lane 的 4 個值中哪一個（2 個 VGPR 的 bf16 half，或 4 個 accumulator register） |
| $a_{\ell,t}, b_{\ell,t}$ | Lane $\ell$ 持有的 operand value |
| $d_{\ell,t}$ | Lane $\ell$ 持有的 accumulator value |
| $B^{\mathsf T}[j, k]$ | $B[k, j]$：B operand 先依 output column，再依 $k$ indexing |
| $A, B, C, D$ | $16\times16$ operand、輸入 accumulator 與 result tile |

![v_mfma_f32_16x16x16_bf16 中每個 lane 持有 A、B 與 D 的哪些 element](figures/ch05-mfma-layout.svg)

### 2.3 Throughput

一條 instruction 執行 $2\cdot16^3 = 8192$ FLOP。晶片峰值為

$$
F = n_{\text{CU}}\cdot f\cdot \phi, \qquad 304 \times 2.1\ \text{GHz} \times 2048 \approx 1.31\ \text{PFLOP/s (dense bf16, MI300X)}
$$

| 符號 | 意義 |
|---|---|
| $n_{\text{CU}}$ | Compute unit（MI300X 為 304） |
| $f$ | 峰值 engine clock |
| $\phi$ | 每 CU、每 clock 的 dense bf16 FLOP（CDNA3 為 2048，分散於 4 個 SIMD） |

### 2.4 為何使用 TN Layout

對照 [leetgpu/022-gemm](../leetgpu/022-gemm/solution.cu) 中不透明的 NVIDIA WMMA fragment：AMD 的 layout 有文件說明，手寫 kernel 會依賴它。

重要結果是：**A operand 是同一 row 中 4 個連續 k 值；B 以 `[N][K]` 儲存時也一樣。** 若兩個 matrix 的 K 都連續（「TN」layout，`C = A · Bᵀ`），每次 operand fetch 都是每 lane 一次對齊的 8-byte read。這就是 AITER asm GEMM 全為 `_tn_` 的原因，也解釋了為何 PyTorch 的 `nn.Linear` weight layout `[out, in]` 恰好合適。

### 2.5 其他 Shape

其他 shape 採相同概念：
- `32x32x8`：每 FLOP 的 instruction 較少，16 個 accumulator register。
- `16x16x32` fp8。
- `v_mfma_f32_16x16x16_f16`。
- `v_mfma_i32_16x16x32_i8` 等。

權威表格位於 *CDNA3 ISA* guide 與 AMD 的 [Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)。Calculator 會印出每條 instruction 確切的 register ↔ element mapping 與 cycle count。

### 2.6 Accumulation Register（AGPR）

Accumulator 可存於 **AGPR**（`a[0:3]`）或 VGPR：
- AGPR 大致讓 wave 可用的 register file 加倍。
- 兩者間搬移需要 `v_accvgpr_read/write`。
- Compiler 會將 accumulator 放入 AGPR。手寫 kernel 也會把 *operand* 暫存其中（第 06 章）。

## 3. 教學 Kernel

### 3.1 結構

[`tutorials/amd/mfma_gemm.hip`](amd/mfma_gemm.hip) 是完整 bf16 TN GEMM，約 120 行 device code，另含 host test 與 timer：

- **Block tile 64×64×32**，256 threads = 4 waves，排列為 2×2。每個 wave 負責 32×32 = 2×2 個 MFMA tile。
- **Global → register → LDS**。每個 thread 在每個 K step 搬移 16 byte A 與 16 byte B。LDS row padding 到 40 個 bf16（80 byte），讓一次 operand read 觸及的 16 row 分散至不同 bank。
- **Double-buffered LDS，每 step 一個 barrier**：
  1. 發出 tile `k+1` 的 global load。
  2. 從 LDS 對 tile `k` 執行 MFMA。
  3. 將 tile `k+1` 寫入另一個 buffer。
  4. Barrier。

  另一個 buffer 上次是在前一 iteration 讀取，而該 iteration 以 barrier 結束，因此可以安全覆寫。
- **Operand fetch** 完全符合上表：

  ```cpp
  // One MFMA operand: lane l supplies row (l % 16), k = k0 + 4 * (l / 16) .. +3.
  const uint16_t* p = tile + (row0 + lane % 16) * kLdsStride + k0 + 4 * (lane / 16);
  return *reinterpret_cast<const Short4*>(p);
  ```
- **Epilogue** 將 `acc[i][j][r]` 寫到 row `4·(lane/16) + r`、column `lane % 16`。

### 3.2 不用 GPU 也能編譯

查看 ISA 不需要 ROCm。一般 clang ≥ 17 已有 AMDGPU back end。[`hip_compat.h`](amd/hip_compat.h) 提供 kernel 用到的少數 HIP macro：

```bash
clang++ -x hip -nogpuinc -nogpulib --cuda-device-only --offload-arch=gfx942 \
        -O3 -S -o mfma_gemm.s tutorials/amd/mfma_gemm.hip
```

### 3.3 閱讀 ISA

以下是 clang 18 產生的 inner loop（已刪節）：

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

1. Accumulator 進入 AGPR：`a[0:15]`、`.agpr_count: 16`。
2. `k0 = 0` 與 `k0 = 16` 的兩次 8-byte read 合併成一次 `ds_read2_b64`（`offset1:4` 是 4 × 8 byte，也就是再往後 16 個 bf16）。
3. `s_waitcnt lgkmcnt(1)` 讓第一個 MFMA 在最後一次 LDS read 尚未完成時就開始。
4. 每個 K step 有 8 個 MFMA，對上 2 個 global load、4 個 LDS read 與 2 個 LDS write。**這個比例太低。**
   - MFMA pipeline 會缺料。
   - 預期只能達到 MI300X 約 1.3 PFLOP/s dense bf16 峰值的一小部分。

   本章後半與接下來兩章都在修正此比例。

### 3.4 為何很慢：Tile Intensity

Block-tile arithmetic 讓問題具體化。負責 $B_M\times B_N$ output tile、以 $B_K$ slice 走過 $K$ 的 workgroup，每個 slice 載入 $(B_M + B_N)B_K$ 個 bf16 值，執行 $2B_MB_NB_K$ FLOP：

$$
I_{\text{tile}} = \frac{2B_MB_NB_K}{2\,(B_M + B_N)\,B_K} = \frac{B_MB_N}{B_M + B_N}\ \frac{\text{flop}}{\text{byte}}, \qquad
I^{\star} = \frac{F}{\beta} \approx \frac{1.31\times10^{15}}{5.3\times10^{12}} \approx 250\ \frac{\text{flop}}{\text{byte}}
$$

| 符號 | 意義 |
|---|---|
| $B_M, B_N, B_K$ | Workgroup tile size（此處為 64、64、32） |
| $I_{\text{tile}}$ | Workgroup 從 L2/HBM 取得的每 byte 所做 FLOP |
| $\beta$ | HBM bandwidth（MI300X 約 5.3 TB/s） |
| $I^{\star}$ | 晶片的 ridge point |

教學 kernel 的 $64\times64$ tile 得到 $I_{\text{tile}} = 32$，只有 ridge point 的八分之一：若沒有 cache 協助，最多只能達峰值約 13%。$256\times256$ tile 得到 128，剩餘差距則由相鄰 tile 間的 L2 與 Infinity Cache reuse 補足。

> 此 kernel 在儲存庫中會為 gfx942 做 compile check，但尚未實際執行：CI 沒有 AMD GPU。在 ROCm 機器上執行
> `hipcc -O3 --offload-arch=gfx942 tutorials/amd/mfma_gemm.hip -o mfma_gemm && ./mfma_gemm 4096 4096 4096`
> 會以 fp64 CPU reference 做內建檢查並印出 TFLOP/s。

## 4. 快速 GEMM 有何不同

### 4.1 各個步驟

下列每個步驟都能在第 06 章的 disassembly 中看到：

| 步驟 | 改變 | 效果 |
|------|--------|--------|
| 1 | **每 wave 更大的 tile**（例如每 wave 16×128 或 64×64，每 workgroup 128×128–256×256） | 每 byte load 對應更多 MFMA；accumulator 增至 128–256 個 register |
| 2 | **Direct-to-LDS load**（`buffer_load_dword … lds`） | Global data 直接寫進 LDS；不需 VGPR staging、`ds_write`，instruction 與 register 都更少 |
| 3 | **Wave 不共享的 operand 略過 LDS** | 若每個 wave 負責 B 的不同 column，可直接將 B 載入 register。Weight 會預先離線 shuffle 成 MFMA operand order（`bpreshuffle`） |
| 4 | **以 counter 做 software pipeline** | 讓 2–3 個 K step 的 load 同時進行，使用 `s_waitcnt vmcnt(N)` 等待，其中 `N` 是之後發出的 load 數，而非 `vmcnt(0)` |
| 5 | **將一切與 MFMA 交錯** | MFMA 會占用 matrix pipeline 數個 cycle。在連續 MFMA 間插入一次 load、LDS read 或 address update，可完全隱藏其 issue cost。Compiler 做得不好，因此最佳 kernel 是 asm 或由程式產生（TensileLite 的 `ScheduleIterAlg`） |
| 6 | 對小 M·N 使用 **Split-K / Stream-K** | Output tile 太少，無法填滿 304 個 CU？沿 K loop 切給多個 workgroup，再用 atomic 或 fix-up pass reduction |
| 7 | **Cache-aware tile order** | 讓同一 XCD 上同時執行的 workgroup，在該 XCD 的 L2 中共享 A/B panel |

### 4.2 填滿機器

第 6 步的重要性可由簡單計數看出。若有 $T$ 個 output tile、$n_{\text{CU}}$ 個 compute unit，且每個 CU 一次執行一個 workgroup：

$$
T = \left\lceil \frac{M}{B_M} \right\rceil\left\lceil \frac{N}{B_N} \right\rceil, \qquad
\text{waves} = \left\lceil \frac{T}{n_{\text{CU}}} \right\rceil, \qquad
\eta_{\text{fill}} = \frac{T}{n_{\text{CU}}\cdot\text{waves}}
$$

| 符號 | 意義 |
|---|---|
| $T$ | Output tile 數（沒有 split-K 時即 workgroup 數） |
| Waves | 涵蓋所有 tile 所需的 workgroup round 數 |
| $\eta_{\text{fill}}$ | 忽略每 tile 不平衡時，CU-time 在做有用工作的比例 |

當 $M = N = 2048$ 且使用 $256\times256$ tile 時，$T = 64$：MI300X 的 304 個 CU 只有 21% 忙碌。若 $T = 320$，需要兩個 wave，第二個只有 5% 滿，因此 $\eta_{\text{fill}} = 53\%$。Split-K 讓 $T$ 乘上 split factor；Stream-K 則將 $T\cdot\lceil K/B_K\rceil$ 次 loop iteration 平分給每個 CU（第 07 章）。

## 5. 實用工具

| 工具 | 用途 |
|------|-----|
| `llvm-objdump -d --mcpu=gfx942 file.co` | 反組譯 code object（`.co`、`.hsaco`） |
| `readelf --notes file.co` | Kernel metadata：VGPR/AGPR/SGPR 數量、LDS 大小、kernarg layout |
| `roc-obj-ls` / `roc-obj-extract` | 從 fat binary / `.so` 取出 code object |
| `rocprofv3 --kernel-trace --stats` | Kernel 時間 |
| `rocprofv3 --pmc SQ_INSTS_VALU_MFMA_MOPS_BF16 …` | Hardware counter（MFMA 使用率、LDS bank conflict `SQ_LDS_BANK_CONFLICT`） |
| rocprofv3 ATT（thread trace）+ Radeon GPU Analyzer / ROCm Compute Viewer | Instruction-level timeline：每個 wave 在何處 stall |
| `rocprof-compute`（Omniperf） | Roofline 與「speed of light」摘要 |

## 重點整理

1. CU 是 SM、wavefront 是 64-lane warp、LDS 是 shared memory；重要差異是 wave64、scalar unit、明確 `s_waitcnt` counter、受 register 限制的 occupancy，以及每 XCD 一份 L2。
2. MFMA 是 operand layout 有文件說明的 wave-level instruction：lane $\ell$ 持有 A（以及 B column）的 row $\ell \bmod 16$ 與 4 個連續 $k$，以及 D 的同一 column 中 4 個 row。
3. 兩個 operand 都讓 K 連續（TN）時，每次 operand fetch 都是每 lane 一次對齊的 8-byte read。
4. 正確的 MFMA GEMM 很容易；快速 GEMM 需要大型 wave tile、direct-to-LDS load、用 counter 做 software pipeline、交錯排程、工作切分，以及 cache-aware tile order：這正是第 06–07 章的主題。

## 練習

1. 將教學 kernel 改用 32×32×8 MFMA（`__builtin_amdgcn_mfma_f32_32x32x8bf16_1k`）。用 Matrix Instruction Calculator 算出新的 operand 與 accumulator layout。

    <details markdown="1"><summary>提示</summary>

    Calculator 的 `--detail-instruction` 與 `--register-layout` 選項（見 `--help`）會印出 CDNA3 instruction 的 register ↔ element table。Accumulator 變成每 lane 16 個 register，而每 lane 持有 A 的某一 row 中 4 個連續 $k$（32 row × 2 組、每組 4 個 $k$，分布於 64 lane）。

    </details>

2. 讓每個 wave 計算 32×64，而非 32×32。Compiler 現在回報幾個 AGPR？Occupancy 有何變化？

    <details markdown="1"><summary>提示</summary>

    Accumulator 從每 lane 16 個 register 加倍為 32 個（2×4 個 tile，每 tile 4 register）。閱讀 `.s` metadata 的 `.agpr_count` 與 `.vgpr_count`，並記得每個 SIMD 有 512 個 register per lane，供其 wave 分享。

    </details>

3. 將 register-staged load 換成 direct-to-LDS load：使用 `__builtin_amdgcn_global_load_lds`（clang 19 或更新版本），或 inline asm。比較 loop 中的 instruction 數量。

4. 對 MI300X 上 $M = 4096$、$N = 1024$、$256\times256$ tile 計算 $\eta_{\text{fill}}$。使用 $S = 4$ 的 split-K 會如何改變結果？

    <details markdown="1"><summary>答案</summary>

    $T = 16\cdot4 = 64$ 個 tile，一個 wave，$\eta = 64/304 = 21\%$。使用 $S = 4$：256 個 workgroup，$\eta = 256/304 = 84\%$（尚未計入合併 partial tile 的成本）。

    </details>
