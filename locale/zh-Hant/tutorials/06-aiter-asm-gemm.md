# 06 – 手寫 AMD GEMM 內部：AITER 的 bf16 Asm Kernel

> **第五部 · AMD 架構與函式庫** · 先備知識：[05 – CDNA3 與 MFMA](05-amd-cdna3-mfma.md) ·
> 下一章：[07 – hipBLASLt 與 TensileLite](07-hipblaslt-tensilelite.md)

[AITER](https://github.com/ROCm/aiter) 是 AMD 用於 LLM inference 的 operator library；vLLM 與 SGLang 在 MI300、MI355 上都會使用它。大多數效能關鍵 kernel 以直接用 GCN/CDNA assembly 撰寫及調校的**預先組譯 code object（`.co`）**發布。本章拆解其中一個：

```
hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co
```

這是 bf16 × bf16 → fp32/bf16 GEMM，使用 128×64 output tile、預先 shuffle 的 weight 與 split-K。它沿用[第 05 章](05-amd-cdna3-mfma.md)的 MFMA 詞彙。文中所有數字都來自 AITER commit `569ae98` 的 disassembly；沒有 GPU 也能重現（見[重現本章](#reproduce-this-chapter)）。

**你將學會**

- GEMM library 如何將呼叫 dispatch 到許多 specialized kernel 之一（tuned table、heuristic、preshuffled weight）；
- 如何閱讀 kernel descriptor，並了解最快的 AMD GEMM 為何讓每個 SIMD 只跑一個肥大的 wave；
- 如何從 register 編號重建 kernel 的工作切分；
- 手寫 kernel main loop 的核心技術：direct-to-LDS load、register double buffering、MFMA interleaving、counter-based pipeline 與無 branch 的 K tail；
- split-K partial 如何合併，以及 AITER bf16 rounding 為何不同於 PyTorch；
- 安全修改及 profile 這類 kernel 的流程。

閱讀時請同時開啟 disassembly；以下每項主張都能對應到其中的 instruction。

## 1. AITER 如何找到並啟動 Kernel

`C = A · Bᵀ`（`nn.Linear`）的 Python call chain 為：

```
aiter.tuned_gemm.gemm_a16w16(A, B, bias)          # aiter/tuned_gemm.py
  └─ get_GEMM_A16W16_config(M, N, K, …)           # lookup in aiter/configs/bf16_tuned_gemm.csv
       └─ libtype ∈ {asm, hipblaslt, triton, skinny, opus, flydsl, torch}
  └─ solMap["asm"] → gemm_a16w16_asm(…)            # aiter/ops/gemm_op_a16w16.py
       └─ C++: csrc/py_itfs_cu/asm_gemm_a16w16.cu  # picks a .co, fills KernelArgs, hipModuleLaunchKernel
```

### 1.1 Tuned Table

`aiter/configs/bf16_tuned_gemm.csv` 對每個 `(gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)` key 保存一列。每列記錄：
- 勝出的 `libtype`；
- `solidx`、`splitK` 與 `kernelName`；
- 測得的 `us` 與 `tflops`。

它由 `csrc/gemm_a16w16/gemm_a16w16_tune.py` 離線產生，會對每個 shape benchmark 每種 backend。在固定 commit 中，232 列分成：120 triton、71 asm、36 opus、5 flydsl。

換句話說，手寫 kernel 只對特定 shape 勝出。實務教訓是：**GEMM library 就是一張 dispatch table，加上一整座 specialized kernel 動物園。** 第 07 章會看到 hipBLASLt 以更大規模做相同事情。

### 1.2 Kernel Table

`hsa/gfx942/bf16gemm/bf16gemm_fp32bf16.csv` 列出 family 中每個 `.co` 的 property：

| co_name | tn | tileM | tileN | pf | bPreshuffle | splitK | subK | bias |
|---------|----|-------|-------|----|-------------|--------|------|------|
| `…_128x64_bshuffle_splitk.co` | 1 | 128 | 64 | 0 | 1 | 0 | 64 | 1 |
| `…_32x64_pf3_splitk.co` | 1 | 32 | 64 | 3 | 0 | 0 | 64 | 1 |
| `…_64x64_splitk_clean.co` | 1 | 64 | 64 | 0 | 0 | 1 | 64 | 1 |

此 family 有 22 個 kernel：tileM ∈ {32, 48, 64, 80, 96, 128, 160}，並各有使用與不使用 preshuffled B 的版本。

### 1.3 Heuristic

沒有相符 tuned row 時，C++ launcher 的 `get_heuristic_kernel` 會：

1. 依 `N % tileN == 0`、preshuffle flag 與 bias support 篩選 kernel。
2. 對支援 split-K 的 kernel，選擇 `splitK = max(2, min(num_cu / tiles, 16, K / subK))`。
3. 將 CU 上 **workgroup wave**（「round」）數量最小化。
4. 平手時，依序選最後一 round idle CU 較少、M padding 較少，以及 `tileM·tileN / (tileM + tileN)`（compute-to-memory ratio）較高者。

這就是人類所用的「先填滿機器，再最大化 reuse」推理。

### 1.4 Argument 與 Launch

`KernelArgs` 是 packed struct，**每個 field 都 padding 到 16 byte**：`ptr_D`、`ptr_C`、`ptr_A`、`ptr_B`、`alpha`、`beta`、stride、`M`、`N`、`K`、`splitk`、`is_out_b16`、`ptr_Bias`、`add_bias`、`ptr_semaphore`。因此 prologue 會從 offset `0x0, 0x10, 0x20, …` 讀取：

```asm
s_load_dwordx2 s[16:17], s[0:1], 0x0     ; ptr_D
s_load_dwordx2 s[4:5],   s[0:1], 0x20    ; ptr_A  -> becomes buffer resource s[4:7]
s_load_dwordx2 s[8:9],   s[0:1], 0x30    ; ptr_B  -> buffer resource s[8:11]
s_load_dword   s25,      s[0:1], 0xe0    ; M
s_load_dword   s26,      s[0:1], 0xf0    ; N
s_load_dword   s27,      s[0:1], 0x100   ; K
s_load_dword   s48,      s[0:1], 0x110   ; splitk
```

Launch grid 是 `(ceil(N/64), ceil(M/128), splitK)`，每 workgroup 256 個 thread。

### 1.5 Preshuffled Weight

`aiter.ops.shuffle.shuffle_weight(w, layout=(16, 16))` 會**離線、只做一次** weight permutation：

```python
# bf16: 16 rows (N) x 32 cols (K) blocks, each stored as [k_chunk(4)][n(16)][8 elems]
x.view(-1, N // 16, 16, K // 32, 4, 8).permute(0, 1, 3, 4, 2, 5)
```

Permutation 後，每 lane 一個 `buffer_load_dwordx4`（64 lane × 16 byte = 一個 16×32 block）會**恰好**送入 MFMA 的 B-operand fragment：lane `l` 得到 column `l % 16` 與 8 個連續 k 值。第 3 節會說明其重要性。

以 index map 表示：將 weight coordinate 拆成 $n = 16n_1 + n_0$ 與 $k = 32k_1 + 8k_2 + k_3$。Permutation 將 element $(n, k)$ 存於

$$
\pi(n, k) = \Bigl(\bigl(n_1\,\tfrac{K}{32} + k_1\bigr)\cdot 4 + k_2\Bigr)\cdot 128 + 8\,n_0 + k_3, \qquad
\ell = 16\,k_2 + n_0
$$

| 符號 | 意義 |
|---|---|
| $n_1, n_0$ | $W$ 的 16-row block，以及 block 內的 row（$0 \le n_0 < 16$） |
| $k_1, k_2, k_3$ | 32-wide K block、其中的 8-element chunk（$0 \le k_2 < 4$），以及 chunk 內 element |
| $\pi(n, k)$ | Shuffled buffer 中的 element offset |
| $\ell$ | Wave 連續讀取 1 KiB（每 lane 16 byte）時接收該 element 的 lane |

每個連續 1 KiB chunk 都是一個 $16\times32$ block；lane $\ell$ 取得 row $n_0 = \ell \bmod 16$ 與 8 個連續 $k$，也就是兩個連續 16x16x16 instruction 的 MFMA B-operand layout。

## 2. Kernel Descriptor：每個 SIMD 一個肥大的 Wave

```
$ llvm-objdump -D -j .rodata --mcpu=gfx942 bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co
	.amdhsa_group_segment_fixed_size 65536     ; all 64 KiB of LDS
	.amdhsa_accum_offset 256                   ; v0-v255 arch VGPRs, a0-a255 AGPRs
	.amdhsa_next_free_vgpr 512                 ; 512 registers per lane: the whole file
	.amdhsa_next_free_sgpr 112
	.amdhsa_ieee_mode 0
	.amdhsa_dx10_clamp 0
```

256 個 thread（4 waves）、每 lane 512 個 register、64 KiB LDS，代表：
- 每個 CU **恰好只容納一個 workgroup**；
- 該 workgroup 在**每個 SIMD 上恰好一個 wave**。

除了 kernel 自己的 instruction schedule，沒有其他東西能隱藏 latency。這與簡單 kernel 的「最大化 occupancy」建議相反，卻是 CDNA 上 peak-performance GEMM 的常態。

## 3. 工作切分：為何 B 完全不碰 LDS

### 3.1 閱讀 Register 編號

Main loop 中的 MFMA 如下：

```asm
v_mfma_f32_16x16x16_bf16 v[44:47], a[128:129], a[0:1],  v[44:47]
v_mfma_f32_16x16x16_bf16 v[48:51], a[128:129], a[8:9],  v[48:51]
...
v_mfma_f32_16x16x16_bf16 v[72:75], a[128:129], a[56:57], v[72:75]
```

從 register 編號可讀出：

- **Accumulator** 是 `v[44:75]`：8 個 tile × 4 register。罕見的是它們位於 *arch VGPR*，不是 AGPR。
- **第一個 operand** `a[128:135]`（4 個 k-step × 2 register）對 8 個 accumulator 都相同。它是一條 16-wide strip，來自 **B**。
- **第二個 operand** `a[0:63]` 會隨 accumulator 改變：8 個不同的 16-row **A** block，合計 128 row。

### 3.2 切分及其結果

因此每個 wave 計算 output 的一塊 **16(N) × 128(M)** slab，而四個 wave 沿 N 切分：4 × 16 = 64 = tileN。這帶來兩個結果：

1. **每個 K step 的 A（128 × 64）供四個 wave 共用**，所以經過 LDS。
2. **B 為每個 wave 私有**，沒有其他 wave 使用它的 16 個 column。經 LDS staging 只會增加 instruction 與 bandwidth；每個 wave 直接將自己的 B strip **從 global memory 載入 AGPR**：

   ```asm
   buffer_load_dwordx4 a[144:147], v38, s[8:11], 0 offen   ; B, 16 bytes per lane
   buffer_load_dwordx4 a[148:151], v39, s[8:11], 0 offen
   ```

   只有 preshuffled weight 才能做到（第 1 節）：每 lane 的 16 byte 已是 MFMA operand fragment。名稱中的 `bshuffle` 就是此意。未 preshuffle 的 `pf3` variant 會改經 LDS，並使用更深 prefetch。

   MFMA 每次只取 4 個 k 值，lane 卻能接收 8 個，為何仍合法？只要 A、B 對 k index 套用相同 permutation，`Σₖ aₖ·bₖ` 就不變。Shuffle 選擇能讓 load 連續的 k 順序。

![128 × 64 tile：A 在 LDS staging 供四個 wave 共用；每個 wave 將自己的 B strip 直接載入 AGPR](figures/ch06-decomposition.svg)

### 3.3 為何可以重新排列 k

對 $\{0, \dots, K-1\}$ 的任何 permutation $\sigma$：

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik}\,B_{kj} = \sum_{k=0}^{K-1} A_{i\sigma(k)}\,B_{\sigma(k)j}
$$

| 符號 | 意義 |
|---|---|
| $\sigma$ | 同樣套用到兩個 operand 的 reduction index 重新排序 |

唯一要求是從 LDS 讀取的 A fragment 使用與 preshuffled B 相同的 $\sigma$；kernel 的 LDS read offset 保證了這點。

## 4. Main Loop

Steady state 是對每個 K=64 step 重複的 block。

### 4.1 每 Block 的 Instruction Budget

由 disassembly 計數：

| 每 wave、每 K=64 step | 數量 | Byte |
|-------------------------|-------|-------|
| `v_mfma_f32_16x16x16_bf16` | 32 | – |
| `buffer_load_dword … offen lds`（A，direct-to-LDS） | 16 | 16 × 64 × 4 = 4 KiB |
| 載入 AGPR 的 `buffer_load_dwordx4`（B） | 2 | 2 KiB |
| 載入 AGPR 的 `ds_read_b128`（下一 step 的 A operand） | 16 | 16 KiB |
| `s_barrier` | 1 | – |

四個 wave 合計將完整 128×64 A tile（16 KiB）載入 LDS，各自載入 64×64 B column（合計 8 KiB），接著發出 128 個 MFMA：
128 × 16·16·16 × 2 = 每 24 KiB fetch 做 1 MFLOP。第 05 章教學 kernel 每 2 個 global load、4 個 LDS read、2 個 LDS write 只發出 8 個 MFMA。

$128\times64$ workgroup tile、K step=64（bf16，每值 2 byte）的 reuse 為：

$$
I_{\text{tile}} = \frac{2\,B_MB_NB_K}{2\,(B_M + B_N)\,B_K} = \frac{128\cdot64}{128 + 64} \approx 42.7\ \frac{\text{flop}}{\text{byte}}, \qquad
\rho = \frac{n_{\text{MFMA}}}{n_{\text{mem}}} = \frac{32}{16 + 2 + 16} \approx 0.94
$$

| 符號 | 意義 |
|---|---|
| $B_M, B_N, B_K$ | Workgroup tile：128（M）、64（N）、64（每 step 的 K） |
| $I_{\text{tile}}$ | 每個從 L2/HBM fetch 的 byte 所做 FLOP |
| $n_{\text{MFMA}}$ | 每 wave、每 K step 的 MFMA 數 |
| $n_{\text{mem}}$ | 每 wave、每 K step 的 memory instruction 數（buffer load 加 LDS read） |
| $\rho$ | 每 memory instruction 的 MFMA 數：約為一，因此每條 memory instruction 都能藏在 MFMA 的 shadow 中 |

第 05 章教學 kernel 也有 $\rho = 8/8 = 1$，但每個 MFMA 周圍都有 address arithmetic、wait 與 barrier；此處整個 loop body 的 schedule 能讓 matrix pipeline 不閒置。

### 4.2 Direct-to-LDS Load

```asm
s_add_u32 m0, 0x100, s42                     ; LDS destination = M0 (+ lane * 4)
buffer_load_dword v22, s[4:7], 0 offen lds   ; global A[...] -> LDS, bypassing VGPRs
```

使用 `lds` modifier 時，每 lane 的 dword 會寫到 LDS 的 `M0 + lane·4`，而非 `v22`（此處 `v22` 只是 address offset）。每條 instruction 搬 256 byte，所以 `M0` 每次增加 `0x100`。`s42`/`s43` 在 iteration 間交替，代表兩個 LDS buffer。

Loop 中完全沒有 `ds_write`。每 step 因此少 16 條 instruction，也省下 staging data 所需 VGPR。

### 4.3 Register Double Buffering

```asm
v_mfma_f32_16x16x16_bf16 v[44:47], a[128:129], a[0:1], v[44:47]   ; compute with a[0:63]...
ds_read_b128 a[64:67], v37 offset:16512                            ; ...while loading a[64:127]
```

Step *k* 的 MFMA 從 `a[0:63]` 讀 A fragment，同時給 step *k+1* 的 `ds_read` 填入 `a[64:127]`；下一個 block 交換角色。B 也在 `a[128:135]`、`a[136:143]`、`a[144:151]` 間輪替。

AMD GPU 無法動態 indexing register，因此 rotation 會在程式中**展開**：
- `label_02CA` 與 `label_0703` 兩個 loop body，各有 6 個 unrolled block；
- 每個 body 有 192 個 MFMA、96 個 direct-to-LDS load 與 96 個 `ds_read_b128`。

整份 disassembly 合計 384 個 MFMA、240 個 direct-to-LDS load 與 208 個 `ds_read_b128`。

### 4.4 Interleaving

觀察一個 block 的開頭：

```asm
s_waitcnt vmcnt(18) lgkmcnt(0)      ; previous step's loads have landed
s_barrier                           ; every wave's A slice is in LDS
v_mfma_f32_16x16x16_bf16 ...
s_add_u32 m0, 0, s42
buffer_load_dword v21, s[4:7], 0 offen lds
v_mfma_f32_16x16x16_bf16 ...
s_add_u32 m0, 0x100, s42
buffer_load_dword v22, s[4:7], 0 offen lds
ds_read_b128 a[64:67], v37 offset:16512
ds_read_b128 a[68:71], v37 offset:16576
v_mfma_f32_16x16x16_bf16 ...
```

模式是反覆執行 **MFMA，再接 1–3 條 memory 或 scalar instruction**。MFMA issue 後仍會讓 matrix core 忙碌數個 cycle；在此期間，wave instruction arbiter 能免費 issue load、`M0` update 或 pointer increment。

![MFMA 讓 matrix core 保持忙碌，同時在其間 issue load、LDS read 與 scalar update](figures/ch06-interleave.svg)

這是 AMD GEMM 最重要的 scheduling 概念，也是 TensileLite `ScheduleIterAlg=3` 自動完成的工作（第 07 章）。

### 4.5 計算 Outstanding Load

`s_waitcnt vmcnt(18)` 表示「最多剩 18 個 vector-memory operation 尚未完成時繼續」。每個 block 發出 16 + 2 = 18 個。Vector memory operation 依序完成，因此 block *k+1* 頂端的 `vmcnt(18)` 會等待 block *k-1* 及更早的所有工作，讓 block *k* 的 load 繼續飛行。這是用單一 counter 表達的**完整一個 block prefetch**，不需額外 register 或 branch。

### 4.6 無 Branch 的 K Tail

```asm
s_add_u32 s31, 0x100, s33
s_cmp_lt_u32 s31, s34
s_cselect_b32 s40, s40, 0     ; pointer increment becomes 0 past the end of K
s_add_u32 s4, s40, s4         ; advance A's buffer base
```

接近 K 結尾時，pointer increment 會設為零，讓 prefetch 安全地重讀最後一個 tile，而非越界讀取。Loop body 因此不需 branch。

Buffer resource（`s[4:7]`、`s[8:11]`）也能由硬體做 bounds check：越界 load 回傳 0，越界 store 會丟棄。TensileLite 等 generated kernel 依賴此行為處理 edge tile。此 kernel 在 prologue 將 `num_records` 設為 `0xFFFFFFF0`（`s_mov_b32 s6, -16`），實際上停用了檢查。

## 5. Epilogue：Split-K 與 bf16 Rounding { #5-epilogue-split-k-and-bf16-rounding }

### 5.1 合併 Partial Tile

`splitk > 1` 時，grid 的 z dimension 會切分 K。每個 z-slice 累積一個 partial 128×64 tile：

- **fp32 output：** 使用 `global_atomic_add_f32` 加上 partial（listing 中 64 個）。
- **bf16 output：** 使用 `global_atomic_pk_add_bf16` 將 partial rounding 後相加，每次 atomic 兩個 bf16（listing 中 48 個）。
- Listing 另有 16 個普通 store，供 non-split path 使用。

小型 **semaphore workspace**（`ptr_semaphore`，16 × 64 `uint32`，在 `gemm_op_a16w16.py` 中依 stream 初始化為零）保存每 tile arrival counter。最後抵達的 workgroup 執行 final phase 並重設 counter。因此 launcher 會 assert `gdx·gdy ≤ 1024`，不同 stream 也需要各自 workspace：共用 counter 會 deadlock。

![Split-K：每個 z-slice 以 atomic 加上 partial tile；per-tile counter 選出最後抵達者](figures/ch06-split-k.svg)

### 5.2 手工 bf16 Rounding

fp32 → bf16 conversion 是手工完成：

```asm
v_cmp_u_f32_e64 s[56:57], v44, v44     ; NaN?
v_add3_u32 v8, v44, v11, 1             ; bits + 0x7fff + 1  (v11 = 0x7fff)
v_cndmask_b32_e64 v4, v8, v10, s[56:57] ; NaN -> 0x7fff0000 (canonical qNaN)
v_perm_b32 v76, v5, v4, s35            ; s35 = 0x07060302: pack the two high halves
```

加上 `0x8000` 再截斷，會將 tie 依 magnitude **遠離零** rounding。這**不是 round-to-nearest-even**；torch 的 `.to(torch.bfloat16)` 加的是 `0x7fff + lsb`。剛好 tie 時結果可能相差 1 ulp，因此逐 bit 比對 reference 時很重要。

對 fp32 值的 32-bit pattern $u$，兩種規則為：

$$
\operatorname{bf16}_{\text{RNE}}(u) = \Bigl\lfloor \frac{u + \texttt{0x7FFF} + \bigl(\lfloor u / 2^{16} \rfloor \bmod 2\bigr)}{2^{16}} \Bigr\rfloor, \qquad
\operatorname{bf16}_{\text{AITER}}(u) = \Bigl\lfloor \frac{u + \texttt{0x8000}}{2^{16}} \Bigr\rfloor
$$

| 符號 | 意義 |
|---|---|
| $u$ | 視為 unsigned integer 的 fp32 bit pattern（NaN 另行處理） |
| $\lfloor u/2^{16}\rfloor \bmod 2$ | 保留部分的最低 bit（bf16 mantissa LSB） |
| $\operatorname{bf16}_{\text{RNE}}$ | Round to nearest、tie to even（PyTorch） |
| $\operatorname{bf16}_{\text{AITER}}$ | Round to nearest、tie away from zero（此 kernel） |

兩者只在低 16 bit 恰為 `0x8000` 且保留的 LSB 為 0 時不同。

### 5.3 Split-K Arithmetic

Split-K 就是對不同 K range 求和：

$$
C = \sum_{z=0}^{S-1} A_{:,\,\mathcal{K}_z}\,B_{\mathcal{K}_z,\,:}, \qquad
\mathcal{K}_z = \Bigl[\,z\,\tfrac{K}{S},\ (z+1)\,\tfrac{K}{S}\Bigr)
$$

| 符號 | 意義 |
|---|---|
| $S$ | Split factor（`splitk`，grid 的 z extent） |
| $\mathcal{K}_z$ | Slice $z$ 處理的 K range |

它讓 workgroup 數乘以 $S$，代價是每 tile 有 $S$ 個 partial result（atomic），且 fp32 summation order 不具 deterministic。

## 6. 修改與 Profile 這類 Kernel

AITER 在 `docs/isa_kernel_optimization.md` 記錄完整流程，script 位於 `docs/examples/isa_optimization/`：

1. **先 round-trip。** `roundtrip.sh <kernel.co>` 會擷取獨立的 `kernel.s`，以 `clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942` 重組，並比較 `.text`、kernel descriptor 與 metadata note。之後的任何差異才是你造成的。
2. **編輯。** 手寫 asm 沒有工具替你檢查 hazard：
   - Assembler 會原樣 encode。
   - Hazard `s_nop` 是必要 wait state（例如 dependent MFMA 之間或 transcendental op 後）。`s_nop N` 提供 N+1 個 wait state。
   - 搬移 load 後，必須重算每個 `s_waitcnt`。
   - 違規不會 fault，而會默默讀到 stale data。
3. **調整大小。** 改變 register 或 LDS usage 時，要同時編輯 `.amdhsa_next_free_vgpr`、`.amdhsa_accum_offset`、`.amdhsa_group_segment_fixed_size` 與 metadata。
4. **測試。** 替換 `hsa/gfx942/…` 中的 `.co` 並執行 op test。AITER 會記錄 `LoadKernel: … hsaco: <path>`。
5. **Profile。**
   - `rocprofv3 --kernel-trace --stats --kernel-include-regex bf16gemm` 做計時。
   - `rocprofv3 --att --kernel-iteration-range 5-5 --att-target-cu 1` 取得 per-instruction thread trace，再用 ROCprof Compute Viewer 查看。它會顯示 loop 受限於 MFMA issue、`s_waitcnt`，還是 LDS bank conflict。

## 重現本章 { #reproduce-this-chapter }

不需要 GPU 或 ROCm；Ubuntu 的 LLVM 18 package 即可。

```bash
git clone --filter=blob:none https://github.com/ROCm/aiter && cd aiter
CO=hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co
llvm-objdump-18 -d --mcpu=gfx942 $CO > gemm.isa
grep -c v_mfma gemm.isa                   # 384
grep -c 'offen lds' gemm.isa              # 240
llvm-readelf-18 --notes $CO | grep -E 'vgpr_count|group_segment|wavefront'

# Round trip with a stock LLVM: point ROCM_PATH at a directory whose llvm/bin
# holds (links to) clang, llvm-objdump, llvm-readelf, llvm-readobj, llvm-objcopy, ld.lld.
mkdir -p /tmp/rocm/llvm/bin
for t in clang llvm-objdump llvm-readelf llvm-readobj llvm-objcopy llvm-mc ld.lld; do
  ln -sf "$(command -v $t-18 || command -v $t)" /tmp/rocm/llvm/bin/$t; done
ROCM_PATH=/tmp/rocm bash docs/examples/isa_optimization/roundtrip.sh $CO
```

使用 LLVM 18 時，round trip 會回報 `.text`、kernel descriptor 與 metadata **完全相同**。Script 也會印出 `e_flags` 的「DIFFERS」，但兩側值都是 `0x54C`；這是 script 比較 LLVM 18 output 的小問題，不是真正差異。

## 重點整理

1. Production GEMM 是 specialized kernel 的 dispatch table；手寫版本只對部分 shape 勝出。
2. 最快的 CDNA GEMM 用完 512 個 register 與 64 KiB LDS：每 SIMD 一個 wave，以 instruction schedule 而非其他 wave 隱藏 latency。
3. 多個 wave 共用的 operand 經 LDS；wave 私有的 operand（preshuffled weight）直接進 register。
4. Main loop 在每個 MFMA 間交錯 1–3 條 memory 或 scalar instruction，以 register double-buffer operand，並用單一 `s_waitcnt vmcnt(N)` pipeline global load。
5. Split-K 用 atomic 加上 partial tile（fp32 中不具 deterministic）；手工 bf16 rounding 在 tie 時可能與 PyTorch 相差 1 ulp。
6. 只有在 byte-identical round trip 後才修改這類 kernel；沒有工具會替你檢查 hazard。

## 練習

1. 反組譯 `bf16gemm_fp32bf16_tn_32x64_pf3_splitk.co`（無 preshuffle）。
   - B 現在經過哪裡？
   - Main loop 的 `vmcnt` 是多少？它提供多少個 K step 的 prefetch（名稱中的 `pf3`）？
2. 由 MFMA 數與每 MFMA 16×16×16 計算每個 `.co` loop body 的 FLOP。
   - 提示：MI300X 的 1307 TFLOP/s dense bf16 峰值 ÷（304 CU × 4 SIMD × 2.1 GHz）≈ 512 FLOP/cycle/SIMD，因此一個 16x16x16 MFMA（8192 FLOP）約需 16 cycle。
   - 每個 K step 給每個 wave 多少 cycle 的 MFMA 工作？
   - MFMA 間必須容納多少 non-MFMA instruction？

    <details markdown="1"><summary>答案</summary>

    每個 wave 在每個 K=64 step 發出 32 個 MFMA（第 4.1 節），約有 $32 \times 16 = 512$ cycle 的 matrix-core 工作，以及 34 條 memory instruction（16 + 2 + 16），另有 scalar 與 address update：大約每個 MFMA 一條其他 instruction，正是第 4.4 節的 interleaving pattern。一個 loop body（6 block）有 192 個 MFMA：$192 \times 8192 \approx 1.57$ MFLOP per wave。

    </details>
3. 寫出 `shuffle_weight(layout=(16,16))` 在 16×32 block 內套用的確切 k permutation。確認 A-side LDS layout 必須使用相同 permutation。
