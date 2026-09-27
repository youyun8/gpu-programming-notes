# 06 – 拆解手寫的 AMD GEMM：AITER 的 bf16 組合語言 kernel

> **第五部 · AMD 架構與函式庫** · 先備知識：[05 – CDNA3 與 MFMA](05-amd-cdna3-mfma.md) ·
> 下一章：[07 – hipBLASLt 與 TensileLite](07-hipblaslt-tensilelite.md)

[AITER](https://github.com/ROCm/aiter) 是 AMD 為 LLM 推論打造的運算子函式庫，vLLM 與 SGLang 在 MI300 與 MI355 上都會使用它。它大部分對效能最關鍵的 kernel，都是以**預先組譯好的程式碼物件（`.co`）**形式發布，直接用 GCN/CDNA 組合語言撰寫並調校。本章要拆解其中一個：

```
hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co
```

這是一個 bf16 × bf16 → fp32/bf16 的 GEMM，輸出分塊為 128×64，使用預先重排的權重與 split-K，並沿用[第 05 章](05-amd-cdna3-mfma.md)的 MFMA 術語。文中引用的每個數字，都是在 AITER commit `569ae98` 的反組譯結果上實際量到的。整個過程不需要 GPU 就能重現（見[重現本章](#reproduce-this-chapter)）。

**你將學到**

- GEMM 函式庫如何把一次呼叫分派給眾多專用 kernel 中的一個（調校表、啟發式規則、預先重排的權重）；
- 如何閱讀 kernel descriptor，並理解為什麼最快的 AMD GEMM 每個 SIMD 只跑一個「肥大」的 wave；
- 如何從暫存器編號反推 kernel 的工作切分；
- 手寫 kernel 主迴圈的技巧：直接載入 LDS、暫存器雙緩衝、在 MFMA 之間穿插指令、以計數器做管線化，以及不含分支的 K 尾端處理；
- split-K 的部分結果如何合併，以及 AITER 的 bf16 捨入為何與 PyTorch 不同；
- 安全地修改與分析這類 kernel 的工作流程。

閱讀時請把反組譯結果一起打開：以下每個論點都指向其中找得到的指令。

## 1. AITER 如何找到並啟動 kernel

`C = A · Bᵀ`（也就是 `nn.Linear`）在 Python 端的呼叫鏈如下：

```
aiter.tuned_gemm.gemm_a16w16(A, B, bias)          # aiter/tuned_gemm.py
  └─ get_GEMM_A16W16_config(M, N, K, …)           # lookup in aiter/configs/bf16_tuned_gemm.csv
       └─ libtype ∈ {asm, hipblaslt, triton, skinny, opus, flydsl, torch}
  └─ solMap["asm"] → gemm_a16w16_asm(…)            # aiter/ops/gemm_op_a16w16.py
       └─ C++: csrc/py_itfs_cu/asm_gemm_a16w16.cu  # picks a .co, fills KernelArgs, hipModuleLaunchKernel
```

### 1.1 調校表

`aiter/configs/bf16_tuned_gemm.csv` 以 `(gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)` 為鍵，每個鍵一列，記錄：
- 勝出的 `libtype`；
- `solidx`、`splitK` 與 `kernelName`；
- 實測的 `us` 與 `tflops`。

這張表由 `csrc/gemm_a16w16/gemm_a16w16_tune.py` 離線產生：它會針對每種形狀，對每個後端做基準測試。在本章固定的 commit 中，232 列的分布是：triton 120、asm 71、opus 36、flydsl 5。

換句話說，手寫 kernel 只在特定形狀上勝出。實務上的結論是：**GEMM 函式庫 = 一張分派表 + 一大群專用 kernel。** 第 07 章會看到 hipBLASLt 以更大的規模做同樣的事。

### 1.2 kernel 清單

`hsa/gfx942/bf16gemm/bf16gemm_fp32bf16.csv` 列出這個家族中的每個 `.co` 及其屬性：

| co_name | tn | tileM | tileN | pf | bPreshuffle | splitK | subK | bias |
|---------|----|-------|-------|----|-------------|--------|------|------|
| `…_128x64_bshuffle_splitk.co` | 1 | 128 | 64 | 0 | 1 | 0 | 64 | 1 |
| `…_32x64_pf3_splitk.co` | 1 | 32 | 64 | 3 | 0 | 0 | 64 | 1 |
| `…_64x64_splitk_clean.co` | 1 | 64 | 64 | 0 | 0 | 1 | 64 | 1 |

這個家族共有 22 個 kernel：tileM ∈ {32, 48, 64, 80, 96, 128, 160}，各有使用與不使用預先重排 B 的版本。

### 1.3 啟發式規則

當調校表中沒有符合的列時，C++ 啟動程式中的 `get_heuristic_kernel` 會：

1. 依 `N % tileN == 0`、預先重排旗標與偏差支援篩選 kernel。
2. 對支援 split-K 的 kernel，選擇 `splitK = max(2, min(num_cu / tiles, 16, K / subK))`。
3. 讓 workgroup 在各 CU 上需要的**輪數**（rounds）最少。
4. 若仍平手，優先選擇最後一輪閒置 CU 較少、M 方向補齊較少，以及 `tileM·tileN / (tileM + tileN)`（運算對記憶體比）較高的 kernel。

這正是人類會採用的推理：「先填滿整台機器，再盡量提高重用率」。

### 1.4 參數與啟動

`KernelArgs` 是一個緊密排列的結構，但**每個欄位都補齊到 16 位元組**：`ptr_D`、`ptr_C`、`ptr_A`、`ptr_B`、`alpha`、`beta`、各步幅、`M`、`N`、`K`、`splitk`、`is_out_b16`、`ptr_Bias`、`add_bias`、`ptr_semaphore`。這就是為什麼 prologue 在位移 `0x0, 0x10, 0x20, …` 讀取它們：

```asm
s_load_dwordx2 s[16:17], s[0:1], 0x0     ; ptr_D
s_load_dwordx2 s[4:5],   s[0:1], 0x20    ; ptr_A  -> becomes buffer resource s[4:7]
s_load_dwordx2 s[8:9],   s[0:1], 0x30    ; ptr_B  -> buffer resource s[8:11]
s_load_dword   s25,      s[0:1], 0xe0    ; M
s_load_dword   s26,      s[0:1], 0xf0    ; N
s_load_dword   s27,      s[0:1], 0x100   ; K
s_load_dword   s48,      s[0:1], 0x110   ; splitk
```

啟動的網格為 `(ceil(N/64), ceil(M/128), splitK)`，每個 workgroup 256 個執行緒。

### 1.5 預先重排的權重

`aiter.ops.shuffle.shuffle_weight(w, layout=(16, 16))` 會**離線、一次性地**重新排列權重：

```python
# bf16: 16 rows (N) x 32 cols (K) blocks, each stored as [k_chunk(4)][n(16)][8 elems]
x.view(-1, N // 16, 16, K // 32, 4, 8).permute(0, 1, 3, 4, 2, 5)
```

重排之後，每個 lane 一次 `buffer_load_dwordx4`（64 個 lane × 16 位元組 = 一個 16×32 區塊）就能*恰好*取得 MFMA 的 B 運算元片段：lane `l` 拿到第 `l % 16` 欄與 8 個連續的 k 值。第 3 節會說明這為何重要。

寫成索引映射：把權重座標拆成 $n = 16n_1 + n_0$ 與 $k = 32k_1 + 8k_2 + k_3$，重排後元素 $(n, k)$ 存放在

$$
\pi(n, k) = \Bigl(\bigl(n_1\,\tfrac{K}{32} + k_1\bigr)\cdot 4 + k_2\Bigr)\cdot 128 + 8\,n_0 + k_3, \qquad
\ell = 16\,k_2 + n_0
$$

| 符號 | 意義 |
|---|---|
| $n_1, n_0$ | $W$ 的 16 列區塊，以及區塊內的列（$0 \le n_0 < 16$） |
| $k_1, k_2, k_3$ | 寬 32 的 K 區塊、區塊內 8 個元素的小段（$0 \le k_2 < 4$）、小段內的元素 |
| $\pi(n, k)$ | 元素在重排後緩衝區中的位移 |
| $\ell$ | 一個 wave 連續讀取 1 KiB（每個 lane 16 位元組）時，拿到該元素的 lane |

每段連續的 1 KiB 就是一個 $16\times32$ 區塊，lane $\ell$ 拿到第 $n_0 = \ell \bmod 16$ 列上 8 個連續的 $k$：正好是兩道連續 16x16x16 指令所需的 MFMA B 運算元配置。

## 2. Kernel descriptor：每個 SIMD 一個肥大的 wave

```
$ llvm-objdump -D -j .rodata --mcpu=gfx942 bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co
	.amdhsa_group_segment_fixed_size 65536     ; all 64 KiB of LDS
	.amdhsa_accum_offset 256                   ; v0-v255 arch VGPRs, a0-a255 AGPRs
	.amdhsa_next_free_vgpr 512                 ; 512 registers per lane: the whole file
	.amdhsa_next_free_sgpr 112
	.amdhsa_ieee_mode 0
	.amdhsa_dx10_clamp 0
```

256 個執行緒（4 個 wave）、每個 lane 512 個暫存器、64 KiB 的 LDS 意味著：
- 每個 CU **恰好只能放一個 workgroup**；
- 這個 workgroup 在**每個 SIMD 上只有一個 wave**。

除了 kernel 自己的指令排程之外，沒有任何東西能隱藏延遲。這與簡單 kernel「盡量提高佔用率」的建議正好相反，但在 CDNA 上追求峰值效能的 GEMM 中是常態。

## 3. 工作切分：為什麼 B 完全不經過 LDS

### 3.1 閱讀暫存器編號

主迴圈中的 MFMA 長這樣：

```asm
v_mfma_f32_16x16x16_bf16 v[44:47], a[128:129], a[0:1],  v[44:47]
v_mfma_f32_16x16x16_bf16 v[48:51], a[128:129], a[8:9],  v[48:51]
...
v_mfma_f32_16x16x16_bf16 v[72:75], a[128:129], a[56:57], v[72:75]
```

從暫存器編號可以讀出：

- **累加器**是 `v[44:75]`：8 個分塊 × 4 個暫存器。少見的是，它們放在*一般的 VGPR*，而不是 AGPR。
- **第一個運算元** `a[128:135]`（4 個 k 步 × 2 個暫存器）對 8 個累加器都相同：它是一條寬 16 的長條，來自 **B**。
- **第二個運算元** `a[0:63]` 隨累加器而變：共有 8 個不同的 16 列 **A** 區塊，合計 128 列。

### 3.2 切分方式與其結果

因此每個 wave 計算輸出中 **16（N）× 128（M）** 的一片，四個 wave 沿 N 分工：4 × 16 = 64 = tileN。這帶來兩個結果：

1. **A（每一步 K 為 128 × 64）四個 wave 都需要**，所以要經過 LDS。
2. **B 是各 wave 私有的**：它的 16 欄沒有其他 wave 會用到。把 B 經由 LDS 中轉只會浪費指令與頻寬，所以每個 wave 把自己的 B 長條**直接從全域記憶體載入 AGPR**：

   ```asm
   buffer_load_dwordx4 a[144:147], v38, s[8:11], 0 offen   ; B, 16 bytes per lane
   buffer_load_dwordx4 a[148:151], v39, s[8:11], 0 offen
   ```

   這之所以可行，是因為權重已經預先重排過（第 1 節）：每個 lane 拿到的 16 位元組本身就是 MFMA 的運算元片段，名稱中的 `bshuffle` 就是這個意思。沒有預先重排的 `pf3` 版本則改走 LDS，並使用更深的預先載入。

   MFMA 一次只吃 4 個 k，為什麼一個 lane 拿到 8 個 k 值也沒問題？只要對 A 與 B 一致地套用同一個 k 的排列，`Σₖ aₖ·bₖ` 就不會改變。重排所選的 k 順序，正是讓載入保持連續的那一種。

![128 × 64 分塊：A 暫存在 LDS 供四個 wave 共用，每個 wave 則把自己的 B 長條直接載入 AGPR](figures/ch06-decomposition.svg)

### 3.3 為什麼可以重新排列 k

以符號表示，對 $\{0, \dots, K-1\}$ 的任何排列 $\sigma$：

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik}\,B_{kj} = \sum_{k=0}^{K-1} A_{i\sigma(k)}\,B_{\sigma(k)j}
$$

| 符號 | 意義 |
|---|---|
| $\sigma$ | 歸約索引的重新排列，對兩個運算元一致套用 |

唯一的要求是：從 LDS 讀出的 A 片段必須使用與預先重排的 B 相同的 $\sigma$，而 kernel 的 LDS 讀取位移保證了這一點。

## 4. 主迴圈

穩定狀態是一個對每一步 K（64）重複的區塊。

### 4.1 每個區塊的指令預算

從反組譯結果統計：

| 每一步 K=64、每個 wave | 數量 | 位元組 |
|-------------------------|-------|-------|
| `v_mfma_f32_16x16x16_bf16` | 32 | – |
| `buffer_load_dword … offen lds`（A，直接載入 LDS） | 16 | 16 × 64 × 4 = 4 KiB |
| 載入 AGPR 的 `buffer_load_dwordx4`（B） | 2 | 2 KiB |
| 載入 AGPR 的 `ds_read_b128`（*下一步*的 A 運算元） | 16 | 16 KiB |
| `s_barrier` | 1 | – |

四個 wave 合力把整個 128×64 的 A 分塊（16 KiB）載入 LDS、各自載入自己的 64×64 B 欄（8 KiB），然後發出 128 道 MFMA：每取得 24 KiB 就完成 128 × 16·16·16 × 2 = 1 MFLOP。對照第 05 章的教學 kernel：每 2 次全域載入、4 次 LDS 讀取與 2 次 LDS 寫入，才發出 8 道 MFMA。

分塊形狀決定了重用率。以 $128\times64$ 的 workgroup 分塊、每步 K 為 64（bf16，2 位元組）計算：

$$
I_{\text{tile}} = \frac{2\,B_MB_NB_K}{2\,(B_M + B_N)\,B_K} = \frac{128\cdot64}{128 + 64} \approx 42.7\ \frac{\text{flop}}{\text{byte}}, \qquad
\rho = \frac{n_{\text{MFMA}}}{n_{\text{mem}}} = \frac{32}{16 + 2 + 16} \approx 0.94
$$

| 符號 | 意義 |
|---|---|
| $B_M, B_N, B_K$ | workgroup 分塊：128（M）、64（N）、64（每步 K） |
| $I_{\text{tile}}$ | 從 L2/HBM 取得的每個位元組所對應的運算量 |
| $n_{\text{MFMA}}$ | 每個 wave 每一步 K 的 MFMA 數 |
| $n_{\text{mem}}$ | 每個 wave 每一步 K 的記憶體指令數（buffer 載入加上 LDS 讀取） |
| $\rho$ | 每道記憶體指令對應的 MFMA 數：約為 1，因此每道記憶體指令都能藏在一道 MFMA 的執行時間裡 |

第 05 章的教學 kernel 同樣是 $\rho = 8/8 = 1$，但它的每道 MFMA 周圍都是位址計算、等待與 barrier；這裡則是整個迴圈主體都經過排程，讓矩陣管線從不閒置。

### 4.2 直接載入 LDS

```asm
s_add_u32 m0, 0x100, s42                     ; LDS destination = M0 (+ lane * 4)
buffer_load_dword v22, s[4:7], 0 offen lds   ; global A[...] -> LDS, bypassing VGPRs
```

加上 `lds` 修飾詞之後，每個 lane 的 dword 會寫到 LDS 的 `M0 + lane·4`，而不是寫進 `v22`（這裡的 `v22` 只是*位址位移*）。每道指令搬運 256 位元組，所以 `M0` 每次增加 `0x100`。`s42`/`s43` 在迭代之間交替使用：它們就是兩個 LDS 緩衝區。

迴圈中完全沒有 `ds_write`。每一步因此省下 16 道指令，以及原本用來中轉資料的 VGPR。

### 4.3 暫存器雙緩衝

```asm
v_mfma_f32_16x16x16_bf16 v[44:47], a[128:129], a[0:1], v[44:47]   ; compute with a[0:63]...
ds_read_b128 a[64:67], v37 offset:16512                            ; ...while loading a[64:127]
```

第 *k* 步的 MFMA 從 `a[0:63]` 讀取 A 片段，同時第 *k+1* 步的 `ds_read` 正在填入 `a[64:127]`；下一個區塊則交換兩者的角色。B 也以同樣方式在 `a[128:135]`、`a[136:143]` 與 `a[144:151]` 之間輪替。

AMD GPU 無法以動態索引存取暫存器，因此這種輪替必須**在程式碼中展開**：
- 兩個迴圈主體 `label_02CA` 與 `label_0703`，各含 6 個展開的區塊；
- 每個主體有 192 道 MFMA、96 次直接載入 LDS，以及 96 次 `ds_read_b128`。

整份反組譯結果合計有 384 道 MFMA、240 次直接載入 LDS，以及 208 次 `ds_read_b128`。

### 4.4 指令穿插

看一個區塊的開頭幾行：

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

模式是：**一道 MFMA，接著 1–3 道記憶體或純量指令**，如此重複。MFMA 發出後，會讓矩陣核心忙上好幾個週期；在這段期間，wave 的指令仲裁器可以免費發出載入、`M0` 更新或指標遞增。

![MFMA 讓矩陣核心保持忙碌，載入、LDS 讀取與純量更新則穿插在其間發出](figures/ch06-interleave.svg)

這是 AMD GEMM 中最重要的一個排程觀念，也正是 TensileLite 的 `ScheduleIterAlg=3` 自動化的內容（第 07 章）。

### 4.5 計算尚未完成的載入

`s_waitcnt vmcnt(18)` 的意思是「當尚未完成的向量記憶體操作不超過 18 筆時就繼續」。每個區塊發出 16 + 2 = 18 筆這類操作。向量記憶體操作依序完成，所以第 *k+1* 個區塊開頭的 `vmcnt(18)` 會等待第 *k-1* 個區塊及更早的所有載入，而第 *k* 個區塊的載入仍在進行。這就是**提前一整個區塊的預先載入**，只用一個計數器就表達出來，不需要額外的暫存器，也沒有分支。

### 4.6 不含分支的 K 尾端

```asm
s_add_u32 s31, 0x100, s33
s_cmp_lt_u32 s31, s34
s_cselect_b32 s40, s40, 0     ; pointer increment becomes 0 past the end of K
s_add_u32 s4, s40, s4         ; advance A's buffer base
```

接近 K 的結尾時，指標增量被設成零，因此預先載入只會無害地重讀最後一個分塊，而不會越界讀取；迴圈主體也就不需要任何分支。

buffer resource（`s[4:7]`、`s[8:11]`）也能在硬體上做邊界檢查：越界的載入傳回 0，越界的儲存直接被丟棄。TensileLite 等自動產生的 kernel 就依賴這一點來處理邊緣分塊。本 kernel 把 `num_records` 設為 `0xFFFFFFF0`（prologue 中的 `s_mov_b32 s6, -16`），等同於關閉這項檢查。

## 5. Epilogue：Split-K 與 bf16 捨入 { #5-epilogue-split-k-and-bf16-rounding }

### 5.1 合併部分分塊

當 `splitk > 1` 時，網格的 z 維度會切分 K。每個 z 切片累加出一個 128×64 的部分分塊：

- **fp32 輸出：** 以 `global_atomic_add_f32` 把部分結果加起來（反組譯清單中有 64 道）。
- **bf16 輸出：** 先捨入，再以 `global_atomic_pk_add_bf16` 相加，每道原子操作處理兩個 bf16 值（清單中有 48 道）。
- 清單中另有 16 道一般的儲存，供不切分的路徑使用。

一個小小的**號誌工作區**（`ptr_semaphore`，16 × 64 個 `uint32`，在 `gemm_op_a16w16.py` 中依 stream 初始化為零）存放每個分塊的抵達計數器。最後抵達的 workgroup 負責最後階段並重設計數器。這就是啟動程式要求 `gdx·gdy ≤ 1024` 的原因，也說明了不同的 stream 需要各自的工作區：共用計數器會造成死結。

![Split-K：每個 z 切片以原子操作加上自己的部分分塊；每個分塊的計數器選出最後抵達者](figures/ch06-split-k.svg)

### 5.2 手工實作的 bf16 捨入

fp32 → bf16 的轉換是手工完成的：

```asm
v_cmp_u_f32_e64 s[56:57], v44, v44     ; NaN?
v_add3_u32 v8, v44, v11, 1             ; bits + 0x7fff + 1  (v11 = 0x7fff)
v_cndmask_b32_e64 v4, v8, v10, s[56:57] ; NaN -> 0x7fff0000 (canonical qNaN)
v_perm_b32 v76, v5, v4, s35            ; s35 = 0x07060302: pack the two high halves
```

加上 `0x8000` 再截斷，會讓恰好落在中間的值*朝絕對值較大的方向*捨入。這**不是**「最接近、平手取偶數」（round-to-nearest-even）；torch 的 `.to(torch.bfloat16)` 加的是 `0x7fff + lsb`。在恰好平手時，兩者可能相差 1 ulp——當你要與參考實作逐位元比對時，這一點很重要。

以 fp32 值的 32 位元樣式 $u$ 表示兩種捨入規則：

$$
\operatorname{bf16}_{\text{RNE}}(u) = \Bigl\lfloor \frac{u + \texttt{0x7FFF} + \bigl(\lfloor u / 2^{16} \rfloor \bmod 2\bigr)}{2^{16}} \Bigr\rfloor, \qquad
\operatorname{bf16}_{\text{AITER}}(u) = \Bigl\lfloor \frac{u + \texttt{0x8000}}{2^{16}} \Bigr\rfloor
$$

| 符號 | 意義 |
|---|---|
| $u$ | 以無號整數表示的 fp32 位元樣式（NaN 另外處理） |
| $\lfloor u/2^{16}\rfloor \bmod 2$ | 保留下來的最低位元（bf16 尾數的 LSB） |
| $\operatorname{bf16}_{\text{RNE}}$ | 最接近、平手取偶數（PyTorch） |
| $\operatorname{bf16}_{\text{AITER}}$ | 最接近、平手遠離零（本 kernel） |

只有在低 16 位元恰好是 `0x8000`、且保留的 LSB 為 0 時，兩者才會不同。

### 5.3 Split-K 的算術

Split-K 本身只是對各 K 範圍求和：

$$
C = \sum_{z=0}^{S-1} A_{:,\,\mathcal{K}_z}\,B_{\mathcal{K}_z,\,:}, \qquad
\mathcal{K}_z = \Bigl[\,z\,\tfrac{K}{S},\ (z+1)\,\tfrac{K}{S}\Bigr)
$$

| 符號 | 意義 |
|---|---|
| $S$ | 切分數（`splitk`，網格的 z 大小） |
| $\mathcal{K}_z$ | 第 $z$ 個切片負責的 K 範圍 |

它把 workgroup 數量乘上 $S$，代價是每個分塊有 $S$ 份部分結果要合併（使用原子操作），而且 fp32 的加總順序不固定。

## 6. 修改與分析這類 kernel

AITER 在 `docs/isa_kernel_optimization.md` 中記錄了完整的工作流程，相關腳本位於 `docs/examples/isa_optimization/`：

1. **先做往返驗證。** `roundtrip.sh <kernel.co>` 會取出獨立的 `kernel.s`，以 `clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942` 重新組譯，再比較 `.text`、kernel descriptor 與中繼資料。之後出現的任何差異，都是你自己造成的。
2. **編輯。** 在手寫組合語言中，沒有任何東西會替你檢查 hazard：
   - 組譯器完全照你寫的內容編碼。
   - 針對 hazard 的 `s_nop` 是必要的等待狀態（例如在相依的 MFMA 之間，或超越函數運算之後）。`s_nop N` 提供 N+1 個等待狀態。
   - 移動載入的位置後，每個 `s_waitcnt` 的計數都必須重新計算。
   - 違反這些規則不會觸發錯誤，只會默默讀到過期的資料。

3. **調整資源。** 改變暫存器或 LDS 用量時，必須同時修改 `.amdhsa_next_free_vgpr`、`.amdhsa_accum_offset`、`.amdhsa_group_segment_fixed_size`，*以及*中繼資料。
4. **測試。** 替換 `hsa/gfx942/…` 中的 `.co`，執行運算子的測試。AITER 會記錄 `LoadKernel: … hsaco: <path>`。
5. **效能分析。**
   - 計時：`rocprofv3 --kernel-trace --stats --kernel-include-regex bf16gemm`。
   - 逐指令的 thread trace：`rocprofv3 --att --kernel-iteration-range 5-5 --att-target-cu 1`，再用 ROCprof Compute Viewer 檢視。它能看出迴圈的瓶頸是 MFMA 發出、`s_waitcnt`，還是 LDS bank 衝突。

## 重現本章 { #reproduce-this-chapter }

不需要 GPU，也不需要 ROCm；Ubuntu 的 LLVM 18 套件就夠了。

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

使用 LLVM 18 時，往返驗證會回報 `.text`、kernel descriptor 與中繼資料都**完全相同**。腳本另外會印出一行 `e_flags` 的「DIFFERS」，但兩邊的值相同（`0x54C`）；這只是腳本比較 LLVM 18 輸出時的小毛病，並非真正的差異。

## 重點整理

1. 生產環境的 GEMM 是一張分派到各種專用 kernel 的表；手寫 kernel 只在部分形狀上勝出。
2. 最快的 CDNA GEMM 會用滿 512 個暫存器與 64 KiB 的 LDS：每個 SIMD 一個 wave，靠指令排程而不是其他 wave 來隱藏延遲。
3. 多個 wave 共用的運算元經過 LDS；某個 wave 私有的運算元（預先重排的權重）則直接載入暫存器。
4. 主迴圈讓每道 MFMA 穿插 1–3 道記憶體或純量指令，在暫存器中對運算元做雙緩衝，並只用一個 `s_waitcnt vmcnt(N)` 就把全域載入管線化。
5. Split-K 以原子操作累加部分分塊（fp32 下結果不具決定性）；手工實作的 bf16 捨入在平手時可能與 PyTorch 相差 1 ulp。
6. 必須先做到逐位元組相同的往返驗證，才能修改這類 kernel；沒有任何工具會替你檢查 hazard。

## 練習

1. 反組譯 `bf16gemm_fp32bf16_tn_32x64_pf3_splitk.co`（沒有預先重排）。
   - B 現在放在哪裡？
   - 主迴圈中的 `vmcnt` 是多少？它提供幾步 K 的預先載入（也就是 `pf3`）？
2. 由 MFMA 的數量與每道 16×16×16，算出一個 `.co` 迴圈主體的 FLOP 數。
   - 提示：MI300X 的稠密 bf16 峰值 1307 TFLOP/s ÷（304 個 CU × 4 個 SIMD × 2.1 GHz）≈ 每個 SIMD 每週期 512 FLOP，所以一道 16x16x16 MFMA（8192 FLOP）約需 16 個週期。
   - 每一步 K 為每個 wave 提供多少週期的 MFMA 工作？
   - MFMA 之間必須容納多少道非 MFMA 指令？

    <details markdown="1"><summary>答案</summary>

    每個 wave 每一步 K（64）發出 32 道 MFMA（第 4.1 節），約為 $32 \times 16 = 512$ 個週期的矩陣核心工作；同時有 34 道記憶體指令（16 + 2 + 16），另加純量與位址更新：大約每道 MFMA 搭配一道其他指令，正是第 4.4 節的穿插模式。一個迴圈主體（6 個區塊）有 192 道 MFMA：每個 wave 約 $192 \times 8192 \approx 1.57$ MFLOP。

    </details>
3. 寫出 `shuffle_weight(layout=(16,16))` 在一個 16×32 區塊內套用的確切 k 排列，並確認 A 端的 LDS 配置必須使用相同的排列。
