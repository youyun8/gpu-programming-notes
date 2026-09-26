# 14 – Triton：用 Python 進行 Block-Level GPU Programming

> **第四部 · 可攜式模型 Kernel** · 先備知識：[03](03-parallel-reduction.md)、
> [04](04-tiled-matmul.md)、[13](13-softmax-attention.md) ·
> 程式：[`examples/14-triton/`](examples/14-triton/test_kernels.py) ·
> 下一章：[08 – 部署本站](08-deploying-this-site.md)

CUDA 要求你撰寫*一個 thread* 的程式，並親自安排數千個 thread 如何合作：哪個 thread 載入哪些 byte、哪些資料進 shared memory、barrier 放在哪裡。Triton 將抽象層級提高一步：你用 Python 撰寫*一個 block* 的程式，以整個 tile 為單位操作（載入 64 × 64 tile、將兩個 tile 相乘、沿此 axis 取 maximum），compiler 再決定 tile 如何分散到 thread、load 如何 vectorize/coalesce、哪些資料經 shared memory staging，以及使用哪些 instruction（tensor core、`cp.async`、TMA）。第三部大多數技術都會自動完成；剩下的恰好是演算法選擇：tile size、tile order、fusion。

**你將學會**

- Triton programming model：program、block、mask 與 pointer tile；
- 四個完整 kernel（vector addition、fused softmax、含 grouped ordering 與 autotuning 的 matrix multiplication、FlashAttention），以及它們如何對應前面章節的 CUDA；
- 第 04 章哪些 GEMM 最佳化由 compiler 完成，哪些仍由你負責；
- Triton kernel 如何編譯、specialize、cache、檢查、在 interpreter 中除錯、benchmark 與 profile；
- 何時選 Triton、CUDA 或函式庫。

## 1. 為何使用 Block-Level 語言

### 1.1 從你手上移交給 Compiler 的工作

| 考量 | CUDA | Triton |
|---|---|---|
| 程式單位 | 一個 thread | 一個 program instance（一個 thread block） |
| 資料 | Register 中的 scalar | 靜態、二次方 shape 的 tensor |
| Thread ↔ element mapping | 你 | Compiler（一個 *layout*） |
| Coalescing、vector width | 你（第 02、04.1 章） | Compiler，依 pointer pattern 與 alignment |
| Shared memory、barrier | 你 | Compiler |
| Bank-conflict swizzle | 你（04.5） | Compiler |
| Multi-stage load pipeline | 你（04.2、04.3） | Compiler，透過 `num_stages` |
| Tensor-core instruction | 你（04.7） | Compiler，來自 `tl.dot` |
| Tile size、grid、tile order | 你 | 你 |
| Fusion（一個 kernel 做什麼） | 你 | 你 |

代價是放棄控制 per-thread code（因此手寫 warp specialization 或 register-level trick 等技術較難或不可能），換得短 5–10 倍、可跨 NVIDIA 與 AMD GPU 的 kernel。

![CUDA 描述一個 thread 並由你選 mapping；Triton 描述一個 block，由 compiler 將它配置到各 warp](figures/ch14-model.svg)

### 1.2 設定

```bash
pip install torch triton          # Triton ships with PyTorch's CUDA wheels too
cd tutorials/examples/14-triton
python3 test_kernels.py           # checks all four kernels against PyTorch
python3 test_kernels.py --bench   # and times them (GPU only)
```

沒有 GPU 時，`test_kernels.py` 會在 import Triton 前設定 `TRITON_INTERPRET=1`。**Interpreter** 會用 NumPy 依序執行每個 program instance：雖然慢，卻執行相同 indexing、mask 與 arithmetic，因此相當於本儲存庫 CUDA emulator 的 Triton 版本，也是 CI 所用方式。在 ROCm 上，同一份程式可透過 PyTorch 的 ROCm build 在 AMD Instinct GPU 執行。

## 2. Programming Model

### 2.1 Program 與 Grid

Triton kernel 是加上 `@triton.jit` decorator 的 Python function。它在 **program instance** grid 上以 `kernel[grid](arg0, arg1, ...)` launch；`grid` 是最多三個 size 的 tuple，或依 compile-time parameter 決定的 function：

```python
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
kernel[grid](x, y, out, n, BLOCK=1024)
```

Kernel 內的 `tl.program_id(axis)` 是 program index（CUDA `blockIdx`），`tl.num_programs(axis)` 是 grid size（`gridDim`）。沒有 `threadIdx`：program 就是一個 block，執行它的 thread 數由 launch option `num_warps` 決定（預設 4）。

### 2.2 Block：靜態、二次方 Tensor

Kernel 內的 value 是 scalar 或 **block**：shape 在 compile time 已知的 tensor。`tl.arange(0, BLOCK)` 建立 `[0, 1, …, BLOCK−1]` vector；`BLOCK` 必須是 `tl.constexpr` 與二次方。Operation 依 NumPy broadcasting 做 element-wise 運算（`x[:, None]`、`y[None, :]`）；reduction 指定 axis（`tl.sum`、`tl.max`、`tl.argmax`）；`tl.dot` 將兩個 2-D block 相乘。

每個不同的 `constexpr` value 都會編譯成不同 kernel，因此 size 以 keyword 傳入：`BLOCK=1024`。

### 2.3 Pointer、Load、Store 與 Mask

Tensor argument 以指向第一個 element 的 pointer 傳入。Pointer 加上一個 offset block 會得到 **pointer block**，`tl.load`/`tl.store` 一次讀寫全部：

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n                               # the tail guard, for the whole block
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
tl.store(out_ptr + offs, x, mask=mask)
```

被 mask 的 lane 不讀也不寫；`other` 是 masked load 的回傳值（選擇後續操作的 identity：sum 用 0，maximum 用 −∞）。

Offset 以 *element* 為單位；Triton 會自行乘上 element size。除非 argument 使其成為 64-bit，否則 offset 是 32-bit integer；tensor 超過 $2^{31}$ 個 element 時，先使用 `pid.to(tl.int64)` cast。

### 2.4 二維 Tile

2-D pointer tile 是 row offset column 與 column offset row 的 broadcast sum：

![rows[:, None] * S 加上 cols[None, :]，broadcast 成 BLOCK_M × BLOCK_N address tile；mask 也以相同方式建立](figures/ch14-pointer-block.svg)

$$
\text{ptr}_{rq} = \text{base} + \text{row}_r \cdot s_0 + \text{col}_q \cdot s_1,
\qquad r < B_M,\ q < B_N
$$

| 符號 | 意義 |
|---|---|
| $\text{base}$ | 指向 matrix element (0, 0) 的 pointer |
| $\text{row}_r,\ \text{col}_q$ | `rows` 第 r 個、`cols` 第 q 個 entry |
| $s_0,\ s_1$ | Matrix 的 row/column stride，以 element 為單位（`x.stride(0)`、`x.stride(1)`） |
| $B_M,\ B_N$ | Tile shape（`BLOCK_M`、`BLOCK_N`） |

同時傳兩個 stride，便能讓同一 kernel 處理 row-major、transposed 與 sliced matrix。Compiler 只對可證明連續且對齊的 dimension 發出 vectorized load；它會從 access pattern 與 argument specialization（第 7.1 節）得知這些性質。若 stride 永遠為 1，像 softmax kernel 一樣從 signature 省略它，可明確表達 contiguity。

### 2.5 語言中沒有什麼

- 一般程式碼**沒有 shared memory 或 barrier**：program 內的資料交換透過 block operation（`tl.sum`、`tl.dot`、`tl.trans`、reshape），compiler 視需要以 shuffle 或 shared memory 實作。
- 除了 global memory 與 atomic（`tl.atomic_add`、`tl.atomic_cas` 等），**program 間不能通訊**，與 CUDA 相同。
- Kernel 內**沒有 dynamic shape**：runtime length 的 row 以 power-of-two block 加 mask 處理，或 loop 多個 block。

## 3. Kernel 1：Vector Addition

[`vector_add.py`](examples/14-triton/vector_add.py) 用十行展示完整 model：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # blockIdx.x
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # a vector of BLOCK indices
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

Grid 有 $\lceil n / B \rceil$ 個 program。`BLOCK = 1024`、`num_warps = 4` 時，128 個 thread 各自負責 8 個 element，compiler 將其載入為每 thread 兩個 16-byte vector：不必手寫第 04.1 章的 float4 load。Kernel 達到與良好 CUDA copy 相同的 bandwidth；總成本為 $12n$ byte（第 00 章第 7 節）。

## 4. Kernel 2：Fused Softmax

[`softmax.py`](examples/14-triton/softmax.py) 以每 row 一個 program 計算 row-wise softmax：

```python
@triton.jit
def softmax_kernel(in_ptr, out_ptr, n_cols, in_stride, out_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out_ptr + row * out_stride + cols, e / tl.sum(e, axis=0), mask=mask)
```

### 4.1 Compiler 產生什麼

`tl.max(x, axis=0)` 是第 03 章的 block reduction：per-thread partial maximum、warp shuffle tree，以及 warp 間的 shared-memory exchange，全由一次呼叫導出。Masked lane 載入 −∞，不會改變 maximum，也因 $e^{-\infty} = 0$ 而不影響 sum。

### 4.2 為何快速

Row 只讀一次進 register、寫一次：每 row $2 \cdot 4 \cdot n$ byte，已是最低值。第 13 章第 2 節的 three-kernel 版本讀三次。優勢並非 Triton 專屬（第 13 章 online warp-per-row CUDA kernel 也如此），但 fusion 在 Triton 中是自然寫法。

### 4.3 限制

`BLOCK = next_power_of_2(n_cols)` 必須放得進 register。Wrapper 對較長 row 使用更多 warp，讓每 thread 到 16,384 column 為止最多保存約 32 個 value；超過數萬 column 後會 spill。解法是第 13 章的 online softmax：以 block loop row，並將 $(m, z)$ 保存為 state（練習 2）。

## 5. Kernel 3：Matrix Multiplication

[`matmul.py`](examples/14-triton/matmul.py) 計算 $C = AB$，每個 program 負責 C 的 $B_M \times B_N$ tile：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k0 in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=(rows[:, None] < M) & (ks[None, :] + k0 < K), other=0.0)
    b = tl.load(b_ptrs, mask=(ks[:, None] + k0 < K) & (cols[None, :] < N), other=0.0)
    acc = tl.dot(a, b, acc)                    # acc += a @ b on tensor cores
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

這就是第 04 章 tiled kernel 的結構，差別在 compiler 接手的部分。

### 5.1 重訪第 04 章技術

| 技術（章節） | 在 Triton 中 |
|---|---|
| Vectorized load（04.1） | Stride 與 alignment 允許時自動完成 |
| Double buffering（04.2） | 自動：shared memory 中的 `num_stages` 個 buffer |
| `cp.async` / TMA（04.3） | Ampere 之後為 pipelined load 自動使用 `cp.async`；Hopper 之後透過 tensor descriptor（`tl.make_tensor_descriptor`）使用 TMA |
| Warp tiling（04.4） | 自動：`tl.dot` layout 將 tile 分到 `num_warps` 個 warp |
| Shared-memory swizzle（04.5） | 自動 |
| Tile-order swizzle（04.5） | 由你負責：`GROUP_M`（第 5.2 節） |
| Split-K / Stream-K（04.6） | 由你負責：增加 K grid axis，再 reduction 或 atomic |
| Tensor core（04.7） | 從 `tl.dot` 自動產生 |

### 5.2 Grouped Tile Ordering

Program 大致依 ID 順序執行。Row-major order 讓同時 in flight 的 program 涵蓋一兩個 tile row，卻橫跨 $B$ 的**所有** tile column，造成 B 反覆從 DRAM 讀取。將 $G$ 個 tile row 分組，可讓 in-flight program 改為涵蓋 C 的近方形區域：

$$
g = \left\lfloor \frac{p}{G\,T_N} \right\rfloor, \quad
G' = \min(T_M - gG,\ G), \quad
m = gG + \big(p \bmod G T_N\big) \bmod G', \quad
n = \left\lfloor \frac{p \bmod G T_N}{G'} \right\rfloor
$$

| 符號 | 意義 |
|---|---|
| $p$ | Program id |
| $T_M,\ T_N$ | C 的 tile row/column 數 |
| $G$ | `GROUP_M`，每 group 的 tile row 數 |
| $g$ | Program $p$ 所在 group |
| $G'$ | 此 group 的 tile row 數（最後一組可能較短） |
| $m,\ n$ | Program $p$ 計算的 C tile |

若 $T_N = 32$ 且有 64 個 program in flight，row-major order 觸及 2 個 A row strip 與 32 個 B column strip（共 34）；$G = 8$ 時則各觸及 8 個（共 16），L2 footprint 約減半、L2 hit 相應增加。這與第 04.5 章 tile-order swizzle 相同。

### 5.3 Autotuning

最佳 tile shape 取決於 GPU、data type 與 matrix size。`triton.autotune` 對 config list 編譯 kernel，在第一次遇到新 `key` 時逐一計時並 cache 勝者：

```python
CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    ...
]
matmul_kernel_tuned = triton.autotune(configs=CONFIGS, key=["M", "N", "K"])(matmul_kernel)
```

| Parameter | 取捨 |
|---|---|
| `BLOCK_M`, `BLOCK_N` | 較大 tile：每 load byte 有更多 reuse（第 04 章第 3 節）、更多 register、較少 program |
| `BLOCK_K` | 較大：較少 loop iteration、每 stage 更多 shared memory |
| `num_stages` | 更多 stage 隱藏更多 latency，也花費 `num_stages` × tile byte 的 shared memory |
| `num_warps` | 每 program 更多 warp：較小 per-warp tile、更多 latency hiding、較少 ILP |

每 program shared memory 約為 $\text{num\_stages} \cdot (B_M + B_N) \cdot B_K \cdot \text{sizeof}$；超過 SM capacity 的 config 無法編譯，會被略過。

### 5.4 Precision

輸入為 `float32` 時，`tl.dot` 在 Ampere 之後預設使用 TF32 tensor core（10-bit mantissa、relative error 約 $10^{-3}$），所以 test script 在 GPU 上放寬 tolerance。`tl.dot(a, b, acc, input_precision="ieee")` 可強制完整 FP32，但速度代價很高。對 `float16`/`bfloat16` input，accumulator 保持 `float32`，epilogue（`acc.to(c_ptr.dtype.element_ty)`）只在最後轉換一次。

### 5.5 Epilogue Fusion

Store 前對 `acc` 做任何 element-wise 操作都不增加 memory traffic：bias add、activation（`tl.where(acc > 0, acc, 0.0)`）、scale、轉成 FP8。手寫 Triton GEMM 最常在此勝過 library call 後接獨立 element-wise kernel。

## 6. Kernel 4：FlashAttention

[`flash_attention.py`](examples/14-triton/flash_attention.py) 對一個 head 實作第 13 章第 5 節 forward pass。每個 program 負責 `BLOCK_Q` 個 query row，並讓 $K$、$V$ 流過：

```python
for start in range(0, kv_end, BLOCK_KV):
    k = tl.load(k_ptr + kv_rows[:, None] * stride_k + dims[None, :], ...)
    v = tl.load(v_ptr + kv_rows[:, None] * stride_v + dims[None, :], ...)
    s = tl.dot(q, tl.trans(k))                         # scores, on-chip
    s = tl.where(valid, s, -float("inf"))              # padding and causal mask
    m_new = tl.maximum(m, tl.max(s, axis=1))
    m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
    p = tl.exp(s - m_safe[:, None])
    alpha = tl.exp(m - m_safe)
    l = l * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    m = m_new
```

逐行對應第 13 章 online-softmax recurrence：

| 程式 | 第 13 章含義 |
|---|---|
| `s = tl.dot(q, tl.trans(k))` | 一個 tile 的 $S = Q K^\top$，scale 已折入 $Q$ |
| `m_new`、`alpha` | 新 running maximum 與 rescaling factor $e^{m_{\text{old}} - m_{\text{new}}}$ |
| `m_safe` | 只看過 masked key 的 row guard（$-\infty - (-\infty)$） |
| `l`、`acc` | Running sum 與未 normalization output，先 rescale 再 update |
| `acc / l[:, None]`（loop 後） | 最終 normalization |

[`examples/13-softmax-attention.cu`](examples/13-softmax-attention.cu) 的 CUDA 版本約 100 行，大多在將 tile 分配到 lane、透過 shared memory staging $K$、$V$；此處兩個 `tl.dot` 都在 tensor core 執行，staging 由 compiler pipeline。Production kernel 還會加入 backward pass、以額外 grid axis 表達多 head/batch，以及 Hopper warp specialization，但核心就是此 loop。

Causal case 的 `kv_end` 讓 loop 在 block query 能看到的最後 key 停止，工作量減半，與 CUDA kernel 相同。

## 7. 編譯、除錯與 Profiling

### 7.1 從 Python 到 Machine Code

![Compiler 將 decorated Python function 依序 lowering 成 Triton IR（block operation）、TritonGPU IR（layout、shared memory、pipeline）與 LLVM IR，最後成為 PTX 或 AMDGCN](figures/ch14-compiler.svg)

首次使用新的 constexpr value、argument dtype 與 **specialization** 組合 launch 時會編譯；其中 Triton 會檢查 pointer 與 integer argument 是否可被 16 整除，以證明 vectorized load alignment。後續 launch 使用 memory/on-disk cache。Launch 回傳 handle，其 `asm` dictionary 保存各 stage：

```python
handle = add_kernel[grid](x, y, out, n, BLOCK=1024)
print(handle.asm["ttgir"])    # layouts chosen by the compiler
print(handle.asm["ptx"])      # look for ld.global.v4.f32 (vectorised loads)
```

TritonGPU IR 能回答效能問題：它顯示每個 tensor layout（`#blocked`、`#mma`、`#shared`）、shared memory 配置位置，以及建立多少 pipeline stage。

### 7.2 除錯

| 工具 | 用途 |
|---|---|
| `TRITON_INTERPRET=1` | 用 NumPy 在 CPU 執行；kernel 內可用 `print()` 與 `pdb` |
| `tl.device_print("x", x)` | 從 compiled kernel 印出（每個 program 都會印；用 mask 或小 grid 限制） |
| `tl.static_print`、`tl.static_assert` | Compile time 印出或檢查 constexpr |
| `tl.device_assert(cond, "msg")` | Runtime assertion（以 `TRITON_DEBUG=1` 啟用） |

常見錯誤是靜態的：`arange` bound 不是二次方、shape 無法 broadcast、compiler 需要 constant 卻收到 non-constexpr，以及 `tl.dot` 某個 dimension 小於 16。

### 7.3 Benchmark 與 Profile

`triton.testing.do_bench(fn)` 會 warm up、重複執行並在 run 間 flush L2，回傳毫秒；`triton.testing.perf_report` 會 sweep size 並繪圖。第 00 章 roofline arithmetic 與第 09 章 profiler 仍適用：Triton kernel 對 Nsight Compute 而言是一般 kernel（`ncu -k regex:matmul_kernel python3 test_kernels.py --bench`），且預設含類似 `-lineinfo` 的資訊，source attribution 會指向 Python 行。

## 8. Triton、CUDA 還是函式庫？

| 情境 | 選擇 |
|---|---|
| 標準 shape 的標準 GEMM、convolution 或 attention | 函式庫（cuBLAS、cuDNN、hipBLASLt、FlashAttention） |
| 函式庫沒有的 fused operation（GEMM + 自訂 epilogue、新 attention variant、fused norm） | Triton |
| 必須在 NVIDIA、AMD 執行的研究程式 | Triton |
| 單一 architecture 最後 10–20%：warp specialization、自訂 pipeline、特殊 data movement | CUDA（或 CUTLASS/CuTe，第 04.7 章） |
| 由不規則 per-thread control flow 主導的演算法（sorting network、graph traversal） | CUDA |

## 重點整理

1. Triton kernel 以靜態、二次方 tensor 描述一個 block；compiler 將其對應到 thread。
2. Pointer block 加 mask 取代 thread indexing 與 bounds check；`other` 提供 masked lane 的 identity。
3. Coalescing、vectorization、shared-memory staging、pipelining、swizzle 與 tensor-core instruction 來自 compiler；tile size、tile order 與 fusion 仍由你決定。
4. `triton.autotune` 依 problem size 搜尋 tile shape、`num_warps` 與 `num_stages`。
5. Interpreter（`TRITON_INTERPRET=1`）可在 CPU 測試；`asm` stage 與 Nsight Compute 解釋效能。

## 練習

1. `add_kernel` 中 `BLOCK` 設為 1000 會如何？Load 省略 `mask` 又會如何？

    <details markdown="1"><summary>答案</summary>

    `tl.arange(0, 1000)` 無法編譯：block size 必須是二次方。沒有 mask 時，最後一個 program 會越界讀取 `x`、`y`（undefined value；`compute-sanitizer` 會回報）；store 若仍有自己的 mask，結果看似正確，kernel 卻是錯的。

    </details>

2. 為長到無法放入 register 的 row 撰寫 softmax kernel：以 `BLOCK` column 分段 loop row，保存 running maximum 與 sum（第 13 章第 3 節），再 loop 一次寫 output。

    <details markdown="1"><summary>提示</summary>

    第一個 loop 中保存 shape `(BLOCK,)` 的 per-lane vector `m`、`z`（`m_new = tl.maximum(m, x)`，`z = z * tl.exp(m - m_new) + tl.exp(x - m_new)`，並加 −∞ guard）；最後以 `M = tl.max(m, 0)`、`Z = tl.sum(z * tl.exp(m - M), 0)` 合併，第二個 loop store `tl.exp(x - M) / Z`。不論 row length，都是兩次 read、一次 write。

    </details>

3. $M = N = K = 4096$、FP16、`BLOCK_M = BLOCK_N = 128`、`BLOCK_K = 32`、`num_stages = 3` 時，一個 program 需要多少 shared memory？A100 SM（164 KB）可容納幾個 program？

    <details markdown="1"><summary>答案</summary>

    $3 \cdot (128 + 128) \cdot 32 \cdot 2 = 49\,152$ byte = 48 KB，依 shared memory 可容納三個。若 `num_warps = 8`（256 thread），128 × 128 FP32 accumulator（每 thread 64 個 value）加上 operand 往往會讓 register file 將數量限制為一或兩個。

    </details>

4. 在 `RELU: tl.constexpr` flag 控制下，為 `matmul_kernel` 加入 fused ReLU epilogue，並與 `torch.relu(a @ b)` 比較。為何 constexpr 優於 runtime flag？

    <details markdown="1"><summary>答案</summary>

    Store 前加入 `if RELU: acc = tl.maximum(acc, 0.0)`。作為 constexpr，每個值會編譯自己的 kernel，branch 消失；runtime flag 會在每個 program 保留一個（uniform、便宜的）branch，也阻止 compiler specialize epilogue。

    </details>

5. `m_safe` 只在 row 至今只看到 masked key 時有用。證明目前的 `flash_attention` 不會發生此事，並舉出會發生的 attention variant。

    <details markdown="1"><summary>答案</summary>

    第一個 tile 永遠從 key 0 開始，對每個 query row 都有效：它存在（$n \ge 1$），且 causal mask 下不是任何 row 的未來 key（超出 $n$ 的 padding row 也看得到）。因此第一個 tile 後每個 `m` 都是 finite。Sliding-window attention（row $r$ 只看 $r-w,\dots,r$，晚期 row 的早期 tile 會全被 mask）、key-padding mask，或以不同順序拜訪 tile（例如從 diagonal 開始）會破壞此條件。沒有 guard 時，`s - m_new` 將成為 $-\infty - (-\infty) = \text{NaN}$。

    </details>

## 實作練習

LeetGPU 與 Tensara 接受 Triton submission；以下問題適合首次移植上述 kernel：

- [LeetGPU – Vector Addition](../leetgpu/001-vector-add/)
- [LeetGPU – Softmax](../leetgpu/005-softmax/)、[Tensara – Softmax](../tensara/softmax/)
- [LeetGPU – Matrix Multiplication](../leetgpu/002-matrix-multiplication/)、
  [Tensara – Matrix Multiplication](../tensara/matrix-multiplication/)
- [LeetGPU – Softmax Attention](../leetgpu/006-softmax-attention/)、
  [LeetGPU – Causal Attention](../leetgpu/053-casual-attention/)
- [Tensara – Layer Norm](../tensara/layer-norm/)（類似第 4 節的 fused reduction）
