# 14 – Triton – 從第一個 Kernel 到生產環境

> **第四部 · Triton** · 一份從入門到進階的獨立指南 ·
> 程式與測試：[`examples/14-triton/`](examples/14-triton/test_kernels.py)

Triton 是一套用來撰寫 GPU kernel 的 Python 語言與編譯器。你只要描述一個
**program instance** 如何處理 vector 或 tile，編譯器便會把工作對應到 GPU
thread、vector load、shared memory 與加速器指令。本章不要求 CUDA 經驗。
需要 CUDA 術語協助理解時，可以把一個 Triton program instance 大致視為一個
CUDA thread block；grid 是所有 block 的集合；block value 則是形狀在編譯期
決定、分散到各 thread 的 tensor。

較高階的抽象不代表不必做效能工程。你仍要選擇工作分解方式、grid、tile
大小、mask、走訪順序、fusion 邊界與 launch 參數。Triton 會負責大部分機械式
的 thread mapping 與 lowering。

**你將學會**

- Triton programming model：program、block、mask 與 pointer tile；
- 六個完整 kernel：vector addition、fused softmax、fused LayerNorm、
  使用 grouped ordering 與 autotuning 的 matrix multiplication、
  FlashAttention，以及 atomic histogram；
- matrix multiplication 的哪些最佳化由編譯器完成，哪些仍由你負責；
- loop、atomic、scan、persistent scheduling、compiler layout，以及
  NVIDIA 與 AMD 之間的可攜性；
- Triton kernel 如何編譯、specialize、cache、檢查、在 interpreter 中除錯、
  benchmark 與 profile；
- 何時該選 Triton、CUDA 或函式庫。

## 1. 為何使用 Block-Level 語言

### 1.1 從你手上移交給編譯器的工作

| 考量 | CUDA | Triton |
|---|---|---|
| 程式單位 | 一個 thread | 一個 program instance（一個 thread block） |
| 資料 | Register 中的 scalar | 形狀在編譯期決定的 block tensor |
| Thread ↔ element mapping | 你 | 編譯器（一個 *layout*） |
| Coalescing、vector width | 你 | 編譯器，依 pointer pattern 與 alignment 決定 |
| Shared memory、barrier | 你 | 編譯器 |
| Bank-conflict swizzle | 你 | 編譯器 |
| Multi-stage load pipeline | 你 | 編譯器，受 `num_stages` 影響 |
| Tensor-core instruction | 你 | 編譯器，來自 `tl.dot` |
| Tile 大小、grid、tile 順序 | 你 | 你 |
| Fusion（一個 kernel 做什麼） | 你 | 你 |

這項取捨刻意限制了低階的 per-thread 控制，換來通常短得多、而且可同時支援
NVIDIA 與 AMD GPU 的 kernel。不過，可攜不代表效能自然相同；每個 backend
仍需使用具代表性的工作負載來測試與調校。

![CUDA 描述一個 thread 並由你選擇 mapping；Triton 描述一個 block，由編譯器將它配置到各 thread](figures/ch14-model.svg)

### 1.2 設定

```bash
pip install torch triton          # Triton ships with PyTorch's CUDA wheels too
cd tutorials/examples/14-triton
python3 test_kernels.py           # checks all six kernels against PyTorch
python3 test_kernels.py --bench   # and times them (GPU only)
```

沒有 GPU 時，`test_kernels.py` 會在 import Triton 前設定
`TRITON_INTERPRET=1`。**Interpreter** 會使用 NumPy 依序執行每個 program
instance：雖然很慢，但執行的是相同的 indexing、mask 與 arithmetic，因此可視為
本儲存庫 CUDA emulator 的 Triton 版本。在 ROCm 上，相同原始碼可搭配相容的
PyTorch 與 Triton ROCm build 執行；第 10 節會說明 backend 特有的注意事項。

## 2. 程式設計模型

### 2.1 Program 與網格

Triton kernel 是加上 `@triton.jit` decorator 的 Python function。它會在
**program instance** 的 grid 上以 `kernel[grid](arg0, arg1, ...)` launch；
`grid` 可以是最多三個 size 的 tuple，也可以是依 kernel 編譯期參數決定的
function：

```python
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
kernel[grid](x, y, out, n, BLOCK=1024)
```

Kernel 內的 `tl.program_id(axis)` 是 program 的 index（CUDA 的
`blockIdx`），`tl.num_programs(axis)` 則是 grid size（`gridDim`）。這裡沒有
`threadIdx`：一個 program 就是一個 block，而執行它的 thread 數量是 launch
選項 `num_warps`（預設為 4）。

### 2.2 Block 與形狀限制

Kernel 內的 value 是 scalar 或 **block**：形狀在編譯期已知的 tensor。
`tl.arange(0, BLOCK)` 會建立 vector `[0, 1, …, BLOCK−1]`；使用這種形式時，
`BLOCK` 必須是 `tl.constexpr`，而且區間長度必須是 2 的次方。
Operation 會依 NumPy broadcasting 做 element-wise 運算（`x[:, None]`、
`y[None, :]`）；reduction 需指定 axis（`tl.sum`、`tl.max`、`tl.argmax`）；
`tl.dot` 則將兩個 2-D block 相乘。

每個不同的 `constexpr` value 都會編譯成獨立 kernel，因此 size 會以 keyword
傳入：`BLOCK=1024`。

`n`、`M`、`N` 之類的 runtime size 可以是任意值。把 tile 向上取整為合法的
編譯期形狀，再 mask 掉超出 runtime shape 的 lane。Operation 本身也有額外
限制；例如，高效的 `tl.dot` tile，其 dimension 必須與 backend 的 matrix
instruction 相容。靜態形狀錯誤是編譯期錯誤，不能靠 kernel 內的 branch 避開。

### 2.3 Pointer、載入、儲存與 Mask

Tensor argument 會以指向第一個 element 的 pointer 傳入。Pointer 加上一個
offset block 會得到一個 **pointer block**，`tl.load`/`tl.store` 會一次讀寫
全部 pointer：

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n                               # the tail guard, for the whole block
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
tl.store(out_ptr + offs, x, mask=mask)
```

被 mask 掉的 lane 不會讀取或寫入；`other` 是 masked load 的回傳值（應選擇
後續操作的 identity：sum 用 0，maximum 用 −∞）。

Offset 以 *element* 為單位：Triton 會自行乘上 element size。除非某個 argument
使它成為 64-bit，否則 offset 是 32-bit integer；若 tensor 超過 $2^{31}$ 個
element，請先使用 `pid.to(tl.int64)` cast。

### 2.4 二維 Tile

一個 2-D pointer tile，是 row offset column 與 column offset row 的
broadcast sum：

![rows[:, None] * S 加上 cols[None, :]，broadcast 成 BLOCK_M × BLOCK_N 的 address tile；mask 也以相同方式建立](figures/ch14-pointer-block.svg)

$$
\text{ptr}_{rq} = \text{base} + \text{row}_r \cdot s_0 + \text{col}_q \cdot s_1,
\qquad r < B_M,\ q < B_N
$$

| 符號 | 意義 |
|---|---|
| $\text{base}$ | 指向 matrix element (0, 0) 的 pointer |
| $\text{row}_r,\ \text{col}_q$ | `rows` 的第 r 個 entry，以及 `cols` 的第 q 個 entry |
| $s_0,\ s_1$ | Matrix 的 row stride 與 column stride，以 element 為單位（`x.stride(0)`、`x.stride(1)`） |
| $B_M,\ B_N$ | Tile shape（`BLOCK_M`、`BLOCK_N`） |

同時傳入兩個 stride，就能讓同一個 kernel 處理 row-major、transposed 與
sliced matrix。編譯器只會沿著可證明為連續且對齊的 dimension 產生 vectorized
load；它會從 access pattern，以及針對 argument value 的 specialization
（第 7.1 節）得知這些性質。若某個 stride 永遠是 1，像 softmax kernel 一樣
把它從 signature 省略，可以明確表達 contiguity。

### 2.5 Block Pointer 與 Tensor Descriptor

明確的 pointer tensor 是最通用的 addressing 形式。**Block pointer** 會把
相同的 base、shape、stride、offset 與 tile shape 包裝起來，並讓 `tl.load`
產生 boundary check：

```python
x_block = tl.make_block_ptr(
    base=x_ptr, shape=(M, N), strides=(stride_m, stride_n),
    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
)
x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
x_block = tl.advance(x_block, (0, BLOCK_N))
```

如果每個 lane 需要不同 predicate 或 irregular address，請使用 pointer tensor。
對規則的 strided tile，則使用 block pointer：它更能清楚表達意圖，但
`boundary_check` 只能取代矩形邊界的 mask，不能取代任意 causal 或 sparse mask。

**Tensor descriptor** 更進一步描述 global tensor，並透過 descriptor operation
搬移 tile：

```python
desc = tl.make_tensor_descriptor(
    x_ptr, shape=[M, N], strides=[stride_m, stride_n],
    block_shape=[BLOCK_M, BLOCK_N],
)
x = desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N])
desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], x)
```

Descriptor 讓具備 descriptor-driven transfer 的 backend（例如 NVIDIA TMA）
得以使用這類功能。相較於一般 pointer，它對 alignment、tile shape 與 target
有更嚴格的要求，而且各 backend 的支援程度不同。除非部署硬體固定，否則應保留
以 pointer 為基礎的路徑；也應在實際使用的 Triton release 與 device 上驗證，
不要假設 descriptor 一定會產生某個特定 instruction。

### 2.6 語言中沒有什麼

- 一般程式碼中**沒有 shared memory 或 barrier**：program 內的資料交換透過
  block operation（`tl.sum`、`tl.dot`、`tl.trans`、reshape）完成，編譯器會
  視需要以 shuffle 或 shared memory 實作。
- 除了 global memory 與 atomic（`tl.atomic_add`、`tl.atomic_cas` 等），
  **program 間不能通訊**，與 CUDA 相同。
- Kernel 內**沒有 dynamic shape**：runtime length 的 row 會用 power-of-two
  block 加 mask 處理，或用 loop 走過多個 block。

## 3. 元素級工作與融合：向量加法

[`vector_add.py`](examples/14-triton/vector_add.py) 用十行就呈現完整 model：

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

Grid 有 $\lceil n / B \rceil$ 個 program。`BLOCK = 1024`、`num_warps = 4`
時，128 個 thread 各自負責 8 個 element；若 alignment 允許，編譯器可將其
lowering 成寬且 coalesced 的 access。這個 kernel 對 FP32 的最低 memory
traffic 是 $12n$ byte：每個 element 讀取兩個 4-byte value，再寫入一個
4-byte value。

### 3.1 是否融合取決於記憶體流量

Elementwise operation 很容易組合，因為每個 intermediate 都會留在 block
value 中：

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
out = tl.maximum(x + y, 0.0) * scale
tl.store(out_ptr + offsets, out, mask=mask)
```

如果拆成不同 kernel，add、ReLU 與 scale 會反覆寫入、讀回同一個 vector。
融合後所需的流量與單純 addition 相同：讀取兩個 input、寫入一個 output。
不過 fusion 也有限度：太多同時存活的 block value 會增加 register pressure、
降低 occupancy，甚至 spill 到 local memory。當 producer-consumer chain
能省下可觀流量時可將它融合，之後再檢查 register 用量並 benchmark。

## 4. 歸約與正規化：融合 Softmax 和 LayerNorm

[`softmax.py`](examples/14-triton/softmax.py) 以每 row 一個 program 計算
row-wise softmax：

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

### 4.1 編譯器產生什麼

`tl.max(x, axis=0)` 是第 03 章的 block reduction：per-thread partial
maximum、warp shuffle tree，以及 warp 間的 shared-memory exchange，全都由
一次呼叫導出。Masked lane 會載入 −∞，因此不會改變 maximum，也因
$e^{-\infty} = 0$ 而不影響 sum。

### 4.2 為何快速

Row 只讀取一次到 register，並寫入一次：每 row 為
$2 \cdot 4 \cdot n$ byte，已是最低值。若把 max、exponentiate、sum 拆開，
就必須重讀資料或把 intermediate 寫入 memory。Fused kernel 的優勢並非
Triton 特有，但 Triton 讓這種 fusion 成為自然的寫法。

### 4.3 限制

`BLOCK = next_power_of_2(n_cols)` 必須放得進 register。Wrapper 對較長的 row
使用更多 warp，讓長度最高到 16 384 column 時，每個 thread 最多保存約 32 個
value；超過數萬個 column 後，kernel 就會 spill。解法是 online softmax：
以 block 為單位 loop 整個 row，並把 $(m, z)$ 保存為 state（練習 2）。

### 4.4 Kernel 3：融合 LayerNorm

[`layer_norm.py`](examples/14-triton/layer_norm.py) 示範多次 reduction，
接著做 elementwise affine transform：

$$
\mu={1\over C}\sum_j x_j,\qquad
\sigma^2={1\over C}\sum_j(x_j-\mu)^2,\qquad
y_j={x_j-\mu\over\sqrt{\sigma^2+\epsilon}}\,w_j+b_j.
$$

```python
x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
mean = tl.sum(x, axis=0) / n_cols
centered = tl.where(mask, x - mean, 0.0)
variance = tl.sum(centered * centered, axis=0) / n_cols
y = centered * tl.rsqrt(variance + eps)
y = y * tl.load(weight_ptr + cols, mask=mask, other=0.0)
y += tl.load(bias_ptr + cols, mask=mask, other=0.0)
tl.store(out_ptr + row * out_stride + cols, y, mask=mask)
```

第二個 `tl.where` 不可省略：masked load 對 mean 貢獻的是零，但
`0 - mean` 仍會影響 variance。這個 kernel 只會各讀一次 input、weight 與
bias，再寫入一次結果；它不會具體寫出 mean、variance 或 normalized
activation。累加使用 FP32。對非常寬的 row，請使用 tiled Welford state
`(count, mean, M2)` 或 multi-kernel reduction，不要把整個 row 都留在 register。

## 5. 矩陣乘法與自動調校

[`matmul.py`](examples/14-triton/matmul.py) 計算 $C = AB$，每個 program
負責 $C$ 的一個 $B_M \times B_N$ tile：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k0 in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=(rows[:, None] < M) & (ks[None, :] + k0 < K), other=0.0)
    b = tl.load(b_ptrs, mask=(ks[:, None] + k0 < K) & (cols[None, :] < N), other=0.0)
    acc = tl.dot(a, b, acc)                    # acc += a @ b on tensor cores
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

這是標準的 tiled matrix multiplication 結構。和低階實作的差異，在於其中
哪些工作交由編譯器完成。

### 5.1 Triton 的矩陣乘法最佳化階梯

| 技術 | 在 Triton 中 |
|---|---|
| [矩陣乘法 2 – 向量化載入](gemm/01-vectorized-loads.md) | Stride 與 alignment 允許時自動完成 |
| [矩陣乘法 3 – 雙緩衝](gemm/02-double-buffering.md) | 自動 pipeline，受 `num_stages` 影響 |
| [矩陣乘法 4 – 非同步複製](gemm/03-async-copies.md) | 由 backend／編譯器選擇；descriptor 可提供支援 TMA 的資料搬移 |
| [矩陣乘法 5 – Warp 分塊](gemm/04-warp-tiling.md) | `tl.dot` layout 會把 tile 分配到 `num_warps` 個 warp |
| [矩陣乘法 6 – 分塊 Swizzle](gemm/05-tile-swizzling.md) | Shared-memory swizzle 自動完成；output tile 順序由你負責（`GROUP_M`） |
| [矩陣乘法 7 – Split-K 與 Stream-K](gemm/06-split-k-stream-k.md) | 由你負責：增加沿 K 的工作，再 reduction 或使用 atomic |
| [矩陣乘法 8 – Tensor Core](gemm/07-tensor-cores.md) | 使用受支援的 `tl.dot` shape 與 dtype 時自動完成 |

### 5.2 分組 Tile 順序

Program 大致會依 ID 順序執行。採 row-major order 時，同一時間 in flight 的
program 會涵蓋一或兩個 tile row，因而跨越 $B$ 的**所有** tile column，
使 $B$ 被反覆從 DRAM 讀取。把 $G$ 個 tile row 分為一組，可讓同時 in flight
的 program 改為涵蓋 $C$ 中近似正方形的區域：

$$
g = \left\lfloor \frac{p}{G\,T_N} \right\rfloor, \quad
G' = \min(T_M - gG,\ G), \quad
m = gG + \big(p \bmod G T_N\big) \bmod G', \quad
n = \left\lfloor \frac{p \bmod G T_N}{G'} \right\rfloor
$$

| 符號 | 意義 |
|---|---|
| $p$ | Program id |
| $T_M,\ T_N$ | $C$ 的 tile row 數與 tile column 數 |
| $G$ | `GROUP_M`，每組的 tile row 數 |
| $g$ | Program $p$ 所在的 group |
| $G'$ | 此 group 的 tile row 數（最後一組可能較短） |
| $m,\ n$ | Program $p$ 計算的 $C$ tile |

若 $T_N = 32$ 且有 64 個 program in flight，row-major order 會觸及 2 個
$A$ row strip 與 32 個 $B$ column strip（共 34 個）；$G = 8$ 時則各觸及
8 個（共 16 個），因此 L2 footprint 約減半，L2 hit 也會相應增加。這與
[矩陣乘法 6 – 分塊 Swizzle](gemm/05-tile-swizzling.md)
使用的是同一個概念。

### 5.3 自動調校

最佳 tile shape 取決於 GPU、data type 與 matrix size。`triton.autotune`
會針對一組 configuration list 編譯 kernel，在第一次遇到新的 `key` 時逐一
計時，並 cache 最快的選項：

```python
CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    ...
]
matmul_kernel_tuned = triton.autotune(configs=CONFIGS, key=["M", "N", "K"])(matmul_kernel)
```

| 參數 | 取捨 |
|---|---|
| `BLOCK_M`, `BLOCK_N` | 較大的 tile：每個載入 byte 有更多 reuse、使用更多 register、program 較少 |
| `BLOCK_K` | 較大：loop iteration 較少、每個 stage 使用更多 shared memory |
| `num_stages` | 更多 stage 可隱藏更多 latency，代價是 `num_stages` × tile byte 的 shared memory |
| `num_warps` | 每個 program 使用更多 warp：per-warp tile 較小、能隱藏更多 latency、ILP 較低 |

每個 program 使用的 shared memory 約為
$\text{num\_stages} \cdot (B_M + B_N) \cdot B_K \cdot \text{sizeof}$；
超過 SM capacity 的 configuration 會編譯失敗並被略過。

### 5.4 精度

輸入為 `float32` 時，`tl.dot` 在 Ampere 之後預設使用 TF32 tensor core
（10-bit mantissa、relative error 約為 $10^{-3}$），所以 test script 在 GPU
上會放寬 tolerance。`tl.dot(a, b, acc, input_precision="ieee")` 可強制使用
完整 FP32，但速度代價很高。對 `float16`/`bfloat16` input，accumulator 仍是
`float32`，而 epilogue（`acc.to(c_ptr.dtype.element_ty)`）只會在最後轉換一次。

### 5.5 Epilogue 融合

Store 前對 `acc` 做任何 elementwise 操作，都不會增加 memory traffic：
bias add、activation（`tl.where(acc > 0, acc, 0.0)`）、scale，或轉成 FP8。
手寫 Triton GEMM 最常在這裡勝過先呼叫 library、再執行獨立 elementwise
kernel 的做法。

## 6. Attention 與線上 Softmax：FlashAttention

[`flash_attention.py`](examples/14-triton/flash_attention.py) 實作單一 head
的 forward pass。每個 program 負責 `BLOCK_Q` 個 query row，並讓 $K$ 與 $V$
依序流過：

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

逐行來看，這就是 online-softmax recurrence：

| 程式 | 意義 |
|---|---|
| `s = tl.dot(q, tl.trans(k))` | 一個 tile 的 $S = Q K^\top$，scale 已折入 $Q$ |
| `m_new`、`alpha` | 新的 running maximum，以及 rescaling factor $e^{m_{\text{old}} - m_{\text{new}}}$ |
| `m_safe` | 保護只看過 masked key 的 row（$-\infty - (-\infty)$） |
| `l`、`acc` | Running sum 與未 normalization 的 output；先 rescale，再 update |
| `acc / l[:, None]`（loop 後） | 最終 normalization |

在低階實作中，大量程式碼都用來把 tile 分配到 lane，並透過 shared memory
stage $K$ 與 $V$；這裡用 `tl.dot` 表達兩次 matrix product，再由編譯器選擇
lowering 方式。Production kernel 還會加入 backward pass、以額外 grid axis
表達多 head 與 batch，以及 Hopper 上的 warp specialization，但核心仍是這個
loop。

在 causal case 中，`kv_end` 會讓 loop 停在該 block query 能看到的最後一個
key，因此工作量減半，與 CUDA kernel 完全相同。

## 7. 迴圈、Atomic 與 Scan

### 7.1 迴圈

Matmul 與 attention kernel 已經使用了 loop。如果 loop bound 完全在編譯期
已知，loop 可能會 unroll；若 bound 取決於 `K` 或 `n`，則會在產生的程式碼中
保留為 loop。`tl.range` 可指定 software-pipeline staging 等 loop attribute：

```python
for k0 in tl.range(0, K, BLOCK_K, num_stages=3):
    ...
```

Unroll 短 loop 可以暴露 instruction-level parallelism，但 unroll 長 loop
會增加 code size 與 live range。資料量決定的工作應使用 runtime loop；只有
少數固定 size 確實很常出現時，才值得為它們 specialize。

### 7.2 Kernel 6：Atomic Histogram

一般 launch 中的 program 不能彼此 synchronize。若要更新共用的 global state，
安全的做法是使用 atomic。完整的
[`histogram.py`](examples/14-triton/histogram.py) 能處理不整齊的 tail，
也會忽略超出 bin range 的 value：

```python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
in_bounds = offsets < n
values = tl.load(values_ptr + offsets, mask=in_bounds, other=-1)
valid = in_bounds & (values >= 0) & (values < n_bins)
bins = tl.where(valid, values, 0)  # masked pointer arithmetic stays in range
tl.atomic_add(histogram_ptr + bins, 1, mask=valid)
```

對 integer addition 而言，結果是 deterministic，但效能不是：若大量 value
落在同一個 bin，就會在同一個 address 上 serialize。Production histogram
通常會讓每個 program 建立一份 private histogram，再用第二個 kernel 合併。
Triton 也提供 compare-and-swap、exchange、min、max 與 bitwise atomic；
支援的 dtype 與 memory semantic 取決於 target。

### 7.3 Scan

Reduction 會把一個 block 對應成一個 value；**scan** 則會回傳每個 prefix。
`tl.cumsum(x, axis=0)` 是 inclusive sum scan，`tl.associative_scan` 則支援
自訂 associative combine operation。這些操作只作用於 block 內；若 scan
長度超過一個 program 的 tile，就需要 hierarchical algorithm：

1. 對每個 tile 做 scan，並儲存它的 total；
2. 對所有 tile total 做 scan；
3. 把前一個 tile 的 total 加到每個 tile。

這幾個 phase 之間沒有 grid-wide barrier，因此應使用不同 launch；只有在
明確的 persistent design 提供安全 protocol 時，才可考慮其他作法。

## 8. Persistent Kernel 與分組排程

Grouped scheduling 改變的是**順序**：第 5.2 節把相鄰的 program ID 對應到
對 cache 友善的 output tile 區域。Persistent kernel 改變的是**生命週期**：
它大約為每個 compute unit launch 一個 resident program，再由每個 program
處理多個 logical tile：

```python
pid = tl.program_id(0)
for tile_id in tl.range(pid, n_tiles, tl.num_programs(0)):
    # map tile_id to coordinates, load, compute, store
    ...
```

Host 端可以 launch `grid=(min(NUM_SMS, n_tiles),)`，其中 `NUM_SMS` 來自 device
property。這能減少一波波 launch scheduling 的成本，也讓一個 program 保留
可重用的 state。它適合 small-tile workload、stream-K design 與 fused
pipeline，但並非放諸四海皆準的加速方式：

- 一個 persistent program 不可使用過多 register 或 shared memory，以免能
  resident 的 program 太少；
- tile 工作量不同時，靜態 round-robin assignment 可能無法妥善 load balance；
- atomic work queue 可平衡不規則工作，但會增加 contention；
- spin-wait protocol 若等待的是尚無法排程的 program，可能造成 deadlock；
  絕不可假設所有 logical program 都能同時 resident；
- 每個 backend 與 architecture 都要分別調校 resident grid。

請先從一般 grouped scheduling 開始。只有 profile 顯示 launch、tail wave
或 cache residency 成本確實能由 persistence 解決時，再改用 persistent
設計。

## 9. 編譯器階段、Layout 與 Warp Specialization

### 9.1 從 Python 到 Machine Code

![編譯器將 decorated Python function 依序 lowering 成 Triton IR（block operation）、TritonGPU IR（layout、shared memory、pipeline）與 LLVM IR，最後成為 PTX 或 AMDGCN](figures/ch14-compiler.svg)

第一次使用新的 constexpr value、argument dtype 與 **specialization** 組合
launch 時會編譯 kernel；除此之外，Triton 也會檢查 pointer 與 integer
argument 是否可被 16 整除，以證明 vectorized load 所需的 alignment。
後續 launch 會命中 in-memory 與 on-disk cache。Launch 會回傳一個 handle，
其 `asm` dictionary 保存每個 stage：

```python
handle = add_kernel[grid](x, y, out, n, BLOCK=1024)
print(handle.asm["ttgir"])    # layouts chosen by the compiler
print(handle.asm["ptx"])      # look for ld.global.v4.f32 (vectorised loads)
```

TritonGPU IR 能回答效能相關問題：它會顯示每個 tensor 的 layout
（`#blocked`、`#mma`、`#shared`）、shared memory 配置的位置，以及建立了
多少 pipeline stage。

### 9.2 Layout 與 Warp Specialization

**Layout** 說明 block value 的各 element 分別由哪些 lane 與 warp 負責。
Blocked layout 適用於一般 elementwise 工作；dot-operand 與 MMA layout
用來餵給 matrix unit；shared layout 則描述 staged data。Layout conversion
可能需要 shuffle 或經由 shared memory 往返，因此，當看似無害的 transpose
或 reshape 造成效能退步時，請檢查 TritonGPU IR。

`num_warps` 控制有多少 warp 合作執行一個 program；它不會讓你手動替每個
warp 指派一片 tensor。編譯器會根據 layout 選擇 mapping。相較之下，
**warp specialization** 會讓不同 warp group 負責不同角色，例如 producer
warp 負責搬移 tile，consumer warp 則執行 matrix instruction。在支援的 Triton
target 上，某些 pipelined loop 可透過 `tl.range` 的 `warp_specialize` 選項
提出這項要求。這是進階且取決於 target 的最佳化：可用功能與合法 layout
會隨 Triton 版本改變，而 NVIDIA 特有的 warp-group 機制也不能直接轉移到
AMD wavefront。請保留 non-specialized configuration，並只選用實測較快的版本。

## 10. CUDA 與 ROCm 可攜性

相同的 pointer arithmetic、mask、reduction 與大部分 `tl.dot` 程式碼，都可
同時編譯到 CUDA 與 ROCm backend。Performance model 可以共用，但最佳常數
通常不能。

| 考量 | 可攜性原則 |
|---|---|
| Warp／wave width | 不要寫死 32-lane 假設；以 block 與 reduction 表達工作 |
| Matrix instruction | 使用受支援的 dtype，並 benchmark 適合各 backend 的 `BLOCK_K` 與 tile shape |
| Shared memory／LDS | Capacity、bank behavior 與 occupancy 不同；重新調校 `num_stages` 與 `num_warps` |
| Descriptor 與 async copy | 把 TMA 與其他 target-specific 路徑視為可選的 fast path |
| Atomic | 驗證 dtype 與 operation 支援，尤其是 low-precision 與 64-bit case |
| Math | 近似 `exp`、division 與 TF32 behavior 可能需要 backend-specific tolerance |
| Profiler | CUDA 使用 Nsight Systems／Compute；ROCm 使用 `rocprof`／Omniperf |

讓 algorithmic code 保持共用，並在 Python wrapper 中選擇一小組
backend-specific configuration。請在兩種 backend 的實際 compiled mode
中測試：interpreter 只能驗證 indexing semantic，無法驗證 target code
generation、matrix-instruction selection 或 asynchronous pipeline。

## 11. 測試、Interpreter、除錯與效能分析

[`test_kernels.py`](examples/14-triton/test_kernels.py) 會把六個 kernel 都與
PyTorch 比較。測試案例包含 wrapper 不 launch 的 empty input、singleton
shape、不整齊的 tail、剛超過 tile boundary 的 dimension、causal mask、
超出範圍的 histogram bin，以及嚴重的 atomic collision。Production test
suite 還應涵蓋每個受支援 dtype、API 承諾支援的 non-contiguous layout、
extreme value、NaN/Inf policy、多個 seed，以及每個實際部署的 backend。

請在 **import Triton 前**設定 `TRITON_INTERPRET=1`，讓 program instance
透過 NumPy 在 CPU 上執行。這非常適合檢查 pointer arithmetic 與 mask，也可
使用一般的 `print` 與 `pdb`。它不會模擬 parallel race、GPU floating-point
細節、layout、occupancy、target instruction 或效能。Interpreter test 通過
是必要證據，但不能取代 GPU validation。

### 11.1 除錯

| 工具 | 用途 |
|---|---|
| `TRITON_INTERPRET=1` | 使用 NumPy 在 CPU 執行；kernel 內可用 `print()` 與 `pdb` |
| `tl.device_print("x", x)` | 從 compiled kernel 印出（每個 program 都會印；請用 mask 或小 grid 限制） |
| `tl.static_print`、`tl.static_assert` | 在編譯期印出或檢查 constexpr value |
| `tl.device_assert(cond, "msg")` | Runtime assertion（以 `TRITON_DEBUG=1` 啟用） |

常見錯誤是靜態的：`arange` bound 不是 2 的次方、shape 無法 broadcast、
編譯器需要 constant 卻收到 non-constexpr value，以及 `tl.dot` 的 operand
在某個 dimension 小於 16。

若 edge value 不正確，請把 grid 縮小到一個 program，並印出 offset、mask 與
載入的 value。若發生 crash，先在 interpreter 中執行極小且 size 不整齊的
shape，再使用 backend 的 GPU memory checker。若出現 numerical error，
請以 FP32 比較 intermediate state，並明確判斷差異來自 algorithm order、
approximate math、TF32，還是錯誤的 mask。

### 11.2 Benchmark 與效能分析

`triton.testing.do_bench(fn)` 會先 warm up，再重複執行，並在 run 之間
flush L2，最後回傳毫秒；`triton.testing.perf_report` 會 sweep size 並繪圖。
對 Nsight Compute 而言，Triton kernel 就是一般 GPU kernel
（`ncu -k regex:matmul_kernel python3 test_kernels.py --bench`）；由於預設
啟用類似 `-lineinfo` 的資訊，source attribution 會指向 Python 行。

Benchmark 時應先 warm up kernel，以免把 compilation 與 autotuning 算進去。
報告中應包含 shape、dtype、stride、backend、GPU、Triton version 與選中的
configuration。除了衍生的 bandwidth 或 FLOP/s，也要比較 latency；更改 tile
前請先 profile，因為 low occupancy、spill、memory stall、layout conversion
與 launch gap 需要不同的解法。

## 12. 生產環境決策檢查表

| 情境 | 選擇 |
|---|---|
| 標準 shape 的標準 GEMM、convolution 或 attention | 函式庫（cuBLAS、cuDNN、hipBLASLt、FlashAttention） |
| 函式庫沒有的 fused operation（GEMM + 自訂 epilogue、新 attention variant、fused norm） | Triton |
| 必須在 NVIDIA 與 AMD 執行的研究程式 | Triton |
| 最後一點 architecture-specific 效能需要自訂 synchronization 或 data movement | CUDA、HIP、CUTLASS 或 CuTe |
| 由不規則 per-thread control flow 主導的 algorithm（sorting network、graph traversal） | CUDA |

在 Triton kernel 上線前，請確認以下問題都有答案：

- **價值：**Fusion、specialization 或新 algorithm，是否能在具代表性的
  production shape 上勝過最合適的現有函式庫？
- **契約：**Wrapper 是否檢查 shape、dtype、stride、alignment、device、
  aliasing 與 empty input behavior？
- **正確性：**是否已針對 odd tail、masked row、extreme value、NaN、atomic
  與 numerical tolerance，和 high-precision reference 比較？
- **涵蓋範圍：**對不支援的 shape、dtype、backend 與失敗的 autotune
  configuration，是否有安全的 fallback？
- **調校：**Key 是否夠具體，不會重用不合適的 configuration，同時也有合理
  上限，避免 first-use autotuning 與 cache 無止境成長？
- **資源：**Compiler output 與 profile 是否顯示可接受的 register、
  shared memory 與 occupancy，而且沒有意外 spill 或 layout conversion？
- **營運：**是否已處理 compilation／autotune warm-up、cache behavior、
  Triton 與 driver versioning、observability 和 rollback？
- **可攜性：**每個宣稱支援的 CUDA／ROCm architecture，是否都在 compiled
  mode 下跑過 correctness 與 performance test？

在量測證明需要 custom kernel 前，優先使用函式庫。選擇能達成目標的最簡單
Triton 設計，並保留 framework 或 library fallback。

## 重點整理

1. Triton kernel 描述一個 program 如何操作形狀在編譯期決定的 block；
   編譯器會將這些 block 對應到 thread 與 target instruction。
2. Pointer block 加 mask 取代 thread indexing 與 bounds check；`other` 提供
   masked lane 所需的 identity。
3. Coalescing、vectorization、shared-memory staging、pipelining、swizzle
   與 tensor-core instruction 由編譯器完成；tile size、tile order 與 fusion
   仍由你決定。
4. `triton.autotune` 會依 problem size 搜尋 tile shape、`num_warps` 與
   `num_stages`。
5. Loop 用來表達 tiled algorithm；atomic 透過 global memory 通訊；scan 只
   作用於 block 內，跨 block 時需要 hierarchical design。
6. Persistent 與 warp-specialized kernel 是經量測後採用、且取決於 target
   的最佳化，不是起點。
7. Interpreter 在 CPU 上檢查 indexing semantic；compiled GPU test、
   IR inspection 與 backend profiler 才能確認 correctness 與 performance。

## 練習

1. **簡單 — mask 與 shape。**在 `add_kernel` 中，如果 `BLOCK` 是 1000
   會發生什麼事？如果 load 省略 `mask` 呢？

    <details markdown="1"><summary>答案</summary>

    `tl.arange(0, 1000)` 無法編譯：block size 必須是 2 的次方。沒有 mask
    時，最後一個 program 會越界讀取 `x` 與 `y`（得到 undefined value；
    `compute-sanitizer` 會回報 out-of-bounds read）；store 若仍有自己的
    mask，結果可能看似正確，但 kernel 其實是錯的。

    </details>

2. **簡單 — masked reduction。**在 `layer_norm_kernel` 中，移除
   `tl.where(mask, x - mean, 0.0)`，直接使用 `x - mean`。為什麼即使 input
   load 使用 `other=0.0`，odd width 還是會失敗？

    <details markdown="1"><summary>答案</summary>

    Padded lane 載入的是零，但 centering 後會變成 `-mean`，而不是零。因此
    每個 padded lane 都會把 `mean²` 加進 variance。每次 reduction 都必須
    在當下套用正確的 identity；load 對 mean 使用的 identity，在做過
    subtraction 後不會自動繼續是 identity。

    </details>

3. **中等 — online reduction。**為長到無法放入 register 的 row 撰寫
   softmax kernel：以 `BLOCK` column 為單位 loop 整個 row，使用第 6 節的
   recurrence 保存 running maximum 與 sum，再 loop 一次寫入 output。

    <details markdown="1"><summary>提示</summary>

    第一個 loop 中保存 shape `(BLOCK,)` 的 per-lane vector `m` 與 `z`
    （`m_new = tl.maximum(m, x)`，
    `z = z * tl.exp(m - m_new) + tl.exp(x - m_new)`，並加上 −∞ guard）；
    最後以 `M = tl.max(m, 0)`、`Z = tl.sum(z * tl.exp(m - M), 0)` 合併，
    第二個 loop 再 store `tl.exp(x - M) / Z`。不論 row length 為何，都是
    兩次 read、一次 write。

    </details>

4. **中等 — 資源計算。**當 $M = N = K = 4096$、使用 FP16，
   `BLOCK_M = BLOCK_N = 128`、`BLOCK_K = 32`、`num_stages = 3` 時，一個
   program 需要多少 shared memory？A100 SM（164 KB）可容納幾個 program？

    <details markdown="1"><summary>答案</summary>

    $3 \cdot (128 + 128) \cdot 32 \cdot 2 = 49\,152$ byte = 48 KB，因此依
    shared memory 計算可容納三個 program。若 `num_warps = 8`（256 個
    thread），128 × 128 FP32 accumulator（每個 thread 64 個 value）加上
    operand，通常會讓 register file 把數量限制為一或兩個。

    </details>

5. **中等 — epilogue fusion。**在 `RELU: tl.constexpr` flag 控制下，
   為 `matmul_kernel` 加入 fused ReLU epilogue，並與
   `torch.relu(a @ b)` 比較。為什麼 constexpr 優於 runtime flag？

    <details markdown="1"><summary>答案</summary>

    Store 前加入 `if RELU: acc = tl.maximum(acc, 0.0)`。作為 constexpr，
    每個 value 都會編譯成自己的 kernel，branch 也會消失；runtime flag
    會在每個 program 中保留一個（uniform 且便宜的）branch，並使編譯器
    無法 specialize epilogue。

    </details>

6. **進階 — privatized atomic。**把 histogram 改成兩個 kernel：第一個
   為每個 program 寫入一份 private histogram，第二個再 reduction 這些
   histogram。它在什麼情況下會比直接使用 atomic 快？

    <details markdown="1"><summary>提示與答案</summary>

    在 `(num_programs, n_bins)` temporary 中為每個 program 分配一個 row，
    先在本地累加（必要時可在該 row 內使用 atomic），再沿所有 program row
    加總每個 bin。這會增加 temporary traffic 與一次 launch，因此，只有當
    所有工作集中在單一 global histogram 所造成的 collision 與 serialize
    成本更高時，才會勝出。請對 skewed 與 uniform distribution 都做
    benchmark；bin count 與 dtype 會顯著影響交叉點。

    </details>

7. **進階 — masked attention。**`m_safe` 只在某個 row 目前只看過 masked
   key 時才有作用。請證明目前的 `flash_attention` 不會發生此事，並舉出
   一種會發生的 attention variant。

    <details markdown="1"><summary>答案</summary>

    第一個 tile 永遠從 key 0 開始，對每個 query row 都有效：它存在
    （$n \ge 1$），而且在 causal mask 下不是任何 row 的未來 key（超出
    $n$ 的 padding row 也看得到它）。因此第一個 tile 後，每個 `m` 都是
    finite。Sliding-window attention（row $r$ 只看
    $r - w, \dots, r$，所以較晚 row 的早期 tile 會全被 mask）、
    key-padding mask，或用不同順序走訪 tile（例如從 diagonal 開始）都會
    發生此情況。沒有 guard 時，`s - m_new` 會變成
    $-\infty - (-\infty) = \text{NaN}$。

    </details>

8. **進階 — global scan。**請為任意長度的 vector 設計 scan。為什麼在
   一般 kernel 中，不能用 spin-waiting grid-wide barrier 取代三次 launch？

    <details markdown="1"><summary>答案</summary>

    先 launch block-local scan 並儲存 tile total；再遞迴 scan 所有 total；
    最後 launch 一個 kernel，把前一個 tile 的 total 加到每個 tile。
    Spin-wait barrier 可能 deadlock：resident program 可能正在等待 logical
    program，而後者必須等前者退出後才能排程。只有 launch 被刻意限制在
    可同時 resident 的 worker 數量內，且 memory protocol 正確時，
    persistence 才能實作 custom barrier；使用不同 launch 才是安全的預設。

    </details>

## 實作練習

LeetGPU 與 Tensara 接受 Triton submission；以下問題適合用來初次移植上述
kernel：

- [LeetGPU – Vector Addition](../leetgpu/001-vector-add/)
- [LeetGPU – Softmax](../leetgpu/005-softmax/)、[Tensara – Softmax](../tensara/softmax/)
- [LeetGPU – Matrix Multiplication](../leetgpu/002-matrix-multiplication/)、
  [Tensara – Matrix Multiplication](../tensara/matrix-multiplication/)
- [LeetGPU – Softmax Attention](../leetgpu/006-softmax-attention/)、
  [LeetGPU – Causal Attention](../leetgpu/053-casual-attention/)
- [Tensara – Layer Norm](../tensara/layer-norm/)（類似第 4 節的 fused reduction）
