# 14 – Triton：從第一個 kernel 到生產環境

> **第四部 · Triton** · 一份從入門到進階的獨立指南 ·
> 程式與測試：[`examples/14-triton/`](examples/14-triton/test_kernels.py)

Triton 是一套用來撰寫 GPU kernel 的 Python 語言與編譯器。你只要描述一個 **program 實例**如何處理向量或分塊，編譯器就會把這些工作對應到 GPU 執行緒、向量載入、共享記憶體與加速器指令。閱讀本章不需要 CUDA 經驗；若借用 CUDA 的說法會比較好懂：一個 Triton program 實例大致相當於一個 CUDA 執行緒區塊，網格就是所有區塊的集合，而區塊值（block value）則是形狀在編譯期就決定、分散在各執行緒上的張量。

更高階的抽象並不代表不需要效能工程。工作如何切分、網格大小、分塊大小、遮罩、走訪順序、融合的邊界與啟動參數，仍然由你決定；Triton 替你處理的是大部分機械性的執行緒對應與底層轉換。

**你將學到**

- Triton 的程式設計模型：program、區塊、遮罩與指標分塊；
- 六個完整的 kernel：向量加法、融合 softmax、融合 LayerNorm、使用分組順序與自動調校的矩陣乘法、FlashAttention，以及以原子操作實作的直方圖；
- 矩陣乘法的哪些最佳化由編譯器完成、哪些仍由你負責；
- 迴圈、原子操作、掃描、常駐（persistent）排程、編譯器的資料配置（layout），以及在 NVIDIA 與 AMD 之間的可攜性；
- Triton kernel 如何被編譯、特化、快取、檢視、在直譯器中除錯、做基準測試與效能分析；
- 何時該選擇 Triton、CUDA 或函式庫。

## 1. 為什麼要用區塊層級的語言

### 1.1 哪些工作從你手上交給了編譯器

| 項目 | CUDA | Triton |
|---|---|---|
| 程式的單位 | 一個執行緒 | 一個 program 實例（一組執行緒） |
| 資料 | 暫存器中的純量 | 形狀在編譯期決定的區塊張量 |
| 執行緒 ↔ 元素的對應 | 你 | 編譯器（一種*資料配置*） |
| 合併存取、向量寬度 | 你 | 編譯器，依指標模式與對齊推斷 |
| 共享記憶體、barrier | 你 | 編譯器 |
| 避免 bank 衝突的 swizzle | 你 | 編譯器 |
| 多階段載入管線 | 你 | 編譯器，受 `num_stages` 影響 |
| Tensor core 指令 | 你 | 編譯器，由 `tl.dot` 產生 |
| 分塊大小、網格、分塊順序 | 你 | 你 |
| 融合（一個 kernel 要做多少事） | 你 | 你 |

這是一種取捨：刻意限制了逐執行緒的底層控制，換來的是通常短得多、而且能同時用於 NVIDIA 與 AMD GPU 的 kernel。不過可攜並不等於效能自動相同：每個後端仍需要以具代表性的情境測試與調校。

![CUDA 描述一個執行緒，對應方式由你決定；Triton 描述一個區塊，由編譯器把它配置到各個 warp 上](figures/ch14-model.svg)

### 1.2 環境設定

```bash
pip install torch triton          # Triton ships with PyTorch's CUDA wheels too
cd tutorials/examples/14-triton
python3 test_kernels.py           # checks all six kernels against PyTorch
python3 test_kernels.py --bench   # and times them (GPU only)
```

沒有 GPU 時，`test_kernels.py` 會在匯入 Triton 之前設定 `TRITON_INTERPRET=1`。**直譯器**以 NumPy 依序執行每個 program 實例：速度很慢，但它執行的是相同的索引計算、遮罩與算術，相當於本儲存庫 CUDA 模擬器在 Triton 上的對應物。在 ROCm 上，只要安裝相容的 ROCm 版 PyTorch 與 Triton，同一份原始碼也能執行；各後端的注意事項見第 10 節。

## 2. 程式設計模型

### 2.1 Program 與網格

Triton kernel 是加上 `@triton.jit` 裝飾器的 Python 函式。它以 `kernel[grid](arg0, arg1, ...)` 的形式，在由 **program 實例**組成的網格上啟動；`grid` 是最多三個維度大小的 tuple，或是一個以 kernel 編譯期參數為輸入的函式：

```python
grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
kernel[grid](x, y, out, n, BLOCK=1024)
```

在 kernel 內，`tl.program_id(axis)` 是 program 的索引（相當於 CUDA 的 `blockIdx`），`tl.num_programs(axis)` 則是網格大小（`gridDim`）。這裡沒有 `threadIdx`：一個 program 就是一個區塊，至於由多少執行緒來執行它，則是啟動選項 `num_warps`（預設為 4）。

### 2.2 區塊與形狀限制

kernel 內的值不是純量，就是**區塊**：形狀在編譯期已知的張量。`tl.arange(0, BLOCK)` 產生向量 `[0, 1, …, BLOCK−1]`；使用這種寫法時，`BLOCK` 必須是 `tl.constexpr`，且區間長度必須是 2 的冪次。運算以 NumPy 的廣播規則逐元素進行（`x[:, None]`、`y[None, :]`）；歸約需要指定軸（`tl.sum`、`tl.max`、`tl.argmax`）；`tl.dot` 則把兩個二維區塊相乘。

`constexpr` 的每一個不同值都會編譯出一個獨立的 kernel，所以大小以關鍵字參數傳入：`BLOCK=1024`。

`n`、`M`、`N` 這類執行期大小可以是任意值：把分塊往上補到合法的編譯期形狀，再用遮罩排除超出執行期形狀的 lane。個別運算也有自己的限制，例如有效率的 `tl.dot` 分塊，維度必須與後端的矩陣指令相容。靜態形狀錯誤是編譯期錯誤，kernel 無法用分支繞過。

### 2.3 指標、載入、儲存與遮罩

張量參數以指向第一個元素的指標傳入。把一個位移區塊加到指標上，就得到一個**指標區塊**，而 `tl.load`/`tl.store` 會一次讀寫其中所有位置：

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n                               # the tail guard, for the whole block
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
tl.store(out_ptr + offs, x, mask=mask)
```

被遮罩排除的 lane 既不讀也不寫；`other` 是被遮罩的載入所傳回的值（請選擇後續運算的單位元：求和用 0，取最大值用 −∞）。

位移的單位是*元素*：Triton 會自己乘上元素大小。除非某個參數讓它們變成 64 位元，否則位移都是 32 位元整數；對超過 $2^{31}$ 個元素的張量，請先用 `pid.to(tl.int64)` 轉型。

### 2.4 二維分塊

二維的指標分塊，是一行列位移與一列欄位移的廣播和：

![rows[:, None] * S 加上 cols[None, :]，廣播成 BLOCK_M × BLOCK_N 的位址分塊；遮罩也以同樣方式建立](figures/ch14-pointer-block.svg)

$$
\text{ptr}_{rq} = \text{base} + \text{row}_r \cdot s_0 + \text{col}_q \cdot s_1,
\qquad r < B_M,\ q < B_N
$$

| 符號 | 意義 |
|---|---|
| $\text{base}$ | 指向矩陣元素 (0, 0) 的指標 |
| $\text{row}_r,\ \text{col}_q$ | `rows` 的第 r 個元素與 `cols` 的第 q 個元素 |
| $s_0,\ s_1$ | 矩陣的列步幅與欄步幅，以元素為單位（`x.stride(0)`、`x.stride(1)`） |
| $B_M,\ B_N$ | 分塊形狀（`BLOCK_M`、`BLOCK_N`） |

同時傳入兩個步幅，同一個 kernel 就能處理列優先、轉置過與切片過的矩陣。編譯器只會沿著它能證明是連續且對齊的維度產生向量化載入；它從存取模式，以及針對參數值所做的特化（第 9.1 節）得知這些性質。若某個步幅恆為 1，就像 softmax kernel 那樣把它從函式簽名中拿掉，連續性便一目了然。

### 2.5 區塊指標與張量描述子

明確的指標張量是最通用的定址方式。**區塊指標**（block pointer）把基底、形狀、步幅、位移與分塊形狀包裝在一起，並讓 `tl.load` 自動產生邊界檢查：

```python
x_block = tl.make_block_ptr(
    base=x_ptr, shape=(M, N), strides=(stride_m, stride_n),
    offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
)
x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
x_block = tl.advance(x_block, (0, BLOCK_N))
```

當每個 lane 需要不同的判斷條件或不規則的位址時，使用指標張量；對規則的步幅分塊，則使用區塊指標：意圖更清楚。但 `boundary_check` 只能取代矩形邊緣的遮罩，無法取代任意的因果遮罩或稀疏遮罩。

**張量描述子**（tensor descriptor）更進一步：它描述一整個全域張量，並透過描述子的操作來搬移分塊：

```python
desc = tl.make_tensor_descriptor(
    x_ptr, shape=[M, N], strides=[stride_m, stride_n],
    block_shape=[BLOCK_M, BLOCK_N],
)
x = desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N])
desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], x)
```

描述子讓支援「由描述子驅動傳輸」的後端（例如 NVIDIA 的 TMA）得以使用這種硬體。它對對齊、分塊形狀與目標硬體的要求比一般指標嚴格，各後端的支援程度也不同。除非部署的硬體固定，否則請保留以指標為基礎的路徑，並在實際使用的 Triton 版本與裝置上驗證，而不要假設使用描述子就一定會產生特定指令。

### 2.6 語言中沒有的東西

- 一般程式碼中**沒有共享記憶體，也沒有 barrier**：program 內部的資料交換透過區塊運算完成（`tl.sum`、`tl.dot`、`tl.trans`、重塑形狀），編譯器視需要以 shuffle 或共享記憶體實作它們。
- 和 CUDA 一樣，**program 之間無法直接溝通**，只能透過全域記憶體與原子操作（`tl.atomic_add`、`tl.atomic_cas` 等）。
- kernel 內**沒有動態形狀**：長度在執行期才決定的列，要用 2 的冪次大小的區塊加上遮罩，或以迴圈逐塊處理。

## 3. 逐元素運算與融合：向量加法

[`vector_add.py`](examples/14-triton/vector_add.py) 用十行就展示了整個模型：

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

網格有 $\lceil n / B \rceil$ 個 program。`BLOCK = 1024`、`num_warps = 4` 時，128 個執行緒各負責 8 個元素；只要對齊條件允許，編譯器就能把它們轉成寬的合併存取。對 FP32 而言，這個 kernel 最少的記憶體流量是 $12n$ 位元組：每個元素兩次 4 位元組讀取與一次 4 位元組寫入。

### 3.1 融合是關於記憶體流量的決策

逐元素運算可以自然地串接，因為每個中間結果都留在區塊值中：

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
out = tl.maximum(x + y, 0.0) * scale
tl.store(out_ptr + offsets, out, mask=mask)
```

若把加法、ReLU 與縮放寫成三個 kernel，同一個向量就會被反覆寫出又讀回；融合之後，它們的記憶體流量與單純的加法相同：兩次輸入讀取、一次輸出寫入。但融合不是越多越好：同時存活的區塊值太多，會增加暫存器壓力、降低佔用率，甚至溢出到區域記憶體。當一條「生產者–消費者」鏈能省下可觀的流量時才融合，然後檢查暫存器用量並實際測量。

## 4. 歸約與正規化：融合 softmax 與 LayerNorm

[`softmax.py`](examples/14-triton/softmax.py) 以每列一個 program 的方式計算逐列 softmax：

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

### 4.1 編譯器產生了什麼

`tl.max(x, axis=0)` 就是第 03 章的區塊歸約：每個執行緒先求部分最大值，再做 warp 內的 shuffle 樹，最後透過共享記憶體在 warp 之間交換——這一切都由一個呼叫衍生出來。被遮罩的 lane 載入 −∞，因此既不影響最大值，也不影響總和（因為 $e^{-\infty} = 0$）。

### 4.2 為什麼快

每一列只讀進暫存器一次、寫出一次：每列 $2 \cdot 4 \cdot n$ 位元組，這是最低限度。把「取最大值 / 取指數 / 求和」拆開的設計，則必須重讀或寫出中間結果。融合 kernel 的優勢並非 Triton 獨有，但 Triton 讓這種融合成為最自然的寫法。

### 4.3 限制

`BLOCK = next_power_of_2(n_cols)` 必須放得進暫存器。包裝函式對較長的列會使用更多 warp，讓每個執行緒在 16 384 欄以內最多只持有約 32 個值；超過數萬欄時，kernel 就會發生暫存器溢出。解法是線上 softmax：以區塊為單位在列上迴圈，並以 $(m, z)$ 作為狀態（練習 3）。

### 4.4 Kernel 3：融合 LayerNorm

[`layer_norm.py`](examples/14-triton/layer_norm.py) 示範了多次歸約之後，再接一個逐元素的仿射轉換：

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

第二個 `tl.where` 不可或缺：被遮罩的載入對平均值的貢獻是零，但 `0 - mean` 仍會對變異數產生貢獻。這個 kernel 只讀取輸入、權重與偏差各一次，並只寫出結果一次；平均值、變異數與正規化後的激活值都不會寫到記憶體。累加以 FP32 進行。對非常寬的列，請改用分塊的 Welford 狀態 `(count, mean, M2)`，或多個 kernel 的歸約，而不要把整列留在暫存器中。

## 5. 矩陣乘法與自動調校

[`matmul.py`](examples/14-triton/matmul.py) 以 $C$ 的每個 $B_M \times B_N$ 分塊對應一個 program 的方式計算 $C = AB$：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k0 in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=(rows[:, None] < M) & (ks[None, :] + k0 < K), other=0.0)
    b = tl.load(b_ptrs, mask=(ks[:, None] + k0 < K) & (cols[None, :] < N), other=0.0)
    acc = tl.dot(a, b, acc)                    # acc += a @ b on tensor cores
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

這是標準的分塊矩陣乘法結構；與底層實作的差別，在於編譯器如何處理它。

### 5.1 矩陣乘法各步驟在 Triton 中的對應

| 技巧 | 在 Triton 中 |
|---|---|
| [矩陣乘法 2 – 向量化載入](gemm/01-vectorized-loads.md) | 步幅與對齊允許時自動完成 |
| [矩陣乘法 3 – 雙緩衝](gemm/02-double-buffering.md) | 自動管線化，受 `num_stages` 影響 |
| [矩陣乘法 4 – 非同步複製](gemm/03-async-copies.md) | 由後端 / 編譯器選擇；描述子可以開放支援 TMA 的搬移方式 |
| [矩陣乘法 5 – Warp 分塊](gemm/04-warp-tiling.md) | `tl.dot` 的資料配置把分塊分散到 `num_warps` 個 warp |
| [矩陣乘法 6 – 分塊重排](gemm/05-tile-swizzling.md) | 共享記憶體的 swizzle 自動完成；輸出分塊的順序由你決定（`GROUP_M`） |
| [矩陣乘法 7 – Split-K 與 Stream-K](gemm/06-split-k-stream-k.md) | 由你負責：沿 K 切分工作，再做歸約或使用原子操作 |
| [矩陣乘法 8 – Tensor Core](gemm/07-tensor-cores.md) | 使用受支援的 `tl.dot` 形狀與資料型別時自動完成 |

### 5.2 分組的分塊順序

program 大致依編號順序執行。若採列優先順序，任何時刻正在執行的 program 只涵蓋一兩列分塊，卻會碰到 $B$ 的*所有*分塊欄，因此 $B$ 會從 DRAM 被反覆讀取。把 $G$ 個分塊列分成一組，就能讓同時執行的 program 涵蓋 $C$ 中一塊接近正方形的區域：

$$
g = \left\lfloor \frac{p}{G\,T_N} \right\rfloor, \quad
G' = \min(T_M - gG,\ G), \quad
m = gG + \big(p \bmod G T_N\big) \bmod G', \quad
n = \left\lfloor \frac{p \bmod G T_N}{G'} \right\rfloor
$$

| 符號 | 意義 |
|---|---|
| $p$ | program 編號 |
| $T_M,\ T_N$ | $C$ 的分塊列數與分塊欄數 |
| $G$ | `GROUP_M`，每組的分塊列數 |
| $g$ | program $p$ 所屬的組 |
| $G'$ | 這一組的分塊列數（最後一組可能較少） |
| $m,\ n$ | program $p$ 計算的 $C$ 分塊 |

以 $T_N = 32$、同時有 64 個 program 執行為例：列優先順序會碰到 $A$ 的 2 條列帶與 $B$ 的 32 條欄帶（共 34 條）；$G = 8$ 時只碰到 $A$ 的 8 條與 $B$ 的 8 條（共 16 條），L2 的佔用量約減半，L2 命中也相應增加。這與[矩陣乘法 6 – 分塊重排](gemm/05-tile-swizzling.md)中的分塊順序重排是同一個概念。

### 5.3 自動調校

最佳的分塊形狀取決於 GPU、資料型別與矩陣大小。`triton.autotune` 會針對一組設定分別編譯 kernel，在第一次遇到新的 `key` 時逐一計時，並快取勝出者：

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
| `BLOCK_M`、`BLOCK_N` | 分塊越大：每載入一個位元組的重用越多，但暫存器越多、program 越少 |
| `BLOCK_K` | 越大：迴圈次數越少，但每個階段需要更多共享記憶體 |
| `num_stages` | 階段越多，能隱藏的延遲越多，代價是 `num_stages` × 分塊位元組數的共享記憶體 |
| `num_warps` | 每個 program 的 warp 越多：每個 warp 的分塊越小、越能隱藏延遲，但指令層級平行度越低 |

每個 program 的共享記憶體用量約為 $\text{num\_stages} \cdot (B_M + B_N) \cdot B_K \cdot \text{sizeof}$；超過 SM 容量的設定會編譯失敗，並被略過。

### 5.4 精度

對 `float32` 輸入，`tl.dot` 在 Ampere 及更新的架構上預設使用 TF32 tensor core（尾數 10 位元，相對誤差約 $10^{-3}$）；因此測試腳本在 GPU 上會放寬容許誤差。`tl.dot(a, b, acc, input_precision="ieee")` 會強制使用完整的 FP32，但速度損失很大。對 `float16`/`bfloat16` 輸入，累加器仍是 `float32`，由 epilogue（`acc.to(c_ptr.dtype.element_ty)`）在最後轉換一次。

### 5.5 Epilogue 融合

在寫出之前對 `acc` 做的任何逐元素運算，都不會增加記憶體流量：加偏差、激活函數（`tl.where(acc > 0, acc, 0.0)`）、縮放、轉成 FP8。手寫的 Triton GEMM 最常勝過「函式庫呼叫 + 另外幾個逐元素 kernel」的地方，就在這裡。

## 6. 注意力與線上 softmax：FlashAttention

[`flash_attention.py`](examples/14-triton/flash_attention.py) 實作單一 head 的前向計算。每個 program 負責 `BLOCK_Q` 列查詢，並讓 $K$ 與 $V$ 依序流過：

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

逐行對照，這就是線上 softmax 的遞迴式：

| 程式碼 | 意義 |
|---|---|
| `s = tl.dot(q, tl.trans(k))` | 一個分塊的 $S = Q K^\top$，縮放係數已事先乘進 $Q$ |
| `m_new`、`alpha` | 新的最大值，以及重新縮放的係數 $e^{m_{\text{old}} - m_{\text{new}}}$ |
| `m_safe` | 保護只看過被遮罩鍵的列（避免 $-\infty - (-\infty)$） |
| `l`、`acc` | 累計的總和與尚未正規化的輸出，先重新縮放再更新 |
| `acc / l[:, None]`（迴圈之後） | 最後的正規化 |

在底層實作中，大部分程式碼都在把分塊分配給各 lane，並把 $K$ 與 $V$ 經由共享記憶體中轉；這裡則由 `tl.dot` 表達兩次矩陣乘法，實作方式交給編譯器選擇。生產環境的 kernel 會再加上反向傳播、以額外的網格軸處理多個 head 與批次，並在 Hopper 上使用 warp 特化，但核心就是這個迴圈。

在因果情況下，`kv_end` 讓迴圈停在這批查詢所能看到的最後一個鍵，工作量因此減半，與 CUDA kernel 完全相同。

## 7. 迴圈、原子操作與掃描

### 7.1 迴圈

矩陣乘法與注意力 kernel 已經用到迴圈。邊界完全在編譯期已知的迴圈可能被展開；邊界取決於 `K` 或 `n` 的迴圈，在產生的程式碼中仍是迴圈。`tl.range` 提供迴圈屬性，例如軟體管線的階段數：

```python
for k0 in tl.range(0, K, BLOCK_K, num_stages=3):
    ...
```

展開短迴圈可以帶出指令層級的平行度，但展開長迴圈會讓程式碼與變數存活範圍膨脹。依資料大小而定的工作請用執行期迴圈；只有在少數幾種固定大小確實很常見時，才針對它們特化。

### 7.2 Kernel 6：以原子操作實作直方圖

在一般的啟動中，program 之間無法互相同步；要更新共用的全域狀態，原子操作是安全的做法。完整的 [`histogram.py`](examples/14-triton/histogram.py) 能處理不整齊的尾端，並忽略超出 bin 範圍的值：

```python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
in_bounds = offsets < n
values = tl.load(values_ptr + offsets, mask=in_bounds, other=-1)
valid = in_bounds & (values >= 0) & (values < n_bins)
bins = tl.where(valid, values, 0)  # masked pointer arithmetic stays in range
tl.atomic_add(histogram_ptr + bins, 1, mask=valid)
```

整數加法的結果具決定性，但效能不然：大量落在同一個 bin 的值，會在同一個位址上被序列化。生產環境的直方圖通常讓每個 program 建立一份私有直方圖，再由第二個 kernel 合併。Triton 也提供比較並交換、交換、最小值、最大值與位元運算的原子操作；支援的資料型別與記憶體語意依目標硬體而定。

### 7.3 掃描

歸約把一個區塊變成一個值；**掃描**則傳回每一個前綴。`tl.cumsum(x, axis=0)` 是包含式的求和掃描，`tl.associative_scan` 則支援任意具結合律的合併運算。它們都只在區塊內有效：比一個 program 的分塊更長的掃描，需要階層式的演算法：

1. 掃描每個分塊，並存下它的總和；
2. 掃描所有分塊的總和；
3. 把前面所有分塊的總和加到每個分塊上。

這幾個階段之間沒有全網格的 barrier，所以請使用分開的 kernel 啟動，除非明確設計的常駐 kernel 提供了安全的協定。

## 8. 常駐 kernel 與分組排程

分組排程改變的是**順序**：第 5.2 節把相鄰的 program 編號對應到輸出中對快取友善的一塊區域。常駐（persistent）kernel 改變的則是**生命週期**：大約每個運算單元只啟動一個常駐 program，再由每個 program 處理多個邏輯分塊：

```python
pid = tl.program_id(0)
for tile_id in tl.range(pid, n_tiles, tl.num_programs(0)):
    # map tile_id to coordinates, load, compute, store
    ...
```

主機端可以用 `grid=(min(NUM_SMS, n_tiles),)` 啟動，其中 `NUM_SMS` 取自裝置屬性。這樣可以省去一輪輪的啟動排程，也讓一個 program 能保留可重用的狀態。它適合小分塊的工作、Stream-K 類的設計與融合管線，但並非萬靈丹：

- 一個常駐 program 不能占用太多暫存器或共享記憶體，否則能同時駐留的 program 會太少；
- 分塊的工作量不一致時，靜態的輪流分配可能造成負載不平衡；
- 以原子操作實作的工作佇列能平衡不規則的工作，但會增加競爭；
- 自旋等待的協定若在等待一個無法被排程的 program，就會死結；絕不要假設所有邏輯 program 會同時駐留；
- 針對每個後端與架構，分別調校常駐網格的大小。

先從一般的分組排程開始。只有當效能分析顯示啟動開銷、尾端那一輪或快取駐留的成本正好是常駐設計能解決的問題時，才改用常駐 kernel。

## 9. 編譯階段、資料配置與 warp 特化

### 9.1 從 Python 到機器碼

![編譯器把加上裝飾器的 Python 函式依序轉換為 Triton IR（區塊運算）、TritonGPU IR（資料配置、共享記憶體、管線化）與 LLVM IR，最後產生 PTX 或 AMDGCN](figures/ch14-compiler.svg)

當 constexpr 值、參數資料型別與**特化條件**出現新的組合時（例如 Triton 會檢查指標與整數參數是否能被 16 整除，以證明向量化載入的對齊），第一次啟動會觸發編譯；之後的啟動則命中記憶體與磁碟上的快取。啟動會傳回一個 handle，其 `asm` 字典保存了每個階段的結果：

```python
handle = add_kernel[grid](x, y, out, n, BLOCK=1024)
print(handle.asm["ttgir"])    # layouts chosen by the compiler
print(handle.asm["ptx"])      # look for ld.global.v4.f32 (vectorised loads)
```

效能問題的答案都在 TritonGPU IR 裡：它顯示每個張量的資料配置（`#blocked`、`#mma`、`#shared`）、共享記憶體配置在哪裡，以及產生了幾個管線階段。

### 9.2 資料配置與 warp 特化

**資料配置**（layout）描述區塊值中的哪些元素由哪些 lane 與 warp 持有。一般的逐元素運算使用 blocked 配置；dot 運算元與 MMA 配置供應矩陣單元；shared 配置描述中轉的資料。配置之間的轉換可能需要 shuffle 或一次共享記憶體往返，所以當一個看似無害的轉置或重塑形狀造成效能倒退時，請檢查 TritonGPU IR。

`num_warps` 控制有多少個 warp 合作處理一個 program；它並不是手動把張量的某一片指派給某個 warp，這種對應由編譯器依資料配置決定。**Warp 特化**則讓不同的 warp 群組擔任不同角色，例如生產者 warp 負責搬移分塊，消費者 warp 負責執行矩陣指令。在支援的 Triton 目標上，選定的管線化迴圈可以透過 `tl.range` 的 `warp_specialize` 選項要求這種做法。這是依目標而定的進階最佳化：可用性與合法的資料配置會隨 Triton 版本改變，而 NVIDIA 專屬的 warp group 機制也無法直接套用到 AMD 的 wavefront。請保留一個不使用特化的設定，只選用實測勝出的版本。

## 10. CUDA 與 ROCm 的可攜性

同樣的指標算術、遮罩、歸約與大部分 `tl.dot` 程式碼，都能同時為 CUDA 與 ROCm 後端編譯。效能模型可以通用，最佳的常數通常不行。

| 項目 | 可攜性原則 |
|---|---|
| Warp / wave 寬度 | 不要在 lane 層級假設寬度為 32；以區塊與歸約來表達工作 |
| 矩陣指令 | 使用受支援的資料型別，並針對各後端測試合適的 `BLOCK_K` 與分塊形狀 |
| 共享記憶體 / LDS | 容量、bank 行為與佔用率各不相同；重新調校 `num_stages` 與 `num_warps` |
| 描述子與非同步複製 | 把 TMA 與其他目標專屬的路徑視為可選的快速路徑 |
| 原子操作 | 確認資料型別與運算是否支援，特別是低精度與 64 位元的情況 |
| 數學函式 | 近似的 `exp`、除法與 TF32 行為，可能需要依後端設定容許誤差 |
| 效能分析工具 | CUDA 上使用 Nsight Systems/Compute，ROCm 上使用 `rocprof`/Omniperf |

讓演算法程式碼保持共用，並在 Python 包裝函式中為各後端挑選一小組設定。請在兩個後端上以實際編譯的模式測試：直譯器只驗證索引語意，並不驗證目標程式碼的產生、矩陣指令的選擇或非同步管線。

## 11. 測試、直譯器、除錯與效能分析

[`test_kernels.py`](examples/14-triton/test_kernels.py) 把六個 kernel 都拿來與 PyTorch 比對。測試案例包括：空輸入（包裝函式不啟動 kernel 就返回）、大小為 1 的形狀、不整齊的尾端、剛好超過分塊邊界的維度、因果遮罩、超出範圍的直方圖 bin，以及大量的原子操作碰撞。生產環境的測試還應涵蓋所有支援的資料型別、API 承諾支援的非連續配置、極端值、NaN/Inf 的處理方式、多組亂數種子，以及每一個部署的後端。

在**匯入 Triton 之前**設定 `TRITON_INTERPRET=1`，就能在 CPU 上以 NumPy 執行各個 program 實例。這很適合檢查指標算術與遮罩，也能使用一般的 `print` 與 `pdb`。但它不會模擬平行的競爭條件、GPU 浮點運算的細節、資料配置、佔用率、目標指令或效能。通過直譯器測試是必要的證據，但不等於在 GPU 上驗證過。

### 11.1 除錯

| 工具 | 用途 |
|---|---|
| `TRITON_INTERPRET=1` | 以 NumPy 在 CPU 上執行；kernel 內可以使用 `print()` 與 `pdb` |
| `tl.device_print("x", x)` | 從編譯後的 kernel 印出資訊（每個 program 都會印：請用遮罩或小網格限制） |
| `tl.static_print`、`tl.static_assert` | 在編譯期印出或檢查 constexpr 值 |
| `tl.device_assert(cond, "msg")` | 執行期的斷言（以 `TRITON_DEBUG=1` 啟用） |

常見的錯誤都是靜態的：`arange` 的邊界不是 2 的冪次、形狀無法廣播、編譯器需要常數的地方傳入了非 constexpr 的值，以及 `tl.dot` 的運算元在某個維度小於 16。

若邊緣的值不正確，把網格縮小到一個 program，印出位移、遮罩與載入的值。若程式當掉，先在直譯器下以很小的奇數形狀執行，再到 GPU 上使用後端的記憶體檢查工具。若是數值誤差，以 FP32 比較中間狀態，並明確判斷差異來自運算順序、近似數學函式、TF32，還是錯誤的遮罩。

### 11.2 基準測試與效能分析

`triton.testing.do_bench(fn)` 會在暖機、重複執行並在每次之間清空 L2 的情況下計時一個函式，傳回毫秒數；`triton.testing.perf_report` 則掃過多種大小並畫出結果。對 Nsight Compute 而言，Triton kernel 就是一般的 GPU kernel（`ncu -k regex:matmul_kernel python3 test_kernels.py --bench`），而且預設就會附上類似 `-lineinfo` 的資訊，讓原始碼對應指向 Python 程式行。

請對已經暖機的 kernel 做基準測試，避免把編譯與自動調校的時間算進去。回報時註明形狀、資料型別、步幅、後端、GPU、Triton 版本與選中的設定。除了推算出的頻寬或 FLOP/s，也要比較延遲；在改變分塊之前先做效能分析：佔用率太低、暫存器溢出、記憶體停滯、資料配置轉換與啟動間隙，需要的解法各不相同。

## 12. 生產環境決策清單

| 情境 | 選擇 |
|---|---|
| 形狀標準的 GEMM、卷積或注意力 | 函式庫（cuBLAS、cuDNN、hipBLASLt、FlashAttention） |
| 沒有函式庫提供的融合運算（GEMM + 自訂 epilogue、新的注意力變體、融合的正規化） | Triton |
| 必須同時在 NVIDIA 與 AMD 上執行的研究程式碼 | Triton |
| 最後那一點架構專屬的效能，需要自訂的同步或資料搬移 | CUDA、HIP、CUTLASS 或 CuTe |
| 以不規則的逐執行緒控制流程為主的演算法（排序網路、圖走訪） | CUDA |

在發布 Triton kernel 之前，請先回答以下所有問題：

- **價值：** 在具代表性的生產形狀上，融合、特化或新演算法是否勝過最合適的函式庫？
- **介面約定：** 包裝函式是否檢查了形狀、資料型別、步幅、對齊、裝置、記憶體重疊與空輸入的行為？
- **正確性：** 是否以高精度的參考實作，測試了不整齊的尾端、被遮罩的列、極端值、NaN、原子操作與數值容許誤差？
- **涵蓋範圍：** 對不支援的形狀、資料型別、後端與自動調校失敗的設定，是否有安全的退路？
- **調校：** 自動調校的 key 是否夠具體，不會沿用不適合的設定；同時又有上限，讓首次使用時的調校時間與快取成長可以接受？
- **資源：** 編譯器輸出與效能分析是否顯示暫存器、共享記憶體與佔用率都在可接受範圍，且沒有意外的溢出或資料配置轉換？
- **維運：** 編譯與自動調校的暖機、快取行為、Triton 與驅動程式的版本、可觀測性與回滾都處理好了嗎？
- **可攜性：** 每一個宣稱支援的 CUDA/ROCm 架構，是否都以編譯模式跑過正確性與效能測試？

在量測結果說明為何需要自訂 kernel 之前，優先使用函式庫；在能達成目標的前提下，選擇最簡單的 Triton 設計，並保留框架或函式庫的退路。

## 重點整理

1. Triton kernel 描述的是一個 program 如何處理形狀在編譯期決定的區塊；編譯器負責把這些區塊對應到執行緒與目標指令。
2. 指標區塊加上遮罩，取代了執行緒索引與邊界檢查；`other` 為被遮罩的 lane 提供單位元。
3. 合併存取、向量化、共享記憶體中轉、管線化、swizzle 與 tensor core 指令都由編譯器產生；分塊大小、分塊順序與融合仍由你決定。
4. `triton.autotune` 針對每種問題大小，搜尋分塊形狀、`num_warps` 與 `num_stages`。
5. 迴圈用來表達分塊演算法；原子操作透過全域記憶體溝通；掃描只在區塊內有效，跨區塊需要階層式做法。
6. 常駐 kernel 與 warp 特化是需要實測、且依目標而定的最佳化，不是起點。
7. 直譯器在 CPU 上檢查索引語意；正確性與效能要靠編譯後的 GPU 測試、IR 檢視與後端的效能分析工具來確立。

## 練習

### 基礎

1. **遮罩與形狀。** 在 `add_kernel` 中，如果 `BLOCK` 是 1000 會發生什麼事？如果載入時省略了 `mask` 呢？

    <details markdown="1"><summary>答案</summary>

    `tl.arange(0, 1000)` 無法編譯：區塊大小必須是 2 的冪次。沒有遮罩時，最後一個 program 會越界讀取 `x` 與 `y`（讀到未定義的值；`compute-sanitizer` 會回報越界讀取）；儲存若仍有自己的遮罩保護，結果可能看起來正確，但 kernel 其實是錯的。

    </details>

2. **遮罩下的歸約。** 在 `layer_norm_kernel` 中，拿掉 `tl.where(mask, x - mean, 0.0)`，直接使用 `x - mean`。為什麼即使輸入的載入使用了 `other=0.0`，寬度為奇數時仍然會出錯？

    <details markdown="1"><summary>答案</summary>

    被填補的 lane 載入的是零，但置中之後變成 `-mean`，而不是零；因此每個被填補的 lane 都會把 `mean²` 加進變異數。單位元必須在每一次歸約時各自套用：載入時為平均值準備的單位元，在做完減法之後並不會自動仍是單位元。

    </details>

### 中等

3. **線上歸約。** 為長到放不進暫存器的列撰寫 softmax kernel：以 `BLOCK` 欄為單位在列上迴圈，用第 6 節的遞迴式保存累計的最大值與總和，再走第二次迴圈寫出結果。

    <details markdown="1"><summary>提示</summary>

    在第一個迴圈中保存形狀為 `(BLOCK,)` 的逐 lane 向量 `m` 與 `z`（`m_new = tl.maximum(m, x)`、`z = z * tl.exp(m - m_new) + tl.exp(x - m_new)`，並加上 −∞ 的保護），最後以 `M = tl.max(m, 0)`、`Z = tl.sum(z * tl.exp(m - M), 0)` 合併；第二個迴圈再寫出 `tl.exp(x - M) / Z`。不論列有多長，都只需要兩次讀取、一次寫入。

    </details>

4. **資源計算。** 當 $M = N = K = 4096$、使用 FP16，且 `BLOCK_M = BLOCK_N = 128`、`BLOCK_K = 32`、`num_stages = 3` 時，一個 program 需要多少共享記憶體？A100 的一個 SM（164 KB）能容納幾個 program？

    <details markdown="1"><summary>答案</summary>

    $3 \cdot (128 + 128) \cdot 32 \cdot 2 = 49\,152$ 位元組 = 48 KB，因此就共享記憶體而言可以容納三個 program。但若 `num_warps = 8`（256 個執行緒），128 × 128 的 FP32 累加器（每個執行緒 64 個值）再加上運算元，通常會讓暫存器檔案把數量限制在一到兩個。

    </details>

5. **Epilogue 融合。** 以 `RELU: tl.constexpr` 旗標控制，為 `matmul_kernel` 加上融合的 ReLU epilogue，並與 `torch.relu(a @ b)` 比對。為什麼 constexpr 比執行期旗標好？

    <details markdown="1"><summary>答案</summary>

    在寫出之前加上 `if RELU: acc = tl.maximum(acc, 0.0)`。作為 constexpr，每個值各自編譯出一個 kernel，分支因此消失；執行期旗標則會在每個 program 中留下一個（一致且便宜的）分支，也讓編譯器無法針對 epilogue 特化。

    </details>

### 進階

6. **私有化的原子操作。** 把直方圖改成兩個 kernel：第一個為每個 program 寫出一份私有直方圖，第二個再把這些直方圖歸約起來。什麼情況下它會比直接使用原子操作更快？

    <details markdown="1"><summary>提示與答案</summary>

    在形狀為 `(num_programs, n_bins)` 的暫存陣列中，讓每個 program 擁有一列；先在本地累加（必要時可在自己那一列中使用原子操作），再沿所有 program 的列加總每個 bin。這會增加暫存資料的流量與一次 kernel 啟動，所以只有當集中在單一全域直方圖上的碰撞，所造成的序列化成本超過這些開銷時，它才會勝出。請對偏斜與均勻兩種分布都做基準測試；bin 的數量與資料型別都會明顯影響交叉點。

    </details>

7. **遮罩下的注意力。** `m_safe` 只有在某一列至今只看過被遮罩的鍵時才會起作用。請證明目前的 `flash_attention` 不會發生這種情況，並舉出一種會發生的注意力變體。

    <details markdown="1"><summary>答案</summary>

    第一個分塊一定從鍵 0 開始，而鍵 0 對每一列查詢都有效：它確實存在（$n \ge 1$），在因果遮罩下也不是任何一列的「未來」（超出 $n$ 的填補列同樣看得到它）。因此處理完第一個分塊之後，每個 `m` 都是有限值。在滑動視窗注意力中就會發生這種情況（第 $r$ 列只看得到 $r - w, \dots, r$，所以後面的列在前幾個分塊中全部被遮罩），使用鍵填補遮罩時，或以不同順序走訪分塊時（例如從對角線開始）也會。這時若沒有保護，`s - m_new` 會變成 $-\infty - (-\infty) = \text{NaN}$。

    </details>

8. **全域掃描。** 為任意長度的向量設計掃描。為什麼在一般的 kernel 中，不能用自旋等待的全網格 barrier 取代三次 kernel 啟動？

    <details markdown="1"><summary>答案</summary>

    先啟動區塊內的掃描並存下各分塊的總和；再遞迴地掃描這些總和；最後啟動一個 kernel，把前面所有分塊的總和加到每個分塊上。自旋等待的 barrier 可能死結：已駐留的 program 也許正在等待某些邏輯 program，而這些 program 必須等已駐留的 program 結束後才能被排程。只有當啟動規模被刻意限制在能同時駐留的數量之內，且記憶體協定正確時，常駐設計才能實作自訂的 barrier；使用分開的 kernel 啟動才是安全的預設做法。

    </details>

## 實作練習

LeetGPU 與 Tensara 都接受 Triton 解答；以下題目很適合拿來初次移植上述 kernel：

- [LeetGPU – 向量加法](../leetgpu/001-vector-add/)
- [LeetGPU – Softmax](../leetgpu/005-softmax/)、[Tensara – Softmax](../tensara/softmax/)
- [LeetGPU – 矩陣乘法](../leetgpu/002-matrix-multiplication/)、
  [Tensara – 矩陣乘法](../tensara/matrix-multiplication/)
- [LeetGPU – Softmax 注意力](../leetgpu/006-softmax-attention/)、
  [LeetGPU – 因果注意力](../leetgpu/053-casual-attention/)
- [Tensara – Layer Norm](../tensara/layer-norm/)（與第 4 節類似的融合歸約）
