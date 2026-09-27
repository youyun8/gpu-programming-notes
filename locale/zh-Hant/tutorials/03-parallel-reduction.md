# 03 – 平行歸約

> **第一部分 · CUDA 基礎** · 先備知識：[01](01-execution-model.md)、[02](02-memory-hierarchy.md) ·
> 下一章：[09 – 分析與效能解析](09-profiling.md)

目標：$s = \sum_i x_i$。相同模式可計算最大值、arg-max、點積、範數、softmax 分母、平均值和變異數，因此是實作題中最常重複使用的基礎元件。歸約受記憶體限制（每個輸入只讀一次，並做一次加法）；真正的難處是如何合併數百萬個值，又不把頻寬浪費在同步上。

**你將學到**

- 從工作量與深度的角度理解平行歸約，以及為何每個執行緒應先循序歸約；
- 在共享記憶體中進行區塊歸約，以及為何「連續定址」是正確的樹；
- Warp shuffle 的作用，以及如何不使用共享記憶體或 barrier 來歸約一個 warp；
- 合併多個區塊結果的四種方法，包括單 pass 的「最後區塊」模式；
- 同時歸約多列（每列使用一個 warp 或區塊）；
- 各方法的浮點準確度，以及 Kahan 加總；
- 使用任何結合運算子的歸約（max、arg-max、softmax 正規化因子、平均值和變異數）。

## 1. 工作量、深度與 Brent 界限

循序加總會執行 $n - 1$ 次加法，形成長度為 $n - 1$ 的鏈。平衡樹執行同樣多次加法，但層數少得多：

$$
W = n - 1, \qquad D = \lceil \log_2 n \rceil, \qquad
T_p \le \frac{W}{p} + D
$$

| 符號 | 意義 |
|---|---|
| $n$ | 輸入數量 |
| $W$ | 工作量：加法總次數 |
| $D$ | 深度：最長相依加法鏈的長度 |
| $p$ | 處理器數量（同時工作的執行緒） |
| $T_p$ | 使用 $p$ 個處理器時的時間步數（Brent 定理） |

此樹具有**工作效率**（$W$ 與循序加總相同），且深度為對數。在 GPU 上，$p$ 有數萬，而 $n$ 有數百萬，因此 $W/p$ 項占主導。實務做法也由此而來：每個執行緒先**循序**加總許多元素（成本低、不需同步），再只將少量逐執行緒結果放入樹中。

## 2. 共享記憶體中的區塊層級樹狀歸約

### 2.1 Kernel

```cpp
constexpr int kBlockSize = 256;

__global__ void reduceSum(const float* input, float* output, int n) {
    __shared__ float cache[kBlockSize];
    const int tid = threadIdx.x;

    // Grid-stride accumulation in a register first: fewer blocks, fewer atomics.
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        local_sum += input[i];
    }
    cache[tid] = local_sum;
    __syncthreads();

    // Sequential addressing: active threads stay contiguous, so full warps
    // either all work or all idle (no divergence) and there are no bank conflicts.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) cache[tid] += cache[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, cache[0]);
}
```

啟動前必須將 `*output` 歸零（`cudaMemset`）。這棵樹有 $\log_2 256 = 8$ 層，每層都以 barrier 結束。

### 2.2 逐步理解這棵樹

![連續定址：每一層都由有效執行緒的前半部加上後半部](figures/ch03-tree.svg)

在第 $j$ 層（跨步 $B/2^{j+1}$），執行緒 $0 \dots \text{stride}-1$ 各自加上距離一個 stride 的元素。最後一層完成後，`cache[0]` 會保存區塊總和。

### 2.3 為何使用這棵樹

兩項特性決定樹在 GPU 上是否快速：

1. **Warp 一致的活動。** 使用 `if (tid < stride)` 時，有效執行緒會形成前綴：stride 為 64 時，warp 0–1 工作，warp 2–7 則整層略過。若樹寫成 `if (tid % (2 * stride) == 0)`（「交錯定址」），每個 warp 都只會有一部分 lane 有效，多數 lane 被遮罩。
2. **對 bank 友善的位址。** lane $\ell$ 讀取 `cache[ℓ + stride]`：這些是位於 32 個 bank 的連續 word。交錯樹會以 $2 \cdot \text{stride}$ 個 word 為跨步讀取，因而發生衝突。

### 2.4 為何每層都需要 Barrier

第 $j+1$ 層會讀取*其他*執行緒在第 $j$ 層寫入的值，因此每層都需要 `__syncthreads()`。Barrier 必須位於 `if` 外：區塊中的所有執行緒都要抵達，包括該層未工作的執行緒。

## 3. Warp Shuffle

### 3.1 Shuffle 是什麼

Shuffle 可用一條指令讀取同一 warp 中另一個 lane 的暫存器，不需要共享記憶體：

| Intrinsic | lane $\ell$ 接收哪個 lane 的值 |
|---|---|
| `__shfl_sync(mask, v, src)` | `src`（所有 lane 使用相同 `src` 時為廣播） |
| `__shfl_up_sync(mask, v, d)` | $\ell - d$（若 $\ell < d$ 則不變） |
| `__shfl_down_sync(mask, v, d)` | $\ell + d$（若 $\ell + d \ge$ width 則不變） |
| `__shfl_xor_sync(mask, v, m)` | $\ell \oplus m$（butterfly） |

`mask` 指定參與的 lane（它們都必須執行此呼叫）；`0xffffffff` 代表整個 warp。可選的最後一個引數 `width`（不超過 32 的二次方數）可將 warp 分成獨立區段。

### 3.2 歸約一個 Warp

Warp 內不需要共享記憶體或 barrier；`__shfl_down_sync` 可讀取另一個 lane 的暫存器：

$$
v^{(s+1)}_\ell = v^{(s)}_\ell + v^{(s)}_{\ell + \delta_s}, \qquad \delta_s = 16, 8, 4, 2, 1
$$

| 符號 | 意義 |
|---|---|
| $\ell$ | Lane 索引 |
| $v^{(s)}_\ell$ | 步驟 $s$ 後 lane $\ell$ 的值 |
| $\delta_s$ | 步驟 $s$ 的 shuffle 位移；5 步後 lane 0 會保存 warp 總和 |

![使用位移 8、4、2、1 的 __shfl_down_sync：經過 log2(width) 個步驟後，lane 0 會保存總和](figures/ch03-shuffle.svg)

若改用 `__shfl_xor_sync`（butterfly），*每個* lane 最後都會得到完整總和；當所有 lane 都需要結果時，可省下一次廣播（softmax、正規化）。

### 3.3 從 Warp 擴展到區塊

```cpp
__device__ float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;   // valid in lane 0
}

__device__ float blockReduceSum(float value) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    value = warpReduceSum(value);
    if (lane == 0) warp_sums[warp_id] = value;
    __syncthreads();

    const int num_warps = (blockDim.x + 31) / 32;
    value = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) value = warpReduceSum(value);
    return value;   // valid in thread 0
}
```

只需兩輪 shuffle 和**一次** barrier，而不是八次：

1. 每個 warp 將其 32 個值歸約到 lane 0；
2. 各 warp 的 lane 0 將一個部分結果寫入 `warp_sums`；
3. barrier 後，warp 0 歸約這些部分結果（最多 32 個）。

若函式會連續呼叫兩次（例如先算總和，再算平方和），請在函式結尾加入 `__syncthreads()`，避免第二次呼叫覆寫 `warp_sums` 時，warp 0 仍在讀取它。

## 4. 經典演進過程

Mark Harris 的 *Optimizing Parallel Reduction in CUDA* 逐步介紹七個版本。這些經驗至今仍適用：

| # | 變更 | 修正的問題 |
|---|---|---|
| 1 | 交錯定址，`if (tid % (2*s) == 0)` |（基準）嚴重分歧：從第一步起，每個 warp 都有一半 lane 閒置 |
| 2 | 跨步索引 `index = 2*s*tid` | 修正分歧，但引入共享記憶體 bank 衝突 |
| 3 | 連續定址（如上） | Bank 衝突 |
| 4 | 在全域載入時先做第一次加法 | 第一層有一半執行緒閒置 |
| 5 | 展開最後一個 warp | 只剩一個 warp 時的 barrier 和迴圈開銷（現今可用 shuffle） |
| 6 | 使用樣板完全展開 | 迴圈開銷 |
| 7 | 每執行緒處理多個元素（網格跨步） | Kernel 變成受頻寬限制：這就是目標 |

歸約只讀取每個輸入一次，因此界限是

$$
T_{\min} = \frac{4n}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $4n$ | 讀取位元組數（float 輸入） |
| $\beta$ | DRAM 頻寬 |

版本 7 加上 `float4` 載入可達到界限的 85–95 %。當每個執行緒先加總數百個元素後，樹本身的成本幾乎可以忽略。

## 5. 跨區塊完成歸約

### 5.1 選項

區塊無法彼此等待，因此逐區塊結果需要第二個步驟：

![完整歸約：逐執行緒加總、各區塊內的歸約，以及跨區塊的最後步驟](figures/ch03-two-level.svg)

| 方法 | 作法 | 具決定性？ |
|---|---|---|
| Atomic | 每個區塊的執行緒 0 執行 `atomicAdd(out, block_sum)` | 否：每次執行的加法順序都可能不同 |
| 兩個 kernel | Kernel 1 寫出 $G$ 個部分結果；kernel 2（單一區塊）歸約它們 | 是 |
| 最後區塊 | 每個區塊寫入部分結果、呼叫 `__threadfence()`、遞增計數器；看見計數達到 $G$ 的區塊負責歸約部分結果 | 是 |
| Cooperative groups | 在一次 cooperative launch 中呼叫 `grid.sync()` | 是 |

本站大多數題目頁面使用雙 kernel 版本（例如 [Tensara – Frobenius Norm](../tensara/frobenius-norm/) 和 [Tensara – MSE Loss](../tensara/mse-loss/)）：當 $G \le 1024$ 個區塊時，第二個 kernel 非常小。

### 5.2 最後區塊模式

單一 kernel 也能以具決定性的方式完成工作：由*最後*完成的區塊歸約所有人的部分結果。

```cpp
__device__ unsigned int g_blocks_done = 0;   // must be 0 at launch; the last block resets it

__global__ void reduceSinglePass(const float* in, float* partials, float* out, int n) {
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) v += in[i];
    v = blockReduceSum(v);

    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = v;
        __threadfence();                                         // 1. publish the partial ...
        const unsigned int done = atomicAdd(&g_blocks_done, 1);  // 2. ... before being counted
        is_last = done == gridDim.x - 1;
    }
    __syncthreads();
    if (is_last) {                                               // block-uniform branch
        float s = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) s += partials[i];
        s = blockReduceSum(s);
        if (threadIdx.x == 0) {
            *out = s;
            g_blocks_done = 0;                                   // ready for the next launch
        }
    }
}
```

#### 為何正確

- fence 保證「寫入我的部分結果」先於「遞增計數器」，所以觀察到計數為 $G - 1$ 的區塊必定能看見全部 $G$ 個部分結果。
- `is_last` 是共享變數，因此整個區塊的分支一致，`blockReduceSum` 內的 barrier 會由所有執行緒抵達。
- 不論區塊完成順序為何，部分結果都會依索引順序相加，因此結果可逐位元重現。

最後區塊會呼叫 `blockReduceSum` 兩次；`is_last` 後方的 barrier 會依第 3.3 節的要求，分隔 `warp_sums` 的兩次使用。

## 6. 歸約多列

Softmax、layer normalization、逐列範數和逐列 arg-max 都會獨立歸約矩陣的每一列。問題變成每列應分配多少執行緒：

| 列長度 | 對應方式 | 原因 |
|---|---|---|
| ≤ 約 1024 | 每列一個 warp | Warp 歸約不需要共享記憶體和 barrier；一個區塊可執行多列 |
| 約 1 K – 32 K | 每列一個區塊 | 每列有足夠執行緒維持頻寬；每列執行一次區塊歸約 |
| 更大、列數少 | 每列數個區塊 | 否則區塊數不足以填滿 GPU；使用 atomic 或第二個 pass 合併 |

```cpp
// One warp per row: in is rows x cols, row-major; out[r] = sum of row r.
__global__ void rowSum(const float* in, float* out, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) return;                       // the whole warp exits together
    const float* p = in + static_cast<size_t>(row) * cols;
    float s = 0.0f;
    for (int c = lane; c < cols; c += 32) s += p[c];   // coalesced: lanes read consecutive columns
    s = warpReduceSum(s);
    if (lane == 0) out[row] = s;
}
```

提早 `return` 是安全的，因為 32 個 lane 的 `row` 相同，所以 `warpReduceSum` 中使用完整 mask 的 shuffle 不是由整個 warp 執行，就是完全不執行。

沿著**欄**歸約則不同：應讓連續執行緒處理連續欄（合併存取），再讓各執行緒沿自己的欄向下走（[Tensara – Max Dim](../tensara/max-dim/)）。

## 7. 準確度

### 7.1 誤差界限

浮點加法不具結合律，因此 GPU 結果會與循序 CPU 加總不同。對具有單位捨入誤差 $u$ 的格式，$n$ 項相加的經典誤差界限（Higham）為：

$$
\lvert \hat{s} - s \rvert \le \gamma_{n-1} \sum_i \lvert x_i \rvert \ \ (\text{sequential}), \qquad
\lvert \hat{s} - s \rvert \le \gamma_{\lceil \log_2 n \rceil} \sum_i \lvert x_i \rvert \ \ (\text{pairwise tree}), \qquad
\gamma_k = \frac{k u}{1 - k u}
$$

| 符號 | 意義 |
|---|---|
| $s, \hat{s}$ | 精確總和與計算所得總和 |
| $u$ | 單位捨入誤差：fp32 為 $2^{-24}$，fp64 為 $2^{-53}$ |
| $\gamma_k$ | $k$ 次相依加法後的誤差成長係數 |

GPU 方法（逐執行緒循序鏈，再進入樹）介於兩者之間：鏈長約為 $n/p$。實務影響如下：

- 對 $n = 10^8$ 個 fp32 值，單一循序累加器可能損失 4–5 位有效數字；樹幾乎不會損失；
- 使用 `double` 累加逐執行緒和逐區塊部分結果，可讓本站會遇到的任何大小都近乎精確。

### 7.2 Kahan 加總

補償加總會將每次加法的捨入誤差帶到下一次：

```cpp
float sum = 0.0f, c = 0.0f;            // c: running compensation (the lost low bits)
for (int i = start; i < n; i += stride) {
    const float y = input[i] - c;      // add back what was lost last time
    const float t = sum + y;           // big + small: low bits of y are lost ...
    c = (t - sum) - y;                 // ... and recovered here (algebraically zero)
    sum = t;
}
```

誤差會變得與鏈長無關（約為 $2u\sum|x_i|$）。每個元素需 4 次浮點運算，但對受記憶體限制的 kernel 而言等同免費。請注意 `--use_fast_math` 可能會重新結合 `(t - sum) - y`，使其變成零。

## 8. 一般化歸約：么半群

### 8.1 模式

任何具有單位元素 $e$ 的**結合**運算子 $\oplus$，都能用完全相同的程式碼歸約：

$$
x_0 \oplus x_1 \oplus \cdots \oplus x_{n-1}, \qquad (a \oplus b) \oplus c = a \oplus (b \oplus c), \qquad e \oplus a = a
$$

| 符號 | 意義 |
|---|---|
| $\oplus$ | 合併運算子 |
| $e$ | 單位元素（padding lane 提供的值） |

| 歸約 | 狀態 | 單位元素 $e$ | 合併 $(a \oplus b)$ |
|---|---|---|---|
| 總和 | $s$ | 0 | $s_a + s_b$ |
| 最大值 | $m$ | $-\infty$ | $\max(m_a, m_b)$ |
| Arg-max（第一個索引） | $(v, j)$ | $(-\infty, \infty)$ | 取較大 $v$；相等時取較小 $j$ |
| Softmax 正規化因子 | $(m, z)$ | $(-\infty, 0)$ | $M = \max(m_a, m_b)$，$z = z_ae^{m_a - M} + z_be^{m_b - M}$ |
| 平均值與變異數（Welford / Chan） | $(n, \mu, M_2)$ | $(0, 0, 0)$ | 見下文 |

正確性不要求交換律，只有任意順序合併的自由度才需要；以上運算子也全都具交換律。

### 8.2 單 Pass 計算平均值與變異數

Chan 用來合併平均值與平方偏差總和的平行公式：

$$
n = n_a + n_b, \qquad \delta = \mu_b - \mu_a, \qquad
\mu = \mu_a + \delta\,\frac{n_b}{n}, \qquad
M_2 = M_{2,a} + M_{2,b} + \delta^2\,\frac{n_a n_b}{n}
$$

| 符號 | 意義 |
|---|---|
| $n_a, n_b$ | 兩個部分結果的元素數 |
| $\mu_a, \mu_b$ | 各自的平均值 |
| $M_{2,a}, M_{2,b}$ | 各自相對於自身平均值的平方偏差總和 |
| $\delta$ | 平均值差 |
| $\mu, M_2$ | 合併後的平均值與平方偏差總和；變異數為 $M_2 / n$ |

#### 以程式碼表示動差狀態

程式碼中，狀態是一個小型 struct，合併則是一個函式；warp 歸約會 shuffle 每個欄位：

```cpp
struct Moments { float n, mean, m2; };

__device__ Moments combine(Moments a, Moments b) {
    const float n = a.n + b.n;
    if (n == 0.0f) return a;                        // both empty: identity
    const float delta = b.mean - a.mean;
    return {n, a.mean + delta * (b.n / n), a.m2 + b.m2 + delta * delta * (a.n * b.n / n)};
}

__device__ Moments warpReduceMoments(Moments v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        Moments o{__shfl_down_sync(0xffffffff, v.n, offset), __shfl_down_sync(0xffffffff, v.mean, offset),
                  __shfl_down_sync(0xffffffff, v.m2, offset)};
        v = combine(v, o);
    }
    return v;   // valid in lane 0
}
```

這能在單一 pass 中算出平均值與變異數，且不會發生 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ 的消去誤差。Tensara 頁面正是實作這種「累加器 struct」概念（[Argmax](../tensara/argmax/)、[Sum Dim](../tensara/sum-dim/)）；softmax 么半群則是 FlashAttention 的關鍵（[Tensara – Scaled Dot-Product Attention](../tensara/scaled-dot-attention/)）。

## 重點整理

1. 每個執行緒先循序加總（網格跨步），再以樹歸約每區塊的少量值：$W/p$ 項占主導。
2. 在共享記憶體中使用連續定址；更好的方法是使用 warp shuffle 加上一次 barrier。
3. 依決定性需求選擇跨區塊步驟：atomic（快速但無法重現）、兩個 kernel 或最後區塊模式（可重現）。
4. 依列長度將列對應到 warp 或區塊。
5. 樹狀歸約比單一累加器更準確；`double` 部分結果或 Kahan 加總可去除剩餘誤差。
6. 任何具結合律的內容（max、arg-max、softmax 統計值、Welford moment）都能使用相同程式碼，只需替換合併運算。

## 練習

1. 為何 `blockReduceSum` 在區塊有 96 個執行緒時仍可運作？若 `blockDim.x` 不是 32 的倍數，而 warp shuffle 使用 `0xffffffff`，會發生什麼問題？

    <details markdown="1"><summary>答案</summary>

    96 個執行緒 = 3 個完整 warp；warp 0 讀取 `warp_sums[0..2]`，lane ≥ 3 則使用零。若最後一個 warp 不完整，完整 mask 會指定不存在的 lane：行為未定義。請使用由 `__activemask()` 推導的 mask，或將區塊補到 32 的倍數，並讓補上的 lane 使用單位元素。

    </details>

2. 將 `warpReduceSum` 中的 shuffle 改成 `__shfl_xor_sync`，並說明為何每個 lane 最後都會得到總和。

    <details markdown="1"><summary>答案</summary>

    步驟 $\delta$ 後，lane $\ell$ 和 $\ell \oplus \delta$ 會保存相同值（兩組的總和）。依序執行 $\delta = 16, 8, 4, 2, 1$ 後，每個 lane 都已合併全部 32 個值。

    </details>

3. 將 softmax 正規化因子的合併 $(m, z)$ 寫成 `__device__` 函式，並手動以小型陣列的兩半檢查。

4. 對 A100 上的 $n = 2^{26}$ 個 float，$T_{\min}$ 是多少？若 kernel 耗時 0.21 ms，相當於峰值頻寬的多少比例？

    <details markdown="1"><summary>答案</summary>

    $4n = 268$ MB，$T_{\min} = 268\text{ MB} / 1.55\text{ TB/s} \approx
    0.173$ ms；0.21 ms 是 82 %。

    </details>

## 實作練習

- [LeetGPU – 歸約](../leetgpu/004-reduction/)
- [LeetGPU – 點積](../leetgpu/017-dot-product/)
- [LeetGPU – Softmax](../leetgpu/005-softmax/)
- [Tensara – Argmax](../tensara/argmax/)、[Tensara – Layer Norm](../tensara/layer-norm/)
