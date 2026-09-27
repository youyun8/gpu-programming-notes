# 13 – Softmax、LayerNorm 與 FlashAttention

> **第二部分 · 平行模式** · 先備知識：[03](03-parallel-reduction.md)、[10](10-warp-primitives.md)；
> 第 5 節也會使用 [矩陣乘法 1](04-tiled-matmul.md) 的分塊方式 ·
> 程式：[`examples/13-softmax-attention.cu`](examples/13-softmax-attention.cu) ·
> 下一章：[矩陣乘法 1 – 基礎](04-tiled-matmul.md)（第三部分）

Transformer 在 GEMM 以外的大部分時間，都花在三種操作：softmax、
正規化（LayerNorm、RMSNorm）和注意力。注意力會結合兩次 GEMM，並在
中間執行 softmax。這三者都是逐列歸約後接逐元素走訪，也都基於相同原因
而變快：使用**線上**形式，在單次走訪中攜帶一個小型狀態，並以結合性
運算子合併狀態。

**你將學到**

- 數值穩定的 softmax，以及它需要走訪記憶體幾次；
- 線上 softmax：重新縮放技巧，以及其狀態為何是么半群；
- 使用 Welford/Chan 統計進行單次走訪 LayerNorm，以及 RMSNorm；
- 為什麼樸素注意力會受限於 $N\times N$ 分數矩陣的記憶體存取；
- FlashAttention：將注意力分塊，讓分數永不離開晶片，並實作一個完整、
  經過測試且包含因果遮罩的 FP32 kernel；
- 生產環境的注意力 kernel 還會加入哪些功能（tensor core、FA2/FA3、
  split-KV 解碼）。

## 1. Softmax

### 1.1 定義與穩定性

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \qquad m = \max_j x_j
$$

| 符號 | 意義 |
|---|---|
| $x$ | 一列 logits |
| $m$ | 該列的最大值 |

兩種形式相等，但只有第二種安全：當 $x > 88.7$ 時，$e^x$ 會超出 FP32
範圍；減去最大值後，每個指數都 $\le 0$，最大的一項則正好為 1。

### 1.2 三次走訪

直接實作會讀取該列三次：第一次求 $m$，第二次求
$z = \sum_j e^{x_j - m}$，第三次寫入 $e^{x_i - m}/z$：

```cpp
__global__ void softmaxThreePass(const float* in, float* out, int cols) {
    const float* x = in + static_cast<size_t>(blockIdx.x) * cols;
    float* y = out + static_cast<size_t>(blockIdx.x) * cols;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, x[c]);   // pass 1
    m = blockAllReduce<true>(m);
    float z = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) z += __expf(x[c] - m);   // pass 2
    z = blockAllReduce<false>(z);
    const float inv = 1.0f / z;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = __expf(x[c] - m) * inv;   // pass 3
}
```

Softmax 對每個元素只執行少量 flop，因此受記憶體限制；其執行時間與
實際抵達 DRAM 的走訪次數成正比：

![對一列資料的記憶體走訪：三次走訪與線上 softmax](figures/ch13-passes.svg)

## 2. 線上 Softmax

### 2.1 重新縮放技巧

攜帶目前最大值 $m$，以及**相對於該最大值**的目前總和 $z$。
當新元素 $x$ 到達時：

$$
m' = \max(m, x), \qquad z' = z\,e^{m - m'} + e^{x - m'}
$$

| 符號 | 意義 |
|---|---|
| $m, z$ | $x$ 到達前的狀態：目前最大值，以及到目前為止所有元素的 $\sum e^{x_j - m}$ |
| $m', z'$ | $x$ 到達後的狀態 |

若 $x$ 沒有提高最大值，$e^{m - m'} = 1$，這就只是
$z + e^{x - m}$；若有提高，舊總和會重新縮放到新的最大值。

![線上 softmax：遇到更大的最大值時，重新縮放目前總和](figures/ch13-online.svg)

### 2.2 么半群

兩個部分狀態可使用同一規則合併。此規則具結合性，單位元素為
$(-\infty, 0)$（第 03 章第 8 節）：

$$
(m_a, z_a) \oplus (m_b, z_b) = \bigl(M,\ z_a e^{m_a - M} + z_b e^{m_b - M}\bigr), \qquad M = \max(m_a, m_b)
$$

| 符號 | 意義 |
|---|---|
| $(m_a, z_a), (m_b, z_b)$ | 該列兩個不重疊部分的 softmax 狀態 |
| $M$ | 合併後的最大值 |

因此，每個 lane 都能處理該列的一部分，再用蝶形 shuffle 合併各 lane
的狀態，方式與加總完全相同：

```cpp
MaxSum s{-INFINITY, 0.0f};
for (int c = lane; c < cols; c += 32) {               // pass 1: max and sum together
    const float v = x[c];
    if (v > s.m) {
        s.z = s.z * __expf(s.m - v) + 1.0f;           // rescale the old sum to the new max
        s.m = v;
    } else {
        s.z += __expf(v - s.m);
    }
}
for (int d = 16; d > 0; d >>= 1) {                    // combine the 32 lanes' states (butterfly)
    const MaxSum o{__shfl_xor_sync(kFullMask, s.m, d), __shfl_xor_sync(kFullMask, s.z, d)};
    s = combine(s, o);
}
const float inv = 1.0f / s.z;
for (int c = lane; c < cols; c += 32) y[c] = __expf(x[c] - s.m) * inv;   // pass 2
```

`combine` 必須處理空狀態：若兩個最大值都是 $-\infty$，
$e^{m - M}$ 會成為 $e^{-\infty + \infty} = \text{NaN}$，因此需直接
傳回空狀態。

### 2.3 選擇映射方式

與所有逐列歸約相同（第 03 章第 6 節）：列長最多數千個元素時，每列
使用一個 warp；更長則每列使用一個區塊。若一列能放入所屬 warp 或
區塊的暫存器（例如 32 個 lane × 32 個值 = 1024 個元素），第一次走訪
後就把它留在那裡。第二次走訪即可讀取暫存器，而 kernel 對每個元素只
搬移必要的 8 位元組。

## 3. 正規化

### 3.1 LayerNorm

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma_i + \beta_i, \qquad
\mu = \frac{1}{n}\sum_i x_i, \qquad \sigma^2 = \frac{1}{n}\sum_i (x_i - \mu)^2
$$

| 符號 | 意義 |
|---|---|
| $x, y$ | 一列（一個 token 的隱藏向量）、輸入與輸出 |
| $\mu, \sigma^2$ | 該列的平均值與（有偏）變異數 |
| $\gamma, \beta$ | 每欄一個的學習縮放與位移 |
| $\epsilon$ | 小型常數（例如 $10^{-5}$） |

看似方便的單次走訪公式
$\sigma^2 = \overline{x^2} - \mu^2$，會在
$|\mu| \gg \sigma$ 時相減兩個很大且十分接近的數字。程式測試使用
平均值約為 1000、散布約為 1 的列：在 FP32 中，該公式幾乎會失去
所有有效位數，Welford 更新則不會。

### 3.2 使用 Welford 與 Chan 進行單次走訪

每個執行緒對自己的元素執行 Welford 更新，再使用 Chan 公式
（第 03 章第 8.2 節）合併部分 $(n, \mu, M_2)$ 狀態：先在 warp 內
使用 shuffle 合併，再透過共享記憶體跨 warp 合併：

```cpp
Moments acc{0.0f, 0.0f, 0.0f};
for (int c = threadIdx.x; c < cols; c += blockDim.x) {   // Welford update, one element at a time
    acc.n += 1.0f;
    const float delta = x[c] - acc.mean;
    acc.mean += delta / acc.n;
    acc.m2 += delta * (x[c] - acc.mean);
}
acc = warpAllReduceMoments(acc);
```

接著第二次讀取該列以寫入 $y$（或像 softmax 一樣保留在暫存器中）。

### 3.3 RMSNorm 與融合

RMSNorm 省略平均值：
$y_i = x_i \gamma_i / \sqrt{\tfrac{1}{n}\sum_j x_j^2 + \epsilon}$。
它只需要平方和，也沒有消去誤差問題。在 Transformer 區塊中，通常會把
它與前一個殘差加法融合
（$x \leftarrow x + \text{sublayer}(x)$，然後正規化），以省下寫入後
重新讀取 $x$
（[LeetGPU – 融合殘差加法 + RMSNorm](../leetgpu/083-fused-residual-add-rms-norm/)）。

## 4. 注意力及其記憶體問題

### 4.1 定義

$$
O = \operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d}} + M\right) V
$$

| 符號 | 意義 |
|---|---|
| $Q, K, V$ | Query、key、value：各為 $N\times d$（一個 head） |
| $d$ | Head 維度（通常為 64 或 128） |
| $N$ | 序列長度 |
| $M$ | 遮罩：值為 0，或在 query 不得看到 key 的位置為 $-\infty$（因果注意力中為 $j > i$） |
| $O$ | 輸出，$N\times d$ |
| softmax | 套用到 $N\times N$ 分數矩陣的每一列 |

### 4.2 樸素方法的成本

教科書實作會寫入 $S = QK^{\mathsf T}/\sqrt{d}$（$N\times N$），
重新讀取它以執行 softmax，再寫入 $P$，最後為 $PV$ 重新讀取 $P$：

$$
W = 4N^2 d, \qquad Q_{\text{naive}} \approx 4\,(4Nd + 4N^2)\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 兩次矩陣乘法的 flop 數 |
| $Q_{\text{naive}}$ | DRAM 流量：$Q, K, V, O$，加上寫入並讀取 $S$ 與 $P$（FP32） |

當 $N = 4096$、$d = 64$ 時，分數流量為 $16N^2 = 268$ MB，而輸入
與輸出的流量只有 $16Nd = 4$ MB：$N^2$ 項占絕大部分，強度僅約為
$4N^2d / 16N^2 = d/4 = 16$ flop/B，而且記憶體需求會隨序列長度
呈平方成長。

## 5. FlashAttention

### 5.1 概念

分塊處理 key，並將每個 query 列的 softmax 視為跨 tile 的線上
softmax（第 2 節）。對一列 query 而言，處理 key tile $t$ 後：

$$
m_t = \max\bigl(m_{t-1}, \max_j s_j\bigr), \qquad
\ell_t = \ell_{t-1}\,e^{m_{t-1} - m_t} + \sum_j e^{s_j - m_t}, \qquad
\mathbf{o}_t = \mathbf{o}_{t-1}\,e^{m_{t-1} - m_t} + \sum_j e^{s_j - m_t}\,\mathbf{v}_j
$$

| 符號 | 意義 |
|---|---|
| $s_j$ | 該列與 tile $t$ 中 key $j$ 的分數 |
| $m_t, \ell_t$ | 目前最大值與目前 softmax 分母 |
| $\mathbf{o}_t$ | 目前尚未正規化的輸出列（$d$ 個值） |
| $\mathbf{v}_j$ | Tile $t$ 中的 value 列 |

最後，$O_i = \mathbf{o}_T / \ell_T$。一個 tile 的分數只存在暫存器中，
絕不儲存任何大小為 $N\times N$ 的資料。

![FlashAttention：讓 K/V tile 流經一個 query 區塊；S 永不離開晶片](figures/ch13-flash.svg)

### 5.2 Kernel

程式的 kernel 為了清楚而使用 CUDA core 撰寫（FP32，$d = 64$）：

| 選擇 | 值 | 原因 |
|---|---|---|
| 每個區塊的 query 數 | 16（4 個 warp × 4 列） | 載入共享記憶體的每個 K/V tile 由 16 列重複使用 |
| 每個 tile 的 key 數 | 32 | 每個 lane 一個 key：分數是該 lane 的點積 |
| 輸出擁有權 | Lane $\ell$ 擁有維度 $\ell$ 和 $\ell + 32$ | $PV$ 乘積分散到整個 warp |

```cpp
for (int kv0 = 0; kv0 < kv_end; kv0 += kBlockKv) {
    __syncthreads();                                  // previous tile fully used (and q_s written)
    // ... load K and V tile kv0 into k_s, v_s (zeros past the end) ...
    __syncthreads();
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int qr = warp * kRowsPerWarp + r;       // row inside the block
        const int qi = q0 + qr;                       // global query index
        const int kj = kv0 + lane;                    // this lane's key
        float s = 0.0f;
        for (int c = 0; c < kHeadDim; ++c) s = fmaf(q_s[qr][c], k_s[lane][c], s);
        if (kj >= n || (causal && kj > qi)) s = -INFINITY;
        const float m_new = fmaxf(m[r], warpMax(s));
        if (m_new == -INFINITY) continue;             // nothing visible yet for this row
        const float p = __expf(s - m_new);            // this lane's unnormalized probability
        const float rescale = __expf(m[r] - m_new);
        l[r] = l[r] * rescale + warpSum(p);
        m[r] = m_new;
        for (int e = 0; e < kDimsPerLane; ++e) acc[r][e] *= rescale;
        for (int j = 0; j < kBlockKv; ++j) {
            const float pj = __shfl_sync(kFullMask, p, j);
            for (int e = 0; e < kDimsPerLane; ++e) acc[r][e] = fmaf(pj, v_s[j][lane + 32 * e], acc[r][e]);
        }
    }
}
```

重要細節：

- **縮放。** $1/\sqrt{d}$ 會在載入 $Q$ 時一次融合進去。
- **記憶體庫衝突。** Lane $j$ 會讀取 `k_s` 的第 $j$ 列；若每列含
  64 個 float，全部 32 個 lane 都會命中同一個記憶體庫。把每列填補
  到 65 個字即可避免衝突（第 02 章第 4.3 節）。`q_s[qr][c]` 是廣播，
  `v_s[j][lane + 32e]` 則讀取連續字。
- **因果遮罩。** Query 之後的 key 會設為 $-\infty$；起點位於該區塊
  最後一個 query 之後的 key tile 會直接略過（`kv_end`），因此省下一半
  工作量。
- **空狀態。** 對一列的前幾個 tile 而言，所有分數都可能被遮蔽；
  `m_new == -INFINITY` 會略過更新，避免計算
  $e^{-\infty + \infty}$。
- **暫存器。** 所有跨列與維度的迴圈都會展開，因此 `m`、`l` 和 `acc`
  會留在暫存器中（ptxas：54 個暫存器，不使用堆疊）。

### 5.3 流量

每個區塊會讀取自己的 query 一次，並完整讀取 $K$ 與 $V$ 一次：

$$
Q_{\text{flash}} = 4\left(2Nd + 2Nd\left\lceil \frac{N}{B_q} \right\rceil\right), \qquad
\frac{Q_{\text{naive}}}{Q_{\text{flash}}} \approx \frac{16N^2}{8N^2d/B_q} = \frac{2B_q}{d}
$$

| 符號 | 意義 |
|---|---|
| $B_q$ | 每個區塊的 query 列數（此處為 16；生產環境 kernel 為 64–128） |
| $Q_{\text{flash}}$ | DRAM 流量；每個 query 區塊都會重新讀取 $K$ 和 $V$（實務上大多由 L2 吸收） |

所需記憶體為 $O(Nd)$，而非 $O(N^2)$，因此才能支援長上下文。更大的
$B_q$ 可進一步減少 $K/V$ 重複讀取，代價則是共享記憶體與暫存器用量。

### 5.4 生產環境 Kernel 加入的功能

| 功能 | 概念 |
|---|---|
| Tensor core | 使用 FP16/BF16 的 `mma.sync`／`wgmma` tile 計算 $S = QK^{\mathsf T}$ 和 $O \mathrel{+}= PV$（[矩陣乘法 8](gemm/07-tensor-cores.md)）；softmax 直接對暫存器中的累加器片段執行 |
| FlashAttention-2 | 同時對 query 區塊和 head/batch 平行化；warp 分割 query 而不是 key，因此不需跨 warp 歸約 |
| FlashAttention-3（Hopper） | TMA 載入及非同步 `wgmma`、生產者／消費者 warp，讓一個 tile 的 softmax 與下一個 tile 的 GEMM 重疊 |
| 反向傳播 | 逐 tile 從 $Q$、$K$ 和已儲存的 $(m, \ell)$ 重新計算 $S$ 與 $P$，而不儲存 $P$ |
| Split-KV 解碼 | 每個序列只有一個 query（解碼）時，將 *key* 分給多個區塊，並使用第 2.2 節的么半群合併部分 $(m, \ell, \mathbf{o})$ 狀態 |
| GQA／MQA、分頁 KV | 多個 query head 共用一個 K/V head；K/V 快取儲存在固定大小的頁面中，並透過表格定址 |

## 重點整理

1. 取指數前先減去該列最大值；softmax 和正規化受記憶體限制，因此要
   計算記憶體走訪次數。
2. 線上 softmax 攜帶 $(m, z)$，並在遇到新的最大值時重新縮放；
   這些狀態形成么半群，因此可跨 lane、warp 與區塊歸約。
3. 使用 Welford/Chan 計算變異數，不要使用
   $\overline{x^2} - \mu^2$。
4. 樸素注意力以 $N\times N$ 分數流量為主；FlashAttention 讓 K/V tile
   流經每個 query 列，並套用線上 softmax，因此分數永不離開晶片，
   記憶體用量為 $O(Nd)$。
5. 生產環境 kernel 使用相同演算法，但會把兩次乘積移到 tensor core，
   並加入更多平行處理與非同步載入。

## 練習

1. 證明第 2.2 節的合併操作具結合性。

    <details markdown="1"><summary>答案</summary>

    將每個狀態寫成 $(m, z) \sim z e^{m}$：合併相當於加上
    $z_a e^{m_a} + z_b e^{m_b}$，再相對於較大的最大值重新表示。
    加法具結合性，因此合併也具結合性（最大值亦然）。

    </details>

2. 當 $N = 8192$、$d = 128$、$B_q = 64$ 時，比較
   $Q_{\text{naive}}$ 與 $Q_{\text{flash}}$。

    <details markdown="1"><summary>答案</summary>

    樸素方法：
    $4(4\cdot8192\cdot128 + 4\cdot8192^2) \approx 1.09$ GB，幾乎全是
    分數流量。Flash：
    $4(2\cdot8192\cdot128 + 2\cdot8192\cdot128\cdot128)
    \approx 1.08$ GB（尚未計入 L2 重複使用），兩者是同一數量級！
    實務上的差異在於，重複讀取的 $K/V$ 會命中 L2（每個區塊反覆讀取
    相同的數 MB），但樸素方法的 $S$ 和 $P$ 流量無法如此。增加 $B_q$
    或把 query 區塊分配到共用 K/V 的 head，也能進一步改善。

    </details>

3. 修改 `softmaxOnline`，讓長度不超過 1024 個元素的列中，每個 lane
   都把自己的 32 個值保留在暫存器，並且只從記憶體讀取該列一次。

4. 在 `flashAttention` 中加入滑動視窗遮罩（一個 query 只能看到前
   $w$ 個 key）。哪些 key tile 可以略過？

    <details markdown="1"><summary>答案</summary>

    當 $i - w < j \le i$ 時，query $i$ 可以看到 key $j$。
    Query 範圍為 $[q_0, q_0 + B_q)$ 的區塊需要
    $(q_0 - w, q_0 + B_q)$ 中的 key：從包含
    $\max(0, q_0 - w + 1)$ 的 tile 開始迴圈，並在 `kv_end` 停止。

    </details>

## 實作練習

- [LeetGPU – Softmax](../leetgpu/005-softmax/)、[Tensara – Softmax](../tensara/softmax/)、
  [Tensara – Log Softmax](../tensara/log-softmax/)
- [LeetGPU – Layer Normalization](../leetgpu/113-layer-normalization/)、[Tensara – Layer Norm](../tensara/layer-norm/)、
  [LeetGPU – RMS Normalization](../leetgpu/050-rms-normalization/)、[Tensara – RMS Norm](../tensara/rms-norm/)
- [LeetGPU – Softmax 注意力](../leetgpu/006-softmax-attention/)、
  [Tensara – 縮放點積注意力](../tensara/scaled-dot-attention/)
- [LeetGPU – 因果注意力](../leetgpu/053-casual-attention/)、
  [LeetGPU – 滑動視窗注意力](../leetgpu/059-sliding-window-attn/)、
  [LeetGPU – 分組 Query 注意力](../leetgpu/080-grouped-query-attention/)
