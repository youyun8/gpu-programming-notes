# 12 – 卷積與樣板

> **第二部分 · 平行模式** · 先備知識：[02](02-memory-hierarchy.md) ·
> 程式：[`examples/12-convolution-stencil.cu`](examples/12-convolution-stencil.cu) ·
> 下一章：[13 – Softmax、LayerNorm 與 FlashAttention](13-softmax-attention.md)

卷積會從相鄰輸入的小視窗計算每個輸出，並由濾波器為輸入加權；樣板
使用固定權重進行相同操作，通常會反覆套用（一次模擬時間步）。相鄰輸出
幾乎共享所有輸入，因此整個問題的重點，是如何只載入每個輸入一次，
再供所有需要它的輸出重複使用。

**你將學到**

- 卷積的定義（一維、二維、三維）、邊界條件與成本模型；
- 為什麼濾波器適合放在 `__constant__` 記憶體；
- 使用 halo 分塊：一維與二維 kernel，以及 halo 的額外成本；
- 可分離濾波器，以及每個執行緒計算多個輸出；
- 樣板：三維 7 點 Jacobi 步驟、樸素方法與 2.5 維分塊；
- 何時應改用 im2col + GEMM、FFT 或 Winograd。

## 1. 定義

### 1.1 卷積

$$
y_i = \sum_{k=-R}^{R} f_k\,x_{i+k}, \qquad
y_{r,c} = \sum_{a=-R}^{R}\sum_{b=-R}^{R} f_{a,b}\,x_{r+a,\,c+b}
$$

| 符號 | 意義 |
|---|---|
| $x, y$ | 輸入與輸出訊號（或影像） |
| $f$ | 每個維度有 $2R + 1$ 個 tap 的濾波器（kernel） |
| $R$ | 濾波器半徑 |
| $i$；$r, c$ | 輸出位置：索引；列與欄 |

嚴格而言，這是*互相關*（卷積會翻轉濾波器）；深度學習使用不翻轉的
形式並稱其為卷積，本筆記也沿用此稱呼。

### 1.2 邊界

接近邊緣的輸出會需要不存在的輸入。常見選擇如下：

| 模式 | 輸出大小 | 缺少的輸入 |
|---|---|---|
|「Same」、補零（本程式） | 與輸入相同 | 當作 0 讀取 |
|「Valid」 | $n - 2R$ | 只計算視窗完全位於範圍內的輸出 |
| 複製／反射 | 與輸入相同 | 限制索引範圍或鏡射索引 |

LeetGPU 和 Tensara 題目會指定使用哪一種；程式碼只在載入時的索引測試
有所不同。

### 1.3 樣板

樣板會使用固定權重，根據相鄰點更新網格中的每一點。熱方程或 Poisson
求解器會使用以下三維 7 點 Jacobi 步驟：

$$
u^{(t+1)}_{x,y,z} = c_0\,u^{(t)}_{x,y,z} + c_1\left(u^{(t)}_{x\pm1,y,z} + u^{(t)}_{x,y\pm1,z} + u^{(t)}_{x,y,z\pm1}\right)
$$

| 符號 | 意義 |
|---|---|
| $u^{(t)}$ | 時間步 $t$ 的網格值 |
| $c_0, c_1$ | 中心點和 6 個面相鄰點的權重 |
| $x\pm1$ | 沿 $x$ 軸的兩個相鄰點（$y$、$z$ 軸亦同） |

邊界單元會保持固定（Dirichlet 條件）。每個步驟之間會交換兩個緩衝區
（「乒乓」），因為每個點都必須讀取相鄰點的舊值。

## 2. 成本模型

對每個輸出而言，$d$ 維卷積會執行 $(2R + 1)^d$ 次乘加。每個輸入和
輸出都至少必須存取一次：

$$
I = \frac{2\,(2R + 1)^d}{8}\ \frac{\text{flop}}{\text{byte}}, \qquad
\text{loads per output: naive } (2R + 1)^d, \quad \text{tiled } \frac{(T + 2R)^d}{T^d}
$$

| 符號 | 意義 |
|---|---|
| $I$ | 算術強度（FP32 輸入與輸出：每個輸出 8 位元組） |
| $d$ | 維度數量 |
| $T$ | 每個區塊的輸出 tile 寬度 |

$3\times3$ 濾波器的 $I = 2.25$，$7\times7$ 濾波器則為
$I = 12.25$ flop/B：小型濾波器受記憶體限制，大型濾波器接近轉折點
（第 00 章）。因此，兩者的目標都是只從 DRAM 載入每個輸入一次。
樸素 kernel 對每個輸出發出 $(2R + 1)^d$ 次載入，並依賴 L1/L2
吸收它們；分塊 kernel 則大約只發出一次。

## 3. 把濾波器放在常數記憶體

```cpp
__constant__ float c_filter2d[(2 * kMaxRadius2d + 1) * (2 * kMaxRadius2d + 1)];
...
cudaMemcpyToSymbol(c_filter2d, host_filter.data(), taps * sizeof(float));
```

在內層迴圈中，一個 warp 的每個 lane 會同時讀取**相同** tap，這正是
常數記憶體能在一個週期內廣播的存取模式（第 02 章第 1.6 節）。
濾波器也是唯讀且很小（上限為 64 KB）。若每個輸出通道的濾波器不同
（例如 CNN），則改為放在共享記憶體或暫存器。

## 4. 使用 Halo 的一維分塊

一個含 256 個執行緒的區塊會計算連續 256 個輸出。它們需要相同位置的
256 個輸入，再加上左右各 $R$ 個輸入，也就是 **halo**：

![一維卷積：區塊的輸入 tile 是輸出 tile 加上左右各 R 的 halo](figures/ch12-halo-1d.svg)

```cpp
__global__ void conv1dTiled(const float* in, float* out, int n, int radius) {
    __shared__ float tile[kThreads1d + 2 * kMaxRadius1d];
    const int base = blockIdx.x * kThreads1d;
    for (int i = threadIdx.x; i < kThreads1d + 2 * radius; i += blockDim.x) {
        const int g = base - radius + i;
        tile[i] = (g >= 0 && g < n) ? in[g] : 0.0f;          // zero padding outside the input
    }
    __syncthreads();
    const int o = base + threadIdx.x;
    if (o >= n) return;                                      // no barrier below: safe
    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k)                  // all lanes read the same c_filter1d[k]: broadcast
        acc = fmaf(c_filter1d[k + radius], tile[threadIdx.x + radius + k], acc);
    out[o] = acc;
}
```

- 載入迴圈會以 256 個執行緒處理 $256 + 2R$ 個元素（有些執行緒會
  載入兩個）。這些讀取會合併。
- 在內層迴圈中，lane 會讀取連續的共享記憶體字
  （`tile[t + k]`），不會發生衝突。
- 提早 `return` 位於唯一的屏障之後，因此是安全的。

## 5. 二維分塊

### 5.1 Tile

![二維分塊：16 × 16 的輸出 tile 需要 (16 + 2R)² 的輸入 tile](figures/ch12-halo-2d.svg)

```cpp
__global__ void conv2dTiled(const float* in, float* out, int height, int width, int radius) {
    constexpr int kSide = kTile2d + 2 * kMaxRadius2d;
    __shared__ float tile[kSide][kSide + 1];                 // +1: column reads of the halo are conflict-free
    const int x0 = blockIdx.x * kTile2d - radius;            // input coordinates of tile[0][0]
    const int y0 = blockIdx.y * kTile2d - radius;
    const int side = kTile2d + 2 * radius;
    for (int i = threadIdx.y * kTile2d + threadIdx.x; i < side * side; i += kTile2d * kTile2d) {
        const int ty = i / side, tx = i % side;
        const int yy = y0 + ty, xx = x0 + tx;
        tile[ty][tx] = (yy >= 0 && yy < height && xx >= 0 && xx < width) ? in[yy * width + xx] : 0.0f;
    }
    __syncthreads();
    ...
    for (int dy = 0; dy < fside; ++dy)
        for (int dx = 0; dx < fside; ++dx)
            acc = fmaf(c_filter2d[dy * fside + dx], tile[threadIdx.y + dy][threadIdx.x + dx], acc);
```

### 5.2 Halo 的成本

每個接觸 halo 的區塊都會載入它，因此載入的額外成本就是輸入 tile
與輸出 tile 的比例：

$$
\text{overhead} = \frac{(T + 2R)^2}{T^2}
$$

| 符號 | 意義 |
|---|---|
| $T$ | 輸出 tile 寬度（此處為 16） |
| $R$ | 濾波器半徑 |

| $T$ | $R = 1$ | $R = 3$ | $R = 4$ |
|---|---|---|---|
| 8 | 1.56 | 3.06 | 4.0 |
| 16 | 1.27 | 1.89 | 2.25 |
| 32 | 1.13 | 1.41 | 1.56 |

較大的 tile 可攤平 halo 成本，卻需要更多共享記憶體和執行緒。
常見折衷是使用 $32\times8$ 區塊，讓每個執行緒沿 $y$ 軸計算多個輸出
（暫存器分塊）：tile 為 $32\times32$，使用 256 個執行緒，而載入
暫存器的每一輸入列可供多個輸出使用。

### 5.3 可分離濾波器

許多濾波器（Gaussian、box、Sobel 的分量）都是外積
$f_{a,b} = g_a h_b$。此時，二維卷積可分成兩次一維走訪：

$$
y = g * (h * x), \qquad \text{MACs per output: } (2R + 1)^2 \ \to\ 2\,(2R + 1)
$$

| 符號 | 意義 |
|---|---|
| $g, h$ | 濾波器的欄因子與列因子 |
| $*$ | 沿欄（$g$）或列（$h$）的一維卷積 |

當 $R = 3$ 時，乘加次數會從 49 降為 14，代價是寫入後重新讀取中間
影像（或把它留在共享記憶體）。

## 6. 樣板

### 6.1 樸素 Kernel

```cpp
const size_t plane = static_cast<size_t>(nx) * ny;
out[i] = c0 * in[i] + c1 * (in[i - 1] + in[i + 1] + in[i - nx] + in[i + nx] + in[i - plane] + in[i + plane]);
```

每個點要載入七次。$x$ 軸相鄰點會命中該點本身所在的快取列，
$y$ 軸相鄰點可能已由相鄰 warp 載入；$z$ 軸相鄰點則相隔整個平面，
只有當 $n_x n_y$ 平面能放入 L2 時，才會留在 L2。

### 6.2 2.5 維分塊

一個區塊擁有定義域中一個 $32\times8$ 的柱體，並沿 $z$ 軸走訪。
每個執行緒把自己柱體在 $z-1$、$z$、$z+1$ 平面的值保存在暫存器中，
而目前平面則放入共享記憶體，讓 $x$ 和 $y$ 軸的相鄰點讀取：

![2.5 維分塊：區塊沿 z 軸走過它在定義域中的柱體](figures/ch12-stencil-25d.svg)

```cpp
float below = load(x, y, 0);                             // registers: planes z-1, z, z+1 of my column
float cur = load(x, y, 1);
for (int z = 1; z < nz - 1; ++z) {
    const float above = load(x, y, z + 1);
    __syncthreads();                                     // everyone is done reading the previous plane
    plane[ty + 1][tx + 1] = cur;
    if (tx == 0) plane[ty + 1][0] = load(x - 1, y, z);   // halo columns and rows of plane z
    if (tx == kBx - 1) plane[ty + 1][kBx + 1] = load(x + 1, y, z);
    if (ty == 0) plane[0][tx + 1] = load(x, y - 1, z);
    if (ty == kBy - 1) plane[kBy + 1][tx + 1] = load(x, y + 1, z);
    __syncthreads();
    if (inside) {
        const float neighbours = plane[ty + 1][tx] + plane[ty + 1][tx + 2] + plane[ty][tx + 1] +
                                 plane[ty + 2][tx + 1] + below + above;
        out[at(x, y, z, nx, ny)] = boundary_xy ? cur : c0 * cur + c1 * neighbours;
    }
    below = cur;
    cur = above;
}
```

- **全域載入**：每個點一次（上方平面），再加上 halo（每個
  $32\times8$ 平面有 $2 \cdot 32 + 2 \cdot 8 = 80$ 次，增加 31%）。
- **每個平面兩次屏障**，分別處理第 04 章第 3.3 節的兩種危障：
  所有人讀取前，平面已完整寫入；下一個平面覆寫前，平面已完整讀取。
  移除第一次屏障會讓程式檢查失敗。
- **平行度**：只有 $\frac{n_x}{32}\cdot\frac{n_y}{8}$ 個區塊。
  對 $512\times512$ 的橫截面而言有 1024 個區塊，已經足夠；若定義域
  很薄，可將 $z$ 分成幾個區塊，各自具有 halo 平面。

### 6.3 時間分塊

7 點樣板的一個步驟會對每個點執行 8 flop（2 次乘法、6 次加法），
並至少搬移 8 位元組（讀寫各一個 float）：約為 1 flop/B，因此不論
分塊多好，都受記憶體限制。若每次走訪 tile 時執行 $k$ 個時間步
（把中間步驟保存在共享記憶體，halo 寬度為 $k$ 個單元），DRAM 流量
大約可除以 $k$。Halo 會隨 $k$ 增長，因此 GPU 上的 $k$ 通常為 2–4。

## 7. 直接卷積以外的方法

| 方法 | 概念 | 適用時機 |
|---|---|---|
| im2col + GEMM | 把每個輸入視窗複製到矩陣的一欄；由一次大型 GEMM 套用所有濾波器 | 通道數很多的 CNN 層（cuDNN、CUTLASS 隱式 GEMM；後者會省略明確複製） |
| FFT | 卷積在頻域中是逐點乘積 | 大型濾波器（$R$ 為數十或更大） |
| Winograd | 對小型 tile 和 $3\times3$ 濾波器使用較少乘法 | $3\times3$ CNN 層 |
| Depthwise | 每個通道一個濾波器，不跨通道歸約 | 受記憶體限制；可直接套用本章的分塊方式 |

## 重點整理

1. 相鄰輸出會共享輸入：把一個 tile 及其 halo 載入共享記憶體一次，
   再重複使用 $(2R + 1)^d$ 次。
2. 若所有執行緒會以相同順序讀取濾波器，請把它放在
   `__constant__` 記憶體。
3. Halo 額外成本為 $(T + 2R)^d / T^d$；較大的 tile（或暫存器分塊）
   可攤平成本。
4. 可分離濾波器會把 $(2R + 1)^2$ 的工作量降為 $2(2R + 1)$。
5. 對三維樣板而言，沿一個軸走訪，並把平面保存在暫存器與共享記憶體
   （2.5 維分塊）；若要提高重複使用率，則在時間上分塊。

## 練習

1. 計算 `conv2dTiled` 在 $R = 4$ 時的 halo 額外成本，以及由
   $32\times8$ 個執行緒計算 $32\times32$ 輸出 tile（每個執行緒沿
   $y$ 軸計算 4 個輸出）時的額外成本。

    <details markdown="1"><summary>答案</summary>

    $(16 + 8)^2 / 16^2 = 2.25$；$(32 + 8)^2 / 32^2 = 1.5625$。

    </details>

2. 把 `conv1dTiled` 改成「valid」模式（輸出長度為 $n - 2R$）。
   載入迴圈和輸出索引各要如何修改？

    <details markdown="1"><summary>答案</summary>

    輸出 $o$ 會讀取輸入 $o \dots o + 2R$，因此 tile 從 `base`
    開始（不向左位移），除了超過輸入末端外，不需補零；輸出條件改為
    `o < n - 2 * radius`。

    </details>

3. 使用兩次 `conv1dTiled` 走訪實作可分離 Gaussian 模糊：一次沿列，
   一次沿欄（提示：欄走訪可先轉置，或讓每個區塊使用高而非寬的 tile）。
   比較它在 $R = 4$ 時與 `conv2dTiled` 的執行時間。

4. 在 `stencil3d25D` 中，哪些載入*沒有*合併？要如何避免？

    <details markdown="1"><summary>答案</summary>

    左右 halo 欄：各有 8 個執行緒載入相隔 32 個 float 的元素。
    可讓整個 warp 載入（例如 warp 0 載入兩欄共 16 個 halo 值），
    或從較寬且對齊的列區段讀取 halo，以降低成本；它們在每個平面的
    336 次載入中只占 16 次。

    </details>

## 實作練習

- [LeetGPU – 一維卷積](../leetgpu/009-1d-convolution/)、[Tensara – 一維卷積](../tensara/conv-1d/)
- [LeetGPU – 二維卷積](../leetgpu/010-2d-convolution/)、[Tensara – 二維卷積](../tensara/conv-2d/)、
  [LeetGPU – Gaussian 模糊](../leetgpu/028-gaussian-blur/)
- [LeetGPU – 三維卷積](../leetgpu/011-3d-convolution/)、[Tensara – 三維方形卷積](../tensara/conv-square-3d/)
- [LeetGPU – 二維 Jacobi 樣板](../leetgpu/069-jacobi-stencil-2d/)、[Tensara – 方框模糊](../tensara/box-blur/)、
  [LeetGPU – 因果 Depthwise Conv1D](../leetgpu/090-causal-depthwise-conv1d/)
