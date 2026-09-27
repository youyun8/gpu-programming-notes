# 04.4 – Warp 分塊

> **第三部分 · 矩陣乘法 · 04.x GEMM 深入解析** ·
> 程式：[`04-warp-tiling.cu`](04-warp-tiling.cu) · 延續：[04.1](01-vectorized-loads.md)、[04.2](02-double-buffering.md) ·
> 下一篇：[04.5 – 分塊 Swizzle](05-tile-swizzling.md)

第 04 章把輸出分成兩層：block 分塊（共享記憶體）與 thread 分塊
（暫存器）。兩者之間還有硬體本來就具備的一層：warp。Warp 以 32 個 lane
為單位發出一條指令，共享記憶體指令也是整個 warp 一起處理，因此影響共享
記憶體流量的是 **warp** 存取的整組位址。Warp 分塊讓每個 warp 負責 block
分塊中一個緊密的子分塊，使這組位址更小且更規則。

**你將學到**

- 為何影響共享記憶體流量的單位是 warp，而不是 thread；
- 如何把 block 分塊拆成 warp 分塊、子分塊與 lane 區塊；
- 如何計算 warp 的共享記憶體 footprint，並選擇 lane 網格；
- 為何 warp 分塊是通往 tensor-core kernel 的橋樑。

## 1. 三個層級

![Warp 分塊：2 × 4 個 64 × 32 warp，每個由 8 × 4 lane 網格涵蓋](../figures/gemm-warp-tile.svg)

```cpp
constexpr int kWarpsM = 2, kWarpsN = 4;                  // 8 warps
constexpr int kWarpTileM = kBlockM / kWarpsM;            // 64
constexpr int kWarpTileN = kBlockN / kWarpsN;            // 32
constexpr int kLanesM = 8, kLanesN = 4;                  // lane grid inside a warp
constexpr int kThreadM = 4, kThreadN = 4;                // one patch per lane
constexpr int kIterM = kWarpTileM / (kLanesM * kThreadM);  // 2 sub-tiles along M
constexpr int kIterN = kWarpTileN / (kLanesN * kThreadN);  // 2 sub-tiles along N
```

Warp $w$ 的 lane $\ell$ 負責 block 分塊中的元素 $(r, c)$，其中

$$
r = 64\left\lfloor \tfrac{w}{4} \right\rfloor + 32\,i_m + 4\left\lfloor \tfrac{\ell}{4} \right\rfloor + i, \qquad
c = 32\,(w \bmod 4) + 16\,i_n + 4\,(\ell \bmod 4) + j
$$

| 符號 | 意義 |
|---|---|
| $w, \ell$ | Warp 索引（0–7）與 lane 索引（0–31） |
| $i_m, i_n$ | Warp 分塊中的子分塊，兩者皆為 0–1 |
| $i, j$ | Lane 的 $4\times4$ 區塊內位置，兩者皆為 0–3 |

每個 lane 仍負責 $2\cdot2\cdot4\cdot4 = 64$ 個輸出，與 04.1 相同；
改變的只是負責的是*哪* 64 個。

## 2. 帶來的效益

每個 $k$ 步驟中，warp 所需的共享記憶體資料包括：對應其列範圍的一段 $A$
欄，以及對應其欄範圍的一段 $B$ 列：

$$
Q_{\text{warp}} = W_M + W_N \quad \text{floats per } k, \qquad
\frac{\text{FMAs}}{\text{float read}} = \frac{W_M W_N}{W_M + W_N}
$$

| 符號 | 意義 |
|---|---|
| $W_M \times W_N$ | 一個 warp 涵蓋的 block 分塊列數與欄數 |
| $Q_{\text{warp}}$ | Warp 在每個 $k$ 步驟從共享記憶體讀取的不同 float 數 |

| 配置 | Warp 涵蓋範圍 | $Q_{\text{warp}}$ | 每個 float 的 FMA 數 |
|---|---|---|---|
| 04.1：$16\times2$ 個 thread，分割區塊 | $16\times128$ | 144 | 14.2 |
| Warp 分塊：$8\times4$ 個 lane，$2\times2$ 個子分塊 | $64\times32$ | 96 | 21.3 |

兩種 warp 每個 $k$ 都計算 2048 個輸出，但使用 warp 分塊者從共享記憶體
少讀取三分之一。對每個 SM 每週期共享記憶體頻寬為 128 位元組的 GPU，
這正是共享記憶體是否會成為共同瓶頸的差異。每次載入仍然不會衝突：
具有相同 $\lfloor \ell / 4 \rfloor$ 的 lane 會讀取相同的 $A$ `float4`
（broadcast），warp 的 4 個不同 $B$ `float4` 則彼此連續。

另有兩項優點：

- **可對應到 tensor core。** Tensor-core 指令由一個 warp 針對固定 fragment
  形狀發出。把 lane 的 $4\times4$ outer product 換成 $16\times8$ MMA，
  block 與 warp 層級都不需改變（[04.7](07-tensor-cores.md)）。
- **各層級可解耦。** Block 分塊、warp 分塊與 thread 分塊都是獨立參數
  （但須符合 `constexpr` 區塊中的整除限制）；CUTLASS 與 TensileLite
  （[第 07 章](../07-hipblaslt-tensilelite.md)）就是這樣描述 kernel。

## 3. 程式碼

全域載入、轉置的 $A$ 儲存方式及雙緩衝都與 04.2 相同。Fragment 載入改為
每個子分塊載入一個 `float4`：

```cpp
const int m_base = warp_m * kWarpTileM + lane_m * kThreadM;   // lane_m = lane / 4
const int n_base = warp_n * kWarpTileN + lane_n * kThreadN;   // lane_n = lane % 4
...
for (int im = 0; im < kIterM; ++im) {
    const float4 v = *reinterpret_cast<const float4*>(&a_s[buf][kk][m_base + im * kLanesM * kThreadM]);
    ...
}
for (int in = 0; in < kIterN; ++in) {
    const float4 v = *reinterpret_cast<const float4*>(&b_s[buf][kk][n_base + in * kLanesN * kThreadN]);
    ...
}
```

Epilogue 會把區塊中每個寬度為 4 的列以一個 `float4` 儲存。

## 4. 選擇形狀

`constexpr` 值必須符合以下限制：

$$
\frac{B_M}{W_M}\cdot\frac{B_N}{W_N} = \frac{\text{threads}}{32}, \qquad
\ell_M \ell_N = 32, \qquad
W_M = i_M\,\ell_M\,t_M, \qquad W_N = i_N\,\ell_N\,t_N
$$

| 符號 | 意義 |
|---|---|
| $B_M, B_N$ | Block 分塊（128、128） |
| $W_M, W_N$ | Warp 分塊（64、32） |
| $\ell_M, \ell_N$ | Warp 內的 lane 網格（8、4） |
| $i_M, i_N$ | 每個 warp 的子分塊數（2、2） |
| $t_M, t_N$ | Lane 區塊（4、4） |

越接近正方形的 warp 分塊可讓 $Q_{\text{warp}}$ 最小；lane 網格則應讓每組
8 個 lane 最多只存取 $B$ 的連續 128 位元組。

## 重點整理

1. 共享記憶體流量由一個 warp 存取的位址集合決定：讓每個 warp 負責緊密、近似正方形的分塊。
2. Block 分塊、warp 分塊與 lane 分塊是獨立參數，只需符合簡單的整除限制。
3. 同一套階層可原封不動延伸至 tensor core，此時 MMA 由 warp 發出。

## 練習

1. 嘗試 $W_M\times W_N = 32\times64$（warp 排列為 $4\times2$，lane 排列為
   $4\times8$）。計算 $Q_{\text{warp}}$，並檢查 $B$ 載入的 bank 行為。

    <details markdown="1"><summary>答案</summary>

    $Q_{\text{warp}} = 32 + 64 = 96$ 個 float，數量相同。$N$ 方向有 8 個
    lane，因此 lane 0–7 會讀取 B 的 8 個連續 `float4`（128 位元組）：
    每組只需一個 wavefront，不會衝突；具有相同
    $\lfloor \ell/8 \rfloor$ 的 lane 會 broadcast A。

    </details>
2. 維持 $64\times32$ warp 分塊，但把 $8\times4$ lane 網格改成
   $4\times8$。$A$ 的 broadcast 會有什麼變化？

    <details markdown="1"><summary>答案</summary>

    子分塊變成 $16\times32$，所以每個 lane 負責 $4\times1$ 個子分塊：
    每個 $k$ 需要 16 個 A 值與 4 個 B 值（64 個 FMA 要載入 20 次，而非
    16 次）。每個 A `float4` 由 8 個 lane 共用，而不是 4 個，但每個 lane
    的總載入次數更多；近似正方形的 $2\times2$ 排列較好。

    </details>
