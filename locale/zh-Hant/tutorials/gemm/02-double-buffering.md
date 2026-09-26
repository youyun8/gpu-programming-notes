# 04.2 – 雙緩衝

> **第三部分 · 矩陣乘法 · 04.x GEMM 深入解析** ·
> 程式：[`02-double-buffering.cu`](02-double-buffering.cu) · 延續：[04.1](01-vectorized-loads.md) ·
> 下一篇：[04.3 – 非同步複製](03-async-copies.md)

在 04.1 的 kernel 中，每個 $k$ 切片都會經過相同的四個階段：發出全域載入、
等待完成、存入共享記憶體、進行運算。Warp 等待載入時無事可做，整個 block
也會在 barrier 等待最慢的 warp。雙緩衝會讓切片 $s+1$ 的載入與切片 $s$
的運算重疊。

**你將學到**

- 單緩衝 kernel 如何因每個 $k$ 切片的載入延遲而停頓；
- 兩個共享記憶體緩衝區如何讓下一個切片的載入與目前的運算重疊；
- 為何使用兩個緩衝區時，每個切片只需要一個 barrier；
- 透過暫存器預先擷取的暫存器成本，以及第二種（暫存器層級）雙緩衝。

## 1. 概念

![單緩衝與雙緩衝的時間軸](../figures/gemm-double-buffer.svg)

使用一個緩衝區時，每個切片的時間是載入延遲加上運算時間；使用兩個時，
則是兩者中較大的一個：

$$
t_{\text{single}} \approx S\,(L + C + 2\beta), \qquad
t_{\text{double}} \approx L + S\,(\max(L, C) + \beta)
$$

| 符號 | 意義 |
|---|---|
| $S$ | $k$ 切片數，$\lceil K / B_K \rceil$ |
| $L$ | Warp 所感受到的一個切片全域載入延遲 |
| $C$ | 一個切片的運算時間：每個 thread 執行 $B_K$ 個步驟，每步 4 個 `LDS.128` 與 64 個 FMA |
| $\beta$ | 一次 `__syncthreads()` 的成本（包括等待最慢的 warp） |

Occupancy 也能隱藏延遲（某個 warp 等待時，其他 warp 可進行運算），但每個
thread 使用約 120 個暫存器時，一個 SM 只能容納 16 個 warp，因此 kernel
必須自行隱藏延遲：這是*指令層級*平行，而不是*執行緒層級*平行
（第 01 章第 5 節）。

## 2. 兩個緩衝區，一個 Barrier

![緩衝區輪替：每個緩衝區交替用於讀取與寫入](../figures/gemm-buffer-rotation.svg)

```cpp
__shared__ __align__(16) float a_s[2][kBlockK][kBlockM + kPadA];
__shared__ __align__(16) float b_s[2][kBlockK][kBlockN];
...
storeSlice(0, load4<kVec>(a, ...), load4<kVec>(b, ...));   // prologue: slice 0
__syncthreads();

for (int s = 0; s < num_slices; ++s) {
    const int buf = s % 2;
    const bool has_next = s + 1 < num_slices;
    float4 a_next = ..., b_next = ...;
    if (has_next) {                                  // 1. issue loads of slice s+1
        a_next = load4<kVec>(a, m, k, row0 + a_row, k1 + a_col);
        b_next = load4<kVec>(b, k, n, k1 + b_row, col0 + b_col);
    }
    for (int kk = 0; kk < kBlockK; ++kk) { ... }     // 2. math on buffer buf
    if (has_next) storeSlice(buf ^ 1, a_next, b_next); // 3. registers -> other buffer
    __syncthreads();                                 // 4. one barrier per slice
}
```

有兩個事實使每個切片只用**一個** barrier 仍能保證正確：

1. **在步驟 $s$ 寫入緩衝區 `buf ^ 1` 是安全的。** 最後讀取它的是步驟
   $s-1$ 的運算，而步驟 $s-1$ 結束時有一個 barrier；每個 thread 都通過
   該 barrier 才會開始步驟 $s$。
2. **在步驟 $s+1$ 讀取緩衝區 `buf ^ 1` 是安全的。** 每個 thread 都會在
   結束步驟 $s$ 的 barrier 前，寫入自己負責的切片 $s+1$ 部分。

單緩衝 kernel 每個切片需要兩個 barrier：儲存後一個（資料就緒），運算後
一個（緩衝區可用）。此處步驟 $s$ 結尾的 barrier 會同時完成兩項工作，
但分別作用在不同緩衝區。

## 3. 為何提早發出載入就足夠

全域載入發出時不會阻塞。只有第一條*使用*已載入暫存器的指令執行時，
warp 才會因 scoreboard wait 而停頓。此處該指令是 `storeSlice`，位於步驟
$s$ 的 512 個 FMA 與 32 個 `LDS.128` 之後；只要這些指令所需時間超過載入
延遲，warp 就不會因全域記憶體而停頓。編譯器不得把載入移到運算之後；
它不會這麼做，因為兩者互相獨立，且載入會提早排程。不過仍可用
`cuobjdump -sass` 確認（`LDG.E.128` 會出現在 `FFMA` 區塊之前）。

代價是每個 thread 需要另外 8 個暫存器存放 `a_next` 與 `b_next`
（ptxas：127 個，而非 117 個），共享記憶體則加倍至每個 block 16.6 KB。

## 4. 第二個層級：暫存器

同樣的概念也適用於下一層。內部迴圈可以在執行 $kk$ 的 FMA 時，從共享
記憶體載入 $kk + 1$ 的 fragment，並使用兩組 `a_frag`/`b_frag` 暫存器。
迴圈完全展開後，編譯器通常會自行完成（把下一次迭代的 `LDS.128` 提前）；
因此程式沒有明寫。手寫 kernel（第 06 章的
`a[0:63]` / `a[64:127]` 交換）則必須自行處理。

## 5. 常見陷阱

- **最後一個切片。** `has_next` 必須同時保護載入與儲存。超出結尾的載入會
  越界（此處 `load4` 會補零，所以沒有危險，但仍是浪費的流量）。
- **Barrier 數量。** 若把 `__syncthreads()` 移入 `if (has_next)`，只有在
  不同 thread 的 `has_next` 值不同時才會形成 divergent barrier；此處不會，
  但仍應讓 barrier 保持無條件執行。
- **測試。** GPU 上漏掉 barrier 只會在某些時序下破壞結果。cuemu 會讓每個
  thread 持續執行到阻塞，因此必定能重現失敗；`CUEMU_REVERSE=1`
  （反向排列 thread）還能抓到正向順序剛好掩蓋的變體。

## 重點整理

1. 在運算切片 $s$ 前先發出切片 $s+1$ 的全域載入；warp 只會在使用已載入暫存器的位置停頓。
2. 使用兩個緩衝區時，步驟 $s$ 結尾的 barrier 會同時發布切片 $s+1$ 並釋放緩衝區 $s$。
3. 經由暫存器預先擷取會耗用暫存器，且每個元素需要兩條指令；`cp.async`（04.3）可同時消除兩項成本。

## 練習

1. 刪除最後的 `__syncthreads()`，並以兩種 thread 順序執行 `--test`。

    <details markdown="1"><summary>答案</summary>

    兩種順序都會失敗：沒有 barrier 時，較快的 warp 會開始步驟 $s+1$，
    並在較慢的 warp 存入其負責部分前讀取緩衝區 `buf ^ 1`；它也可能覆寫
    其他 warp 仍在讀取的緩衝區。

    </details>
2. 使用暫存器改成三重緩衝。為何幫助不大？[04.3](03-async-copies.md)
   又採取什麼做法？

    <details markdown="1"><summary>答案</summary>

    每多一個傳輸中的切片，每個 thread 就要再用 8 個暫存器暫存；而每個
    切片已有 512 個 FMA 的工作，通常早已足以隱藏延遲。`cp.async` 不需
    暫存器，而是讓多個切片在共享記憶體中同時進行傳輸。

    </details>
