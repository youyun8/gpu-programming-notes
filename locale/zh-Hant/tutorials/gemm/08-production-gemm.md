# 矩陣乘法 9 – Production GEMM

> **第三部分 · 矩陣乘法** ·
> 先備知識：[矩陣乘法 8 – Tensor Core](07-tensor-cores.md) ·
> 路線圖：[矩陣乘法 1–9](README.md)

針對單一方陣寫出快速 kernel，是一個重要的里程碑；但 production GEMM
真正面對的是 dispatch 問題。矩陣形狀、資料型別、layout、融合操作和 GPU
架構都會改變。最佳實作可能是 persistent kernel、grouped launch、函式庫
呼叫，或是針對某個形狀 bucket 選用的小型自訂 kernel。

**你將學到**

- persistent、batched 與 grouped kernel 適合哪些情況；
- grouped expert GEMM 如何支援 mixture-of-experts 模型；
- CUTLASS 與 CuTe 如何表示分塊 kernel；
- 如何融合 epilogue 而不犧牲精度；
- 如何對有限數量的形狀 bucket 進行 autotune；
- 何時應使用函式庫，何時值得維護自訂 kernel；
- 如何對 production 實作進行 benchmark、profile 與驗證。

## 1. 從工作負載開始

選擇 kernel 前，先記錄完整操作：

$$
D = f\left(\alpha\,\operatorname{op}(A)\operatorname{op}(B)
          + \beta C + \operatorname{bias}\right)
$$

針對每個呼叫位置收集：

- $M$、$N$、$K$、batch size 及其分布；
- 輸入、累加器與輸出型別；
- row-major 或 column-major layout、轉置方式與 leading dimension；
- bias、縮放、activation、residual 與 quantization 操作；
- latency、throughput、workspace 與 determinism 需求；
- GPU 型號，以及該 process 是否與其他工作共用 GPU。

優先最佳化常見形狀，以及對 tail latency 影響重大的形狀。在 $4096^3$
表現最佳的 kernel，未必適合 $M$ 很小的 decode 階段 GEMM，也未必適合
含有許多不同形狀的 batch。

## 2. Persistent Kernel

一般 GEMM 會為每個輸出分塊啟動一個 block。**Persistent kernel** 只啟動
約等於可同時執行數量的 block，並讓每個 block 從工作佇列取得數個分塊：

```cpp
for (int tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
    const TileCoord coord = schedule(tile);
    gemmTile(coord);
}
```

這種設計可以：

- 讓每個常駐 block 處理數個分塊，消除 wave quantization；
- 在不同工作項目之間，讓 weight 或排程狀態保持在較快的記憶體中；
- 合併許多小型 GEMM 時，降低 launch 開銷；
- 支援[矩陣乘法 7](06-split-k-stream-k.md)所介紹的 Stream-K 工作分配。

Persistent 並不一定更快。長時間存活的 block 可能獨占 GPU、降低公平性，
也會放大不良分塊排程的代價。若 block 會互相等待，kernel 啟動的 block
數不得超過可同時駐留的數量。請用 occupancy API 取得上限，不要假設每個
SM 只能駐留一個 block。

## 3. Batched、Grouped 與 Expert GEMM

### 3.1 Batched GEMM

Strided batched GEMM 會對許多矩陣配對套用相同形狀與 layout：

$$
C_b = A_b B_b,\qquad b = 0,\ldots,B-1
$$

規則的 stride 可降低定址成本，也讓函式庫能為整個 batch 選擇同一個
kernel。Pointer-array batched GEMM 可支援彼此無關的 allocation，但會增加
指標載入。矩陣非常小時，讓一個 warp 或 block 負責一個矩陣，避免 launch
開銷成為主因。

### 3.2 Grouped GEMM

Grouped GEMM 接受一組 $M$、$N$、$K$、stride 或指標各不相同的問題。
Persistent scheduler 會把分塊映射到各個問題：

```text
problem 0: tiles [0, count0)
problem 1: tiles [count0, count0 + count1)
...
```

可用 prefix sum 找到每個問題。若問題清單很長，可建立直接的 tile map，
或採用兩層 lookup，避免每個分塊都掃描整份清單。請把資料型別、layout
與 epilogue 相容的問題分在一起；否則主迴圈和 epilogue 中的 branch 會
浪費發射槽。

### 3.3 Grouped Expert GEMM

Mixture-of-experts 推論會把不同數量的 token 分派給各 expert。若 expert
$e$ 收到 $M_e$ 個 token，其 projection 為

$$
Y_e = X_e W_e,\qquad
X_e \in \mathbb{R}^{M_e\times K},\quad
W_e \in \mathbb{R}^{K\times N}.
$$

$M_e$ 並不規則，也可能為零。Grouped expert GEMM 會把非空 expert
放進同一次 launch，並在所有 expert 之間排程分塊。良好的實作會：

- 不複製矩陣，直接使用 routing metadata；
- 在有助於提高分塊使用率時，把 $M_e$ 相近的 expert 分入同一 bucket；
- 略過空的 expert；
- 保留 token 至輸出的映射，供後續 scatter 使用；
- 盡可能融合每個 expert 的 bias、scale 或 quantization；
- 動態平衡 expert，避免單一大型 expert 造成很長的尾端。

請量測完整的 route → GEMM → scatter 路徑。若新增 metadata 轉換或額外
packing kernel，即使矩陣運算本身變快，整體仍可能更慢。

## 4. CUTLASS 與 CuTe

[CUTLASS](https://github.com/NVIDIA/cutlass) 提供經過調校的 GEMM building
block 與 device-level kernel。目前版本的 CUTLASS 使用 CuTe，以 layout
描述 tensor：一個 shape，加上一個從邏輯座標映射到儲存位置的方式。

本路線的概念可直接對應如下：

| 本路線 | CUTLASS/CuTe 概念 |
|---|---|
| Block、warp 與指令分塊 | Tiled MMA 與 collective mainloop |
| Global-to-shared pipeline | Collective copy、`cp.async` 或 TMA |
| 共享記憶體 XOR layout | Swizzled CuTe layout |
| 融合輸出迴圈 | Epilogue collective |
| 分塊啟動順序 | Scheduler 或 thread-block swizzle |

需要函式庫層級的控制、但不想手寫 PTX 時，可使用 CUTLASS template。
CuTe layout 讓位址映射明確且可組合，但其型別可能不易閱讀。請保留小型
參考實作，並在每次變更 layout 或 schedule 時測試邊界形狀。

## 5. Epilogue 融合

歸約結束時，累加器已在暫存器中。請在儲存前套用縮放、bias、residual、
activation、clamping 或輸出轉換：

```cpp
float x = alpha * acc + beta * old_c;
x += bias[col];
x = gelu(x);
out[row * ld + col] = convert<Output>(x);
```

融合可移除中間 launch，也能避免再讀寫一次輸出。GEMM 較小或窄長時，
效益最明顯。

不要寫出一個充滿 runtime branch 的萬用 epilogue。請編譯一小組常見組合，
再透過 dispatch 選擇。除非 profile 顯示另一個 specialization 的效益足以
抵銷程式碼大小與維護成本，否則不常見的操作鏈應放在另一個 kernel。

Tensor-core 累加器 layout 可能與所需的儲存 layout 不一致。請依
coalescing 與暫存器壓力，在直接從暫存器儲存、透過共享記憶體交換，
或函式庫 epilogue 之間做選擇。

## 6. 混合精度與準確度

輸入型別、乘法型別、累加器型別與輸出型別是各自獨立的選擇。常見策略
包括：FP16 或 BF16 輸入搭配 FP32 累加、TF32 乘法搭配 FP32 累加，
以及經縮放的 FP8 輸入搭配 FP32 或 FP16 累加。

不要只檢查一組隨機矩陣：

- 與較高精度的 CPU 或 GPU 參考結果比較；
- 若 API 定義了相關行為，納入零、subnormal、大數值、NaN 與 infinity；
- 測試累加誤差較大的長 $K$；
- 測試 quantized 路徑實際使用的 scale 與 rounding 規則；
- 測試拆分歸約，因為重新結合會改變 rounding；
- 同時使用 absolute error 與 relative error，並依資料型別和 $K$ 設定
  tolerance。

一個實用的 normalized residual 是

$$
r = \frac{\lVert C_{\text{test}}-C_{\text{ref}}\rVert_F}
         {\lVert A\rVert_F\lVert B\rVert_F+\epsilon}.
$$

Determinism 與準確度是不同需求。Atomic split-K 可能在 tolerance 內保持
準確，但每次執行的低位元不同。呼叫端需要可重複結果時，請提供具
determinism 的 workspace reduction。

## 7. Autotuning 與形狀 Bucket

沒有任何一種分塊形狀能在所有情況勝出。候選參數包括：

- block、warp 與指令分塊形狀；
- pipeline stage 數與共享記憶體 layout；
- split-K factor 或 Stream-K schedule；
- persistent grid 大小與分塊順序；
- vector width、輸入型別與 epilogue specialization。

搜尋範圍必須有限。編譯或執行前，先排除超出暫存器、共享記憶體、對齊或
架構限制的候選項目。先驗證正確性，再暖機 GPU，最後比較數次 timing
sample。

Production dispatch 不應為每個精確形狀儲存一筆表格。請依工作負載行為
定義**形狀 bucket**，例如：

- $M$ 很小的 decode 形狀；
- 又高又窄或又矮又寬的矩形；
- 大型、接近正方形的形狀；
- 對齊與未對齊的 $K$；
- 常見的 expert token 數範圍。

對代表性形狀進行調校，再驗證 bucket 邊界。對調校集合以外的形狀，請提供
可靠 fallback。結果應依 GPU 架構、軟體版本、資料型別、layout 與 epilogue
建立 cache key；某款 GPU 的結果不是可攜的定論。

## 8. 函式庫還是自訂 Kernel？

先從 cuBLAS、cuBLASLt、CUTLASS 或其他持續維護的廠商函式庫開始。函式庫
通常提供廣泛的形狀支援、架構 dispatch、workspace 選擇、融合 epilogue，
以及多年累積的正確性調校。

只有量測顯示下列一項或多項因素能帶來明顯的 end-to-end 效益時，才值得
使用自訂 kernel：

- 不尋常的固定形狀或 layout；
- 函式庫無法表達的融合；
- 與應用程式 metadata 緊密結合的 grouped expert schedule；
- 嚴格的 workspace、determinism 或 latency 限制；
- 函式庫尚未提供的新架構功能。

請保留函式庫作為 fallback 與比較 baseline。自訂路徑必須承擔跨形狀、
device、driver 與未來架構的測試成本。只在單一 benchmark 達到高峰值
TFLOP/s 還不夠。

## 9. Benchmark 與 Profiling

請 benchmark 應用程式真正執行的操作：

1. 在計時區段外配置並初始化資料。
2. 暖機，直到時脈、cache 與函式庫的 lazy initialization 穩定。
3. 在相同 stream 上以 GPU event 計時；回報 median 與 tail latency。
4. 執行足夠工作，準確量測小型 kernel，但要保留實際 launch dependency。
5. 依應用程式情境 flush 或保留 cache，並清楚註明採用哪一種。
6. 比較完全相同的運算、型別、融合與 determinism 策略。
7. 計時後驗證輸出。

實用 throughput 可回報為

$$
\text{TFLOP/s} = \frac{2MNK}{t\cdot10^{12}},
$$

但也應回報 latency 與 end-to-end 時間。Grouped GEMM 則需加總所有問題的
$2M_iN_iK_i$。

使用 Nsight Systems 找出 launch gap、同步與重疊情況。使用 Nsight Compute
檢查實際 occupancy、tensor-core utilization、全域與共享記憶體 throughput、
L2 hit rate、bank conflict、stall 與暫存器 spill。只 profile 一小組具
代表性的情況：詳細 profiling 會改變 timing，也可能讓 kernel 依序執行。

請用前面章節的模型解讀 counter。Tensor-core utilization 低，可能來自
分塊太少、pipeline stall、昂貴的 epilogue 或 load imbalance；單看這個
數值無法判斷修正方式。

## 10. Production 檢查清單

- [ ] API 已定義 shape、layout、stride、alias、資料型別與支援的 epilogue。
- [ ] 空矩陣、邊界分塊、奇數 leading dimension 與大型索引都能正確處理。
- [ ] 每條向量化路徑都會檢查對齊，並提供安全 fallback。
- [ ] 使用 FP32 累加，或其他已記錄的精度策略。
- [ ] 測試涵蓋對抗性數值、長 $K$、拆分歸約與每種融合 epilogue。
- [ ] 明確區分 deterministic 與 non-deterministic 模式。
- [ ] Dispatcher 具有函式庫或通用 kernel fallback。
- [ ] 調校資料以 GPU 架構與軟體版本為 key。
- [ ] 已記錄 workspace 大小、初始化方式與 stream ownership。
- [ ] Persistent 與跨 block protocol 不會 deadlock。
- [ ] Benchmark 涵蓋 production 形狀、cold 與 warm cache 行為、
      median latency、tail latency 與 end-to-end 影響。
- [ ] Profiling 顯示沒有意外的 spill、bank conflict 或 serialization。
- [ ] 函式庫與自訂 baseline 執行完全相同的操作。
- [ ] 不支援的 device 會明確失敗，而不是默默使用錯誤的指令路徑。

## 重點整理

1. Production GEMM 是理解工作負載的 dispatcher，不是單一萬用 kernel。
2. Persistent 與 grouped scheduling 可提高不規則問題或大量小型問題的
   使用率，但必須仔細平衡負載。
3. CUTLASS 與 CuTe 封裝了本路線使用的相同分塊、pipeline 與 layout 概念。
4. 融合、混合精度策略與形狀 bucket 調校必須一起驗證。
5. 在 end-to-end benchmark 證明自訂路徑值得維護前，優先使用持續維護的
   函式庫。

## 練習

1. 為包含 prefill GEMM、decode GEMM 與 64 個 expert 的工作負載設計形狀
   bucket。列出代表性形狀與 fallback。
2. 在 tensor-core epilogue 加入 bias 與 GELU。分別對大型方陣 GEMM 與
   小型 $M$ GEMM，比較融合與分開的 kernel。
3. 為十個分塊數不同的問題建立 grouped scheduler。比較 prefix-sum lookup
   與直接 tile-to-problem map。
4. 針對三種形狀調校 block shape、stage 數與 split-K factor。記錄正確性、
   暫存器、共享記憶體、median latency 與勝出的設定。
5. 使用相同資料型別與 epilogue，比較自訂 kernel 與 cuBLASLt 或 CUTLASS。
   判斷效益來自矩陣運算、融合，還是排程。
