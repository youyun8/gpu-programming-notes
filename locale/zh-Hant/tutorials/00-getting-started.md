# 00 – 開始使用

> **第一部分 · CUDA 基礎** · 先備知識：C++（指標、樣板）·
> 下一章：[01 – 執行模型](01-execution-model.md)

本章會建立後續章節和實作頁面預設你已具備的所有基礎。讀完後，即使你沒有自己的 GPU，也能編譯、執行、檢查、計時並評估 CUDA kernel。

**你將學到**

- 在哪裡執行 CUDA 程式碼，以及如何在 CPU 上測試；
- `nvcc` 會從 `.cu` 檔案產生什麼（PTX、SASS、fatbin），以及這為何會影響相容性；
- GPU 程式的完整生命週期：配置、複製、啟動、同步、複製回來、釋放；
- LeetGPU 或 Tensara 提交內容的形式；
- 非同步 API 如何呈現錯誤，以及如何捕捉錯誤；
- 如何正確計算 kernel 執行時間，並把時間換算成頻寬和 FLOP/s；
- 屋頂線模型，也就是後續每章使用的評估基準。

## 1. 在哪裡執行程式碼

### 1.1 選項

| 選項 | 說明 |
|--------|-------|
| [LeetGPU](https://leetgpu.com) playground | 免費。可在瀏覽器中執行 CUDA、Triton 和 PyTorch，並提供 GPU 模擬器模式。 |
| [Tensara](https://tensara.org) | 會在真實 GPU（T4、A100、H100……）上評測提交內容。 |
| Google Colab / Kaggle | 免費 T4 GPU；在 notebook 儲存格中使用 `!nvcc`。 |
| 雲端 VM（Lambda、RunPod……） | 若要完整使用 Nsight 分析功能，就需要這個選項。 |
| 此 repo 的 [cuemu](../tools/cuemu/README.md) | 在你的 **CPU** 上以官方參考測試執行任何解答。它只檢查正確性，不檢查速度。 |

### 1.2 合理的開發流程

1. 在本機以支援 C++ 的編輯器撰寫 kernel。
2. 使用 cuemu 檢查正確性（幾秒即可，不需要 GPU）。
3. 提交，或在 Colab 上執行以計時。
4. 只有當時間遠離第 7 節的界限時，才進行分析（NVIDIA 使用 Nsight Compute，AMD 使用 `rocprofv3`）來找出原因。

先確保正確，再追求速度：無法測試的最佳化不值得信任。

## 2. 工具鏈

```bash
nvcc --version                         # CUDA toolkit
nvidia-smi                             # driver and GPU
nvcc -O3 -arch=native -o hello hello.cu && ./hello
```

### 2.1 `nvcc` 會產生什麼

`.cu` 檔案混合了供兩種處理器使用的程式碼。`nvcc` 會將它們分開：

![nvcc 將 .cu 檔案拆成主機端程式碼和裝置端程式碼；裝置端程式碼會變成 PTX 和 SASS，兩者都嵌入 fatbin](figures/ch00-nvcc.svg)

- **主機端程式碼**（所有未標記為 `__global__` 或 `__device__` 的內容）會交給一般 C++ 編譯器（`g++`、`clang` 或 MSVC）。
- **裝置端程式碼**會先編譯成 **PTX**，也就是理想化 GPU 的虛擬指令集，再由 `ptxas` 編譯成 **SASS**，也就是真實架構的機器碼。
- 兩者都會嵌入執行檔內的 **fatbin**。執行時，驅動程式會選擇符合 GPU 的 SASS；若沒有，就即時編譯 PTX。

### 2.2 架構與相容性

每款 NVIDIA GPU 都有一個*運算能力* `X.Y`，寫成 `sm_XY`：

| GPU | 運算能力 | 重要功能 |
|---|---|---|
| T4 | sm_75（Turing） | FP16 tensor core（`mma.m16n8k8`） |
| A100 | sm_80（Ampere） | `cp.async`、BF16/TF32 tensor core、每個 SM 164 KB 共享記憶體 |
| RTX 30xx / 40xx | sm_86 / sm_89 | 消費級 Ampere / Ada |
| H100 | sm_90（Hopper） | TMA、`wgmma`、執行緒區塊叢集 |
| B200 | sm_100（Blackwell） | Tensor 記憶體、`tcgen05` 指令 |

PTX/SASS 分工帶來以下規則：

- **SASS 是精確對應的。** `sm_80` 的 SASS 可在運算能力 8.x 且 x ≥ 0 的裝置上執行（同一主版本內具有二進位相容性），但不能在 7.5 或 9.0 上執行。
- **PTX 向前相容。** `compute_80` 的 PTX 可即時編譯給任何後續 GPU，但第一次啟動時會有編譯延遲，也無法使用較新的功能。
- `-arch=sm_80` 是「`sm_80` 的 SASS 加上 `compute_80` 的 PTX」的簡寫。`-gencode arch=compute_90,code=sm_90` 明確寫出兩部分；重複使用它即可建置支援多款 GPU 的 fatbin。
- `wgmma` 等功能只存在於「架構特定」目標 `sm_90a`，其程式碼無法在其他 GPU 上執行。

### 2.3 實用旗標

| 旗標 | 效果 |
|------|--------|
| `-O3` | 主機端最佳化。裝置端程式碼預設會最佳化。 |
| `-lineinfo` | 在分析器中顯示原始碼行號，且不會降低速度。 |
| `-G` | 裝置端除錯建置。非常慢；只搭配 `cuda-gdb` 使用。 |
| `--use_fast_math` | 使用近似 `expf`、`sinf`、除法，並將非正規數歸零。通常無法通過嚴格誤差限制。 |
| `-Xptxas -v` | 印出每個 kernel 的暫存器、共享記憶體和 spill 資訊。 |
| `-std=c++17` | 在裝置端程式碼中使用現代 C++（樣板、`constexpr`、lambda）。 |
| `-keep` | 保留包括 `.ptx` 在內的中間檔案。 |

### 2.4 查看編譯器做了什麼

若要回答「這個迴圈有沒有展開？」或「這次載入有沒有變成 128 位元寬？」等問題，編譯器輸出才是最可靠的依據：

```bash
nvcc -O3 -arch=sm_80 -Xptxas -v -c kernel.cu      # registers, spills ("bytes stack frame")
cuobjdump -ptx kernel.o | less                    # the PTX
cuobjdump -sass kernel.o | grep -E "LDG|STG|FFMA" # the machine instructions
```

值得在 SASS 中辨識的項目包括：`LDG.E.128` / `STG.E.128`（16 位元組全域存取）、`LDS` / `STS`（共享記憶體）、`FFMA`（FP32 融合乘加）、`HMMA`（tensor core）、`BAR.SYNC`（`__syncthreads()`），以及 `LDL` / `STL`（區域記憶體：通常代表暫存器 spill，應盡量避免）。

## 3. 第一個完整程式

### 3.1 兩套記憶體

CPU（主機）和 GPU（裝置）各自擁有獨立記憶體。`cudaMalloc` 傳回的指標是裝置位址：主機不可對它解參照，資料只能透過明確複製在兩者間移動（或使用第 3.4 節的受控記憶體）。

### 3.2 生命週期

```cpp
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void scaleKernel(const float* in, float* out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // one element per thread
    if (idx < n) out[idx] = 2.0f * in[idx];                  // the last block is partial
}

int main() {
    const int n = 1 << 20;
    std::vector<float> h_in(n, 1.0f), h_out(n);

    float *d_in = nullptr, *d_out = nullptr;                 // 1. allocate on the device
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);   // 2. copy in

    constexpr int kBlockSize = 256;                          // 3. launch
    const int num_blocks = (n + kBlockSize - 1) / kBlockSize;
    scaleKernel<<<num_blocks, kBlockSize>>>(d_in, d_out, n);

    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);  // 4. copy out (waits)
    std::printf("out[0] = %f\n", h_out[0]);

    cudaFree(d_in);                                          // 5. free
    cudaFree(d_out);
}
```

各步驟及實際發生的事情如下：

1. **`cudaMalloc`** 保留裝置記憶體。配置速度很慢（可能需要數毫秒），因此實際程式會配置一次並重複使用緩衝區。
2. **`cudaMemcpy` 主機 → 裝置**會跨越 PCIe 或 NVLink，遠慢於 GPU 自身的記憶體：PCIe 4.0 x16 連線約可傳輸 25 GB/s，A100 的 HBM 則約為 1500 GB/s。對小型 GPU 工作而言，搬移資料往往才是真正的成本。
3. **啟動** `kernel<<<grid, block>>>(args)` 只會將工作*排入佇列*並立即返回。一般形式為 `<<<grid, block, dynamic_shared_bytes, stream>>>`。
4. **`cudaMemcpy` 裝置 → 主機**會等待預設 stream 上較早的所有工作，因此返回時可保證 kernel 已執行完畢。
5. **`cudaFree`** 釋放記憶體（也會同步）。

### 3.3 啟動設定

`grid` 和 `block` 都是 `dim3` 值（可為一維、二維或三維）。執行緒總數是各維度分量的乘積：

$$
N_{\text{threads}} = (G_x G_y G_z)\,(B_x B_y B_z), \qquad
B_x B_y B_z \le 1024, \qquad
G_x \le 2^{31} - 1, \quad G_y, G_z \le 65535
$$

| 符號 | 意義 |
|---|---|
| $G_x, G_y, G_z$ | 網格維度：各軸上的區塊數 |
| $B_x, B_y, B_z$ | 區塊維度：各軸上每個區塊的執行緒數 |
| $N_{\text{threads}}$ | 啟動的執行緒總數 |

第 01 章會說明如何選擇這些數字。

### 3.4 受控記憶體

`cudaMallocManaged` 傳回的指標在主機端與裝置端都有效；頁面會依需求遷移。這很適合製作原型，但分頁錯誤會讓第一次存取的計時產生誤導。實作平台會提供裝置指標，因此其餘內容都使用明確的裝置記憶體。

## 4. 提交內容的結構

兩個平台都會提供**裝置指標**，並呼叫一個 C linkage 進入點。你需要撰寫 kernel 並啟動它。

```cpp
#include <cuda_runtime.h>

__global__ void scaleKernel(const float* in, float* out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 2.0f * in[idx];
}

// LeetGPU entry point
extern "C" void solve(const float* in, float* out, int n) {
    constexpr int kBlockSize = 256;
    scaleKernel<<<(n + kBlockSize - 1) / kBlockSize, kBlockSize>>>(in, out, n);
    cudaDeviceSynchronize();
}

// Tensara entry point
extern "C" void solution(const float* in, float* out, size_t n) { /* ... */ }
```

| | LeetGPU | Tensara |
|---|---|---|
| 進入點 | `solve(...)` | `solution(...)` |
| 大小 | 通常為 `int` | 通常為 `size_t` |
| 計時 | `solve` 的實際經過時間 | `solution` 的 GPU 時間，多次執行後取平均 |
| 容許誤差 | 依題目而定 | 依題目而定（`rtol`、`atol`）；大型歸約通常較寬鬆 |
| 硬體 | 可選擇 GPU 或模擬器 | T4、A100、H100、L40S…… |

請一律從起始程式碼複製確切的函式簽章：各題目的參數順序和整數型別都可能不同。

這些指標都是裝置指標。在主機端對它們解參照會導致程式崩潰。部分 Tensara 題目會傳入可能位於任一端的 `shape` 陣列；`cudaMemcpy(..., cudaMemcpyDefault)` 在兩種情況下都能正確複製它（請參閱 [Tensara – Argmax](../tensara/argmax/)）。

## 5. 開發時檢查錯誤

### 5.1 兩類錯誤

因為啟動是非同步的，所以錯誤分成兩類，並在不同時機回報：

| 類型 | 範例 | 回報位置 |
|---|---|---|
| 啟動錯誤 | 每區塊執行緒過多、共享記憶體過多、網格無效 | 啟動後立刻呼叫的 `cudaGetLastError()` |
| 執行錯誤 | 越界存取、位址未對齊、`__trap()` | *下一個同步呼叫*（`cudaDeviceSynchronize`、`cudaMemcpy`……） |

執行錯誤具有**黏著性**：它會破壞 CUDA context，使處理程序中之後的每個 API 呼叫都傳回相同錯誤。唯一復原方式是重新啟動處理程序。這就是錯誤有時看似發生在距離肇因 kernel 很遠的 `cudaMemcpy` 中的原因。

### 5.2 檢查巨集

```cpp
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

myKernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());        // launch configuration errors
CUDA_CHECK(cudaDeviceSynchronize());   // errors raised while running
```

每次啟動後呼叫 `cudaDeviceSynchronize()` 會讓程式序列化，因此請在開發時使用，計時時移除（或放在除錯旗標後）。

### 5.3 compute-sanitizer

`compute-sanitizer ./app` 會以插樁方式執行程式，並回報第一個錯誤存取所屬的執行緒、區塊和原始碼行（以 `-lineinfo` 建置）：

| 工具 | 可找出的問題 |
|---|---|
| `--tool memcheck`（預設） | 越界及未對齊的全域／共享記憶體存取、洩漏 |
| `--tool racecheck` | 共享記憶體資料競爭（缺少 `__syncthreads()`） |
| `--tool initcheck` | 讀取未初始化的全域記憶體 |
| `--tool synccheck` | 區塊內並非所有執行緒都抵達的 barrier |

它會讓程式慢 10–100 倍，因此請以小型輸入執行。

## 6. 計算 kernel 執行時間

### 6.1 CUDA Event

請使用 CUDA event。它們記錄在 GPU 的 stream 上，因此測量的是 GPU 時間，而非啟動開銷：

```cpp
cudaEvent_t start_event, stop_event;
cudaEventCreate(&start_event);
cudaEventCreate(&stop_event);

scaleKernel<<<grid, block>>>(d_in, d_out, n);          // warm-up
cudaEventRecord(start_event);
for (int rep = 0; rep < kReps; ++rep) scaleKernel<<<grid, block>>>(d_in, d_out, n);
cudaEventRecord(stop_event);
cudaEventSynchronize(stop_event);

float elapsed_ms = 0.0f;
cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
const double seconds_per_call = elapsed_ms * 1e-3 / kReps;
```

### 6.2 常見問題

- **沒有暖機。** 第一次啟動包含模組載入，可能還包含 PTX 即時編譯。務必捨棄第一次結果。
- **主機端計時器沒有同步。** 用 `std::chrono` 包住啟動只會測到排入佇列的時間（幾微秒）。使用主機端計時器時，停止計時前必須同步。
- **重複次數太少。** 10 µs 的 kernel 會被啟動開銷和計時器解析度主導；請取多次呼叫的平均值。
- **快取已暖機。** 在同一小型輸入上重複執行 kernel 會讓資料留在 L2（資料中心 GPU 約有 40–50 MB），使「受 DRAM 限制」的 kernel 看起來比實際 pipeline 中更快。若要測量 DRAM 數據，請使用比 L2 大的輸入，或在每次執行間清除快取。
- **時脈加速和降頻。** GPU 會隨溫度和功率調整時脈。評測平台會鎖定時脈；在自己的機器上，請比較相同條件下的執行結果。

### 6.3 從時間換算速率

單獨的時間資訊很有限。請將它換算成速率，再與硬體限制比較：

$$
\beta_{\text{eff}} = \frac{Q}{t}, \qquad F_{\text{eff}} = \frac{W}{t}, \qquad
I = \frac{W}{Q}
$$

| 符號 | 意義 |
|---|---|
| $t$ | 每次呼叫的實測時間（秒） |
| $Q$ | kernel *必須*在 DRAM 之間搬移的位元組數（輸入讀一次、輸出寫一次） |
| $W$ | 有效浮點運算次數（一次 FMA 計為 2 次） |
| $\beta_{\text{eff}}$ | 有效頻寬，位元組/秒 |
| $F_{\text{eff}}$ | 實際達成的吞吐量，flop/s |
| $I$ | 算術強度，每位元組的浮點運算次數 |

## 7. 屋頂線模型

### 7.1 界限

kernel 不可能比以峰值速率完成算術運算所需的時間更快，也不可能比以峰值頻寬搬移必要位元組所需的時間更快。兩者較大者就是**屋頂線界限**：

$$
t \ \ge\ T_{\min} = \max\left(\frac{W}{F},\ \frac{Q}{\beta}\right), \qquad
F_{\text{eff}} \le \min(F,\ I\beta), \qquad
I^{\star} = \frac{F}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $F$ | GPU 對所用資料型別的峰值計算吞吐量 |
| $\beta$ | DRAM 峰值頻寬 |
| $T_{\min}$ | kernel 時間的下限 |
| $I^{\star}$ | 轉折點：$I < I^{\star}$ 的 kernel 受記憶體限制，$I > I^{\star}$ 的 kernel 受計算限制 |

在雙對數座標上，可達到的 FLOP/s 會形成屋頂形狀：低強度時斜率為 $\beta$，超過轉折點後則平坦於 $F$。

![A100 的屋頂線：低於轉折點時，kernel 受頻寬限制；高於轉折點時，則受計算限制。各點標示每個 kernel 的強度所決定的界限](figures/ch00-roofline.svg)

### 7.2 常見 GPU 的數據

| GPU | FP32 $F$ | $\beta$ | $I^{\star}$（FP32） |
|---|---|---|---|
| T4 | ~8 TFLOP/s | ~320 GB/s | ~25 flop/B |
| A100 40 GB | ~19.5 TFLOP/s | ~1.55 TB/s | ~13 flop/B |
| H100 SXM | ~67 TFLOP/s | ~3.35 TB/s | ~20 flop/B |

使用 tensor core 時，計算屋頂會提高 8–16 倍（A100 的密集 FP16 為 312 TFLOP/s），因此轉折點會移到約 200 flop/B：除了大型矩陣乘法之外，幾乎所有工作都受記憶體限制。

### 7.3 計算範例

| Kernel | $W$ | $Q$ | $I$（flop/B） | A100 上的界限 |
|---|---|---|---|---|
| $y = 2x$，$n$ 個 float | $n$ | $8n$ | 1/8 | 記憶體：$8n / \beta$ |
| $c = a + b$ | $n$ | $12n$ | 1/12 | 記憶體：$12n / \beta$ |
| $n$ 個 float 的總和 | $n$ | $4n$ | 1/4 | 記憶體：$4n / \beta$ |
| $C = AB$，$n\times n$ | $2n^3$ | $\ge 12n^2$ | 最高 $n/6$ | 大型 $n$ 時受計算限制 |

對 $n = 2^{26}$ 的 $y = 2x$ 而言：$Q = 537$ MB，所以 $T_{\min} = 0.35$ ms。實測 0.40 ms 代表已達峰值頻寬的 87 %：可以收工了。本站每個題目頁面的*成本分析*小節都會進行這類計算。

### 7.4 屋頂線無法告訴你的事

- **它只計算必要流量。** 存取未合併（第 02 章）的 kernel 會搬移比 $Q$ 更多的位元組，因此低於屋頂線。
- **它忽略延遲。** 平行度太低的 kernel 無法讓足夠多的要求同時進行，因此達不到 $\beta$（第 01 章第 5 節）。
- **還有其他屋頂。** L2 和共享記憶體各自都有頻寬；kernel 可能受其中一項限制（矩陣乘法 1 主要就在討論共享記憶體屋頂）。
- **峰值數字只是峰值。** 實務上限約為規格表頻寬的 90 %；若指令混合不全是 FMA，FLOP/s 的比例還會更低。

## 8. 使用 cuemu 在 CPU 上測試

[cuemu](../tools/cuemu/README.md) 會將解答轉成一般 C++，以 `clang++` 建置，並把每個 CUDA 執行緒當成使用者空間 fiber 執行，一次執行一個執行緒區塊。`__syncthreads()` 是真正的 barrier，warp shuffle 也確實會在 32 個 lane 間交換值。接著以平台本身的 PyTorch 參考實作和平台容許誤差，比較縮小版官方測試案例的輸出：

```bash
scripts/fetch_upstream.sh                              # problem definitions -> .upstream/
python3 tools/cuemu/run_tests.py leetgpu/001-vector-add
python3 tools/cuemu/run_tests.py tensara/softmax
python3 tools/cuemu/run_tests.py --reverse leetgpu/004-reduction   # reversed thread order
python3 tools/cuemu/cuemu.py run tutorials/gemm/01-vectorized.cu -- --test   # a program with main()
```

它能捕捉索引錯誤、缺少邊界檢查、越界寫入（每個緩衝區後方都有 guard page）、布局錯誤和數值錯誤。`--reverse` 會以相反順序排程執行緒，可揭露許多缺少 barrier 的問題。它無法判斷速度，而且區塊會逐一執行，因此無法找出區塊*之間*的競爭；這類問題請在真實 GPU 上使用 `compute-sanitizer`。

## 9. 提交前檢查清單

- [ ] 已精確複製函式簽章（順序、`int` 與 `size_t`、`const`）。
- [ ] 每個執行緒都會檢查索引（`if (idx < n)`），包括最後一個不完整區塊。
- [ ] 當 $n$ 或位元組位移可能超過 $2^{31}$ 時使用 64 位元索引。
- [ ] 先將要累加寫入（atomic）的緩衝區歸零。
- [ ] 釋放暫時配置的 `cudaMalloc`；若 kernel 可能仍在使用記憶體，先呼叫 `cudaDeviceSynchronize()` 再呼叫 `cudaFree`。
- [ ] 成本分析已說明界限，且實測時間接近該界限。

## 重點整理

1. `nvcc` 會產生 PTX（可攜）和 SASS（精確對應）；請提供目標 GPU 的 SASS，並提供 PTX 以支援未來 GPU。
2. 啟動是非同步的：必須在同步點後讀取錯誤和計時結果，而且執行錯誤具有黏著性。
3. 暖機後使用 event 計時，並重複多次。
4. 務必將時間與屋頂線界限 $\max(W/F,\ Q/\beta)$ 比較：它能告訴你 kernel 是否已最佳化完成；若尚未完成，也能指出該改善哪項資源。

## 練習

1. 使用 `-arch=sm_80` 編譯第 3.2 節的程式，並以 `cuobjdump --list-elf --list-ptx` 列出嵌入的映像。接著也為 sm_75 和 sm_90 建置 fatbin。

    <details markdown="1"><summary>提示</summary>

    `nvcc -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=[sm_90,compute_90]`
    會嵌入三個 SASS 映像和 `compute_90` PTX。

    </details>

2. 以每區塊 2048 個執行緒啟動 `scaleKernel`，並印出 `cudaGetLastError()` 和 `cudaDeviceSynchronize()` 的傳回值。

    <details markdown="1"><summary>答案</summary>

    啟動會失敗並傳回 `cudaErrorInvalidConfiguration`（「invalid configuration argument」），由 `cudaGetLastError()` 回報。kernel 從未執行，因此同步會傳回 `cudaSuccess`；啟動錯誤不具黏著性。

    </details>

3. 在 H100 SXM 上將兩個各含 $10^8$ 個 float 的向量相加，$T_{\min}$ 是多少？若耗時 0.45 ms，實測頻寬是多少？

    <details markdown="1"><summary>答案</summary>

    $Q = 12 \cdot 10^8$ B = 1.2 GB，因此 $T_{\min} = 1.2 / 3350$ s ≈ 0.36 ms。
    0.45 ms 相當於 $1.2\text{ GB} / 0.45\text{ ms} \approx 2.7$ TB/s，即峰值的 80 %。

    </details>

4. 不同步就以 `std::chrono` 計算 kernel 執行時間，並解釋結果。

## 實作練習

- [LeetGPU – 向量加法](../leetgpu/001-vector-add/)、
  [Tensara – 向量加法](../tensara/vector-addition/)
- [LeetGPU – 色彩反轉](../leetgpu/007-color-inversion/)、
  [Tensara – ReLU](../tensara/relu/)
