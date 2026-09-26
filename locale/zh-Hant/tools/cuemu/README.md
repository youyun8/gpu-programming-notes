# cuemu – 在 CPU 上執行 CUDA 解答

`cuemu` 讓這個儲存庫中的每個解答都能在沒有 NVIDIA GPU 的情況下測試。CI 就是用它，將全部 185 個解答與各平台自己的參考實作比對。

## 運作方式

1. **轉譯**（`cuemu.py`）。透過少量原始碼層級的改寫，將 `.cu` 檔轉成 C++：
   - 移除 CUDA 標頭；內嵌本機標頭（`#include "x.cuh"`），因此這些標頭也會一起轉譯。
   - `extern __shared__ T x[]` 變成 `T* x = cuemu::dynamicSmem<T>()`。
   - `kernel<<<grid, block, smem>>>(args)` 變成
     `cuemu::Launcher(grid, block, smem)(kernel, args)`。

   其餘所有內容，包括 `__global__`、`threadIdx`、`__shared__`、原子操作、`half`/`bfloat16`、向量型別與 WMMA，都由 `cuemu.h` 提供。
2. **建置。** 結果會以 `clang++ -std=c++20 -O2 -shared` 建成共享函式庫，並依內容雜湊快取（對於自帶 `main()` 的程式，`cuemu.py run` 會改為建置可執行檔）。
3. **執行。** 每個 CUDA 執行緒都以使用者空間 fiber 執行，而 block 逐一執行。這能提供真正的語意：
   - `__syncthreads()` 是真正的 barrier。分歧 barrier 會回報為 deadlock。
   - Warp 操作（`__shfl*_sync`、`__ballot_sync`、`__match_any_sync`、`__reduce_*_sync`）會讓一個 warp 的 32 個 lane 會合。
   - WMMA fragment 會檢查真實硬體可能默默造成效能損失的對齊規則（32-byte 指標、`ldm·sizeof(T) % 16 == 0`）。
   - 透過 `<cuda_pipeline.h>` 使用的 `cp.async`（`__pipeline_memcpy_async`、`__pipeline_commit`、`__pipeline_wait_prior`）會以**延後**複製模擬：資料只會在 wait 涵蓋其群組時到位，因此過早讀取 pipeline stage 會像在 GPU 上一樣失敗。
   - 使用 inline PTX `ldmatrix` / `mma.sync.m16n8k16` 的 kernel，可在 `#ifdef __CUEMU__` 下呼叫 `cuemuLdmatrix` 與 `cuemuMmaM16N8K16`；它們實作了 PTX 文件所定義的 fragment layout（請見 [tutorials/gemm/09-mma-sync.cu](../../tutorials/gemm/09-mma-sync.cu)）。
4. **檢查**（`run_tests.py`）。共享函式庫會由 `ctypes` 載入，並執行上游測試案例。輸出會使用上游容許誤差，與上游 PyTorch 參考結果比較。
   - 每個 buffer 後方的 **guard page** 會攔截越界寫入。
   - 會檢查輸入是否遭到**修改**。
   - 每個問題都在自己的 process 中執行，並設有 timeout。
   - `--reverse` 會以反向順序排程執行緒，藉此揭露缺少的 barrier。
   - Tensara 的 benchmark 尺寸太大，無法模擬，因此 `tensara_small_cases.py` 會從相同 generator 衍生縮小版與奇數尺寸的變體。
   - `lowp_reference.py` 為只能在 GPU 上執行的 FP4/FP8 參考實作提供 CPU 替代版本：flashinfer NVFP4，以及使用 swizzled scale 的 `scaled_mm`。

## 使用方式

```bash
scripts/fetch_upstream.sh                                # 將問題定義複製到 .upstream/
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-test.txt

python3 tools/cuemu/run_tests.py leetgpu/001-vector-add  # 一個問題
python3 tools/cuemu/run_tests.py --all -j 8              # 全部
python3 tools/cuemu/run_tests.py --platform tensara --all --json build/tensara.json
python3 tools/cuemu/run_tests.py --reverse leetgpu/004-reduction   # 偵測 race
python3 tools/cuemu/cuemu.py translate leetgpu/022-gemm/solution.cu  # 查看轉譯結果
python3 tools/cuemu/cuemu.py run tutorials/gemm/03-cp-async.cu -- --test  # 自帶 main() 的程式
```

| 環境變數 | 預設值 | 意義 |
|----------------------|---------|---------|
| `CUEMU_MAX_ELEMENTS` | 2²² | 跳過 tensor 更大的測試案例（README 可用 `cuemu_max_elements:` 提高上限） |
| `CUEMU_TIMEOUT` | 300 | 每個問題的 timeout，單位為秒 |
| `CUEMU_CXX` | `clang++` | 編譯器 |

## 限制

- **只檢查正確性。** CPU 上的計時沒有意義。
- **不支援：** CUB、Thrust、cuBLAS/cuDNN，以及 inline PTX（上方的 `__CUEMU__` hook 除外）。這裡的解答刻意避開它們。
- **Cooperative groups：** 模擬了其中一部分：`this_thread_block()`、含 shuffle、vote 與 `sync()` 的 `tiled_partition<N>()`，以及搭配 `cg::plus`、`cg::less` 和 `cg::greater` 的 `cg::reduce` / `cg::inclusive_scan`。不支援 grid group 與 cluster group。
- **排程：** 執行緒只會在 barrier 與 warp collective 處交錯執行。沒有 barrier 的 data race 可能在這裡通過，卻在 GPU 上失敗；`--reverse` 能抓到最常見的一類。
- **Top-p sampling** 無法重現 `torch.multinomial` 的 RNG。對此，runner 會檢查每個抽樣 token 是否都位於 nucleus 內。
