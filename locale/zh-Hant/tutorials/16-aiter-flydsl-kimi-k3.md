# 16 – AMD GPU 上用於 Kimi K3 的 AITER 與 FlyDSL

> **第五部 · AMD Production Kernel** · 先備知識：
> [05 – CDNA3 與 MFMA](05-amd-cdna3-mfma.md)、
> [06 – AITER Assembly GEMM](06-aiter-asm-gemm.md)，以及
> [15 – 模型系統中的 Triton](15-triton-model-systems.md) ·
> 下一章：[08 – 部署本站](08-deploying-this-site.md)

本章將 AMD kernel stack 連結到完整模型。Kimi K3 是很好的案例，因為它結合低精度權重、sparse expert、Kimi Delta Attention（KDA）、gated MLA 與新的 activation。

各工具分工不同：

```text
Quark checkpoint conversion
    → SGLang, vLLM, or ATOM runtime
        → AITER operator dispatch
            → FlyDSL, Triton/Gluon, CK, HIP, assembly, or Opus kernel
```

Quark 改變模型資料；AITER 選擇並包裝 operator；FlyDSL 撰寫與編譯 kernel。三者都不是完整的 serving runtime。

**你將學會**

- 從 vector addition 到 MFMA 的 FlyDSL layout 與 launch model；
- AITER 如何 dispatch 調校過的 kernel；
- 目前已公開的所有 K3 相關 AITER/FlyDSL operator family；
- 完整 MoE 資料路徑，包括 MXFP4/MXFP8 與 SiTUv2；
- K3 哪些部分仍使用 Triton，而非 FlyDSL；
- 如何測試一個 kernel，以及如何重現八 GPU K3 配方。

## 1. 可重現環境

本章使用：

- [AITER v0.1.23](https://github.com/ROCm/aiter/tree/v0.1.23)；
- [FlyDSL v0.3.4.1](https://github.com/ROCm/FlyDSL/tree/v0.3.4.1)；
- [Quark release/0.12](https://github.com/amd/Quark/tree/release/0.12)。

AITER v0.1.23 固定使用 FlyDSL 0.3.4.1。線上 FlyDSL 文件可能描述較新的 API，因此重現本章時請使用固定 tag 的範例。

先檢查機器：

```bash
rocminfo | rg 'gfx'
python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__, "ROCm:", torch.version.hip)
print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name())
PY
```

ROCm 刻意使用 PyTorch 的 `torch.cuda` namespace，請勿將它改為 `torch.rocm`。

| Target | 應假設的狀態 |
|---|---|
| `gfx942` (MI300X/MI325X) | AITER 支援廣泛；適合測試個別 kernel |
| `gfx950` (MI350X/MI355X) | 已發布完整 K3 配方的目標 |
| RDNA target | 許多 AITER 路徑仍屬實驗性 |

支援一個 kernel，不代表完整的 2.8 兆參數模型能在該 target 上容納或執行。

## 2. 從最小 Kernel 認識 FlyDSL

FlyDSL 是以明確 tensor、layout、copy atom 與 MMA atom 為核心的 Python DSL。Kernel 分兩層：

```python
@flyc.kernel
def vector_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                      tiled_copy: fx.TiledCopy):
    # Device program: partition tiles among threads and copy fragments.
    ...

@flyc.jit
def vector_add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
    # Host program: construct layouts and launch the device kernel.
    ...
```

執行官方第一個範例：

```bash
git clone --branch v0.3.4.1 https://github.com/ROCm/FlyDSL.git
cd FlyDSL
pip install flydsl==0.3.4.1 pytest pandas
python3 examples/01-vectorAdd.py
python3 -m pytest tests/kernels/test_vec_add.py
```

範例建立 `UniversalCopy128b` copy atom，將 \(128\)-thread layout 對應到 output tile、切分 source 與 destination tensor，並對邊界 copy 加 predicate。這些分別對應第 01、02 章中的 `float4`、thread-to-element mapping 與 bounds check。

### 2.1 學習階梯

| 步驟 | 固定 tag 的範例 | 新概念 |
|---|---|---|
| 1 | `examples/01-vectorAdd.py` | tensor、layout、predicated 128-bit copy |
| 2 | `examples/02-tiledCopy.py` | global/shared-memory tiling 與 partition |
| 3 | `examples/03-tiledMma.py` | register fragment 與 tiled MFMA |
| 4 | `examples/04-preshuffle_gemm.py` | production weight layout 與 GEMM |

關鍵問題始終是：**此刻這個 thread 擁有哪些 logical value？** Layout 回答這個問題；copy atom 或 MMA atom 則說明哪個 instruction 會搬移或使用這些值。

## 3. AITER 是 Dispatcher，不是單一 Kernel 語言

AITER 提供可從 PyTorch 呼叫的 operator。首次呼叫時可能 JIT 編譯 backend，之後快取結果。Dispatch 會考慮：

- GPU architecture；
- data type 與 quantization format；
- matrix 或 attention shape；
- alignment 與 layout；
- environment override；
- tuned CSV configuration file 中的項目。

選到的實作可能是 FlyDSL、Triton/Gluon、Composable Kernel、HIP、assembly 或其他 generator。因此，「使用 AITER」不代表「使用 FlyDSL」。

[`aiter/configs/model_configs`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/configs/model_configs)
下的 K3 專用 tuning file 涵蓋：

- BF16 與 A4W4 block-scaled GEMM；
- A8W8 preshuffled GEMM；
- A16W4、A8W4、A4W4 與 I4 fused MoE；
- FP8 FMHA。

Tuning 綁定 shape 與 architecture。未知 shape 可能使用通用設定或另一個 backend。

## 4. 量化與資料 Layout

K3 原生 QAT representation 使用 MXFP4 權重與 MXFP8 activation。另一個 AMD checkpoint 在 attention 中結合 MXFP4 與 per-token-per-channel FP8。名稱同時描述 arithmetic 與 storage。

| 格式 | 主要概念 | Kernel 工作 |
|---|---|---|
| FP8 | 每個 element 一個八位元浮點值 | amax、scale、convert |
| INT8/INT4 | 整數值加 scale | reduce、round、clamp、pack |
| MXFP8 | 小 block 內的 FP8 值共用 scale | block reduction 與 scale layout |
| MXFP4 | E2M1 值共用 E8M0 scale，通常每 block 32 個值 | block reduction、nibble packing、scale swizzle |

Preshuffling 在 inference 前重新排列權重，讓每個 MFMA wave 能以連續或無 conflict 的 access 載入所需 fragment。它不是可隨意加入或移除的數學 transpose。Converter、GEMM 與 scale layout 必須一致。

先做隔離測試：

```bash
cd aiter
python3 op_tests/test_quant_mxfp4.py
python3 op_tests/test_gemm_a8w8.py
```

測量速度前，先將反量化輸出與 FP32 reference 比較。測試應包含全零 block、最大有限值、partial group 與 rounding midpoint。

## 5. K3 MoE Pipeline

K3 有 896 個 routed expert，每個 token 選 16 個，也有 shared expert。Production path 是一條 pipeline：

```text
router logits
  → grouped top-k
  → route + quantize + scatter
  → expert GEMM 1 (gate/up)
  → SiTUv2 + activation quantization
  → expert GEMM 2 (down)
  → weighted reduce/combine
```

### 5.1 Routing

Routing 選擇 expert 並產生 weight；scatter stage 接著依 expert 分組 token row，並為 GEMM padding／對齊工作。AITER 公開程式包含供 K3 使用的 strided grouped top-k，以及進行融合搬移和量化的
[`moe_fused_route_quant_scatter.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py)。

Routing 的 FLOP 很少，但對以下事項敏感：

- 穩定的 token/expert 順序；
- expert capacity 與 empty expert；
- 單一 token 選到重複 expert；
- quantization scale 所有權；
- expert-parallel destination rank。

### 5.2 Expert GEMM 1

第一個 expert GEMM 產生 gate 與 up vector。公開 FlyDSL 路徑包括
[`mxfp4_gemm1.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm1.py)
與
[`two-stage mixed MoE GEMM`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py)。
Tile 必須在 MFMA 使用率、expert size，以及通常分到單一 expert 的少量 token 之間取捨。

### 5.3 SiTUv2 與 Requantization

Activation 就是第 15 章的公式：

$$
y =
\left[4\tanh(g/4)\sigma(g)\right]
\left[25\tanh(u/25)\right].
$$

AITER 的 `situv2_and_mul_quant` 將該算式與 row-scale 計算及 FP8 轉換融合：

```python
from aiter.ops.activation import situv2_and_mul_quant

out = torch.empty((tokens, width), device="cuda", dtype=aiter.dtypes.fp8)
scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)
situv2_and_mul_quant(out, x, scale, width, 4.0, 25.0)
```

將 `out.float() * scale` 與 FP32 reference 比較，以驗證量化結果。固定 tag 的官方測試也會檢查 zero row 與 empty batch。

```bash
python3 op_tests/test_situv2_and_mul_quant.py -d bf16 -m 1 -n 768
```

### 5.4 Expert GEMM 2 與 Combine

[`mxfp4_gemm2.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm2.py)
將 expert activation 投影回 model width。Combine stage 將每個結果乘以 routing weight，並對每個 token 選到的 expert 做 reduction。使用 expert parallelism 時，dispatch 與 combine 之間可能還有通訊。

分別測試每個邊界：routed row order、quantized activation 與 scale、GEMM output，最後才是 final weighted sum。

## 6. Attention 與 State-Space Operator

### 6.1 Gated MLA

AITER 與 FlyDSL 提供 K3 gated MLA layer 的 building block：

- FMHA 與 paged attention；
- QK normalization、RoPE 與 quantization；
- KV gather 與 B-projection；
- MLA split reduction；
- FP8 attention configuration。

相關 entry point 包括
[`fmha_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/fmha_kernels.py)、
[`qk_norm_rope_quant.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/qk_norm_rope_quant.py)
與
[`mla_reduce_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/mla_reduce_kernels.py)。

### 6.2 KDA

KDA 是重要界線：AITER 的 Triton/Gluon tree 下有 K3 KDA 實作，包括 `kimi_delta_attn`、`gated_delta_net` 與 chunk delta attention。本章所用版本中，沒有對等的公開 FlyDSL KDA 路徑。

已發布的 AMD SGLang 配方雖會啟用 AITER/FlyDSL，仍選擇 `--attention-backend triton`。正確的說法是 FlyDSL 加速 K3 的 MoE/GEMM/support 路徑；不能說每個 K3 attention kernel 都是 FlyDSL。

## 7. 通訊與多 GPU 執行

完整模型需要 tensor 和／或 expert parallelism。AITER/FlyDSL family 包括自訂 all-reduce、reduce-scatter/all-gather、dispatch/combine 與 MegaMoE 通訊。

通訊正確性牽涉的不只一個 kernel：

- 每個 rank 必須使用相同 token/expert metadata；
- send/receive count 必須一致，包括 empty expert；
- reduction dtype 與順序會影響誤差；
- stream 與 event 必須保護 buffer；
- graph capture 需要穩定 address 與 launch topology。

先執行單 GPU operator test。FlyDSL 將多 GPU all-reduce 測試獨立放在 `tests/kernels/test_allreduce.py`。

## 8. 測試階梯

### 8.1 FlyDSL smoke test

```bash
python3 examples/01-vectorAdd.py
python3 -m pytest tests/kernels/test_vec_add.py
python3 -m pytest tests/kernels/test_preshuffle_gemm.py -m "not large_shape"
python3 -m pytest tests/kernels/test_quant.py
python3 -m pytest tests/kernels/test_moe_gemm.py
python3 -m pytest tests/kernels/test_flash_attn_fwd.py
```

FlyDSL 將不需要裝置、可攜式編譯、ROCDL lowering 與真實 GPU 執行分開測試。不要在未閱讀 marker 的情況下，假設 compile test 通過就代表數值正確。

### 8.2 AITER operator test

安裝符合 ROCm 與 Python 的 v0.1.23 wheel，或遞迴建置固定 tag 的儲存庫：

```bash
git clone --recursive --branch v0.1.23 https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
python3 op_tests/test_rmsnorm2d.py
python3 op_tests/test_gemm_a8w8.py
python3 op_tests/test_mla.py
python3 op_tests/test_moeTopkSoftmax.py
python3 op_tests/test_moe_2stage.py
```

每張 benchmark table 都要保留 correctness 欄。Steady-state 計時必須排除 JIT compilation。

### 8.3 完整 K3 配方

已發布的完整模型配方需要八張 MI350X 或 MI355X GPU，以及約 1.56 TB checkpoint storage：

```bash
madengine run --tags pyt_sglang_kimi-k3 --keep-model-dir --live-output
```

主要環境選項包括：

```bash
SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1
AITER_FLYDSL_FORCE=1
AITER_SITUV2_A8W4=1
```

完整 launch flag 請以 [ROCm MAD K3 配方](https://github.com/ROCm/MAD/blob/develop/benchmark/kimi_k3/README.md)為準。小 kernel 測試成功不能取代這項 end-to-end validation。

## 9. 從 Quark 到 Runtime 的流程

AMD K3 checkpoint 配方使用直接轉換，避免以一般方式載入完整模型：

```python
template = LLMTemplate.get("kimi_k3")
quant_config = template.get_config(
    scheme="mxfp4",
    layer_config={"*self_attn*": "ptpc_fp8"},
)
ModelQuantizer(quant_config).direct_quantize_checkpoint(
    pretrained_model_path="moonshotai/Kimi-K3",
    save_path=output_dir,
)
```

這是資料格式邊界。請一併記錄 Quark 版本、template、layer override、runtime image 與 checkpoint identifier。模擬某種格式的 runtime flag 通常只是綁定版本的重現 workaround，不是一般需求。

## 10. 除錯與調校

請依此順序：

1. **參考：** 與 FP32 PyTorch 實作比較。
2. **邊界：** 測試 zero token、one token、奇數尺寸、partial group 與 empty expert。
3. **Dispatch：** 記錄 architecture、選到的 backend 與 tuning row。
4. **Compiler：** 檢查 FlyDSL/MLIR/ROCDL output 與 resource usage。
5. **Profiler：** 用 `rocprofv3` 測量 wave、MFMA 使用率、LDS conflict、cache miss 與 launch gap。
6. **End to end：** 測量 scheduler、通訊與模型輸出品質。

如果 `AITER_FLYDSL_FORCE=1` 讓某個 shape 失敗，先確認 FlyDSL 是否支援該 shape。強制 backend 只會移除 fallback，不會增加 coverage。

## 重點整理

1. FlyDSL 在 Python 中明確呈現 layout、copy 與 MFMA ownership。
2. AITER dispatch 多個 backend family；請檢查實際選到哪一個。
3. K3 的主要 FlyDSL 路徑是 quantized MoE：route、scatter、兩個 GEMM、SiTUv2/requantize 與 combine。
4. Gated MLA 有 FlyDSL building block，而公開 KDA 路徑目前使用 Triton/Gluon。
5. 確切的 quantized layout 與 tuning row 都是正確性的一部分。
6. 完整 K3 驗證需要受支援的八 GPU 配方；個別 kernel 可在較小、受支援的 AMD 系統上學習與檢查。

## 練習

1. 追蹤 FlyDSL vector-add 範例中的 thread 與 value layout。
2. 以奇數 \(M\) 執行 preshuffled GEMM，找出哪些 predicate 保護邊界 tile。
3. 將 AITER 的 SiTUv2 FP8 輸出與第 15 章 Triton FP32 輸出比較。
4. 在有／無 `AITER_FLYDSL_FORCE` 時，記錄一個 K3 MoE shape 的 AITER dispatch。
5. 畫出 two-stage MoE test 每個 stage 之間的 buffer 與 scale。
