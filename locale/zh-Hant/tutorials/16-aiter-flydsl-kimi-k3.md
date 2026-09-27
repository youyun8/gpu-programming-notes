# 16 – AITER 中的 FlyDSL – 在 AMD GPU 上執行 Kimi K3

> **第六部 · Kimi K3 案例** · 先備知識：
> [矩陣乘法 1 – 基礎](04-tiled-matmul.md)與
> [矩陣乘法 8 – Tensor Core](gemm/07-tensor-cores.md)、
> [Triton – 從第一個 Kernel 到生產環境](14-triton.md)，以及
> [CDNA3 與 MFMA](05-amd-cdna3-mfma.md) ·
> 上一章：[15 – SGLang 中的 Triton：以 Kimi K3 服務為例](15-triton-model-systems.md) ·
> 下一章：[08 – 部署本站](08-deploying-this-site.md)

本章將沿著一條 production 路徑，從 Kimi K3 checkpoint 一路追蹤到
AMD GPU 上的 kernel。內容聚焦於透過 AITER 使用的 FlyDSL kernel，
但會清楚區分各軟體的邊界：

```text
Quark converts checkpoint data
    → SGLang, vLLM, or ATOM schedules a request
        → AITER exposes and dispatches operators
            → FlyDSL, Triton/Gluon, CK, HIP, assembly, or Opus runs a kernel
```

Quark 不是 serving runtime。AITER 不是單一 kernel 語言。FlyDSL
也沒有實作每一個 AITER operator。特別是本章介紹的公開 K3 配方，
仍由 Triton/Gluon 路徑處理 Kimi Delta Attention（KDA）。

**你將學會**

- Kimi K3、Quark、serving runtime、AITER 與 FlyDSL 各自的角色；
- FlyDSL 的 tensor、layout、copy 與 MMA 模型；
- AITER 如何結合 dispatch、JIT 與 AOT 封裝，以及調校表；
- 量化與預先重排的儲存方式如何成為 kernel contract；
- 從 routing 到 weighted combine 的完整 K3 MoE 路徑；
- 哪些面向 K3 的公開 AITER family 使用 FlyDSL，哪些不使用；
- MLA、expert parallelism 與通訊如何搭配 MoE 路徑；
- 如何先測試單一 operator，再執行八張 GPU 的 MAD 配方。

## 1. 固定公開軟體堆疊的版本

本章的範例與原始碼連結使用：

- [AITER v0.1.23](https://github.com/ROCm/aiter/tree/v0.1.23)；
- [FlyDSL v0.3.4.1](https://github.com/ROCm/FlyDSL/tree/v0.3.4.1)；
- [Quark release/0.12](https://github.com/amd/Quark/tree/release/0.12)。

AITER v0.1.23 固定使用 FlyDSL 0.3.4.1。FlyDSL 變化很快，因此請使用
固定 tag 的範例，不要從目前的線上文件複製 API。

建置前先檢查機器：

```bash
rocminfo | rg 'gfx'
python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__, "ROCm:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())
PY
```

ROCm 刻意使用 PyTorch 的 `torch.cuda` namespace。請勿將它改成
`torch.rocm`。

| 目標 | 本章可安全預期的支援程度 |
|---|---|
| `gfx942` (MI300X/MI325X) | 許多 AITER operator，以及實用的本機 kernel 測試 |
| `gfx950` (MI350X/MI355X) | 已發布八張 GPU K3 配方的目標 |
| RDNA 目標 | 本章討論的許多路徑仍屬實驗性或不受支援 |

支援個別 operator，不代表同一張 GPU 能容納完整的 2.8 兆參數模型，
也不代表已發布的配方支援該 GPU。

## 2. 讓每一層只負責一項工作

| 層級 | 工作 | 不保證的事項 |
|---|---|---|
| Kimi K3 model | 定義架構、權重、896 個 routed expert、top-16 routing、KDA 與 gated MLA | 特定的 AMD kernel backend |
| Quark | 量化並轉換 checkpoint tensor | request scheduling 或 runtime dispatch |
| SGLang、vLLM 或 ATOM | 管理 batch、cache、graph capture、parallel rank 與 backend 選擇 | 每個選到的 AITER operator 都使用 FlyDSL |
| AITER | 提供可由 PyTorch 呼叫的 AMD operator，並選擇實作 | 所有 operator 都使用同一種實作語言 |
| FlyDSL | 使用 Python 描述 layout、copy、MMA 工作並啟動執行 | 完整的 model server |

這項區分也能用於除錯。轉換後 checkpoint 中錯誤的 scale，無法靠修改
MFMA tile 修正。本機 GEMM 結果正確，也不能證明 expert-parallel metadata
已正確交換。

## 3. FlyDSL：從數值到 MFMA

FlyDSL 是一種 Python DSL，明確區分 device program 與 host 端 launch
program：

```python
@flyc.kernel
def vector_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    tiled_copy: fx.TiledCopy,
):
    # Device program: partition tensors and move each thread's values.
    ...

@flyc.jit
def vector_add(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    # Host program: build layouts, specialize, and launch the kernel.
    ...
```

確切的 API 會持續演進，但思考模型保持不變。

### 3.1 Tensor 與 layout

`fx.Tensor` 是 storage 的 view。它的 layout 會將 logical coordinate
對應到 physical offset。Layout 可以描述 global-memory tile、LDS tile
或 register fragment，也能拆分及組合維度。

在每個階段都要問：

> 這個 thread 或 wave 擁有哪些 logical value？這些值儲存在哪裡？

這個問題取代手寫的 index 算術。Layout 是正確性的一部分，不只是
效能提示。

### 3.2 Copy atom 與 tiled copy

Copy atom 描述一種合法的搬移方式，例如向量化的 128-bit copy。
Tiled copy 會在較大的 logical tile 上重複使用該 atom，並將各部分
分配給 thread。來源與目的地的 partition 必須對 value ownership
有一致的定義。

在邊緣 tile，copy 仍需要 predicate。只有 alignment、有效 lane 與
storage layout 都符合 contract 時，寬向量操作才安全。

### 3.3 MMA atom 與 tiled MMA

MMA atom 代表一個硬體矩陣 instruction。在 CDNA 上，通常就是 MFMA
操作。Tiled MMA 會在較大的 output tile 上展開該 atom，並定義：

- 哪些 lane 持有 A 與 B fragment；
- 每個 lane 擁有哪些 accumulator value；
- wave 如何在 M、N、K 維度重複執行工作；
- register fragment 如何連接 global 或 LDS copy。

Kernel 通常會用 pipeline 搭配 copy 與反覆執行的 MMA step，接著套用
epilogue 並儲存結果。這就是[分塊矩陣乘法](04-tiled-matmul.md)中介紹的
矩陣路徑，現在改用 layout algebra 表達。

### 3.4 依序學習四個固定版本的步驟

```bash
git clone --branch v0.3.4.1 https://github.com/ROCm/FlyDSL.git
cd FlyDSL
pip install flydsl==0.3.4.1 pytest pandas
python3 examples/01-vectorAdd.py
```

| 步驟 | 固定 tag 的範例 | 新概念 |
|---|---|---|
| 1 | [`examples/01-vectorAdd.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/01-vectorAdd.py) | tensor、layout、predicate 與 128-bit copy |
| 2 | [`examples/02-tiledCopy.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/02-tiledCopy.py) | global/LDS tile 與 thread partitioning |
| 3 | [`examples/03-tiledMma.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/03-tiledMma.py) | register fragment 與 tiled MFMA |
| 4 | [`examples/04-preshuffle_gemm.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/04-preshuffle_gemm.py) | production weight layout 與 GEMM |

不要一開始就研究完整的 MoE kernel。請先透過這四個範例追蹤一個 tile：
global memory → LDS 或 register → MMA fragment → output。

## 4. AITER 的 Dispatch、編譯與調校

AITER 為 runtime 提供穩定的 operator 介面。選到的實作可能是
FlyDSL、Triton/Gluon、Composable Kernel、HIP、assembly、Opus 或其他
generator。

Dispatch 可能依據下列資訊：

- GPU architecture；
- input、weight、accumulator 與 output type；
- 量化與 scale format；
- M、N、K、token 數、page size 與 attention mode；
- stride、alignment、preshuffle state 與其他 layout 資訊；
- environment override；
- 符合條件的調校設定。

因此：

> 「runtime 使用 AITER」不代表「runtime 使用 FlyDSL」。

### 4.1 JIT

FlyDSL wrapper 可以在第一次呼叫時針對 kernel 進行 specialize 與編譯，
之後由 AITER 快取結果。Cold call 包含 Python dispatch、code generation、
編譯與載入。Warm call 應重複使用已編譯的 specialization。

計時前務必先 warm up。比較執行結果時，也要記錄 cache state。

### 4.2 AOT

AITER 也提供 AOT manifest，包括公開的 MoE family FlyDSL manifest。
這些 manifest 描述可在執行模型前建置並封裝的 kernel。AOT 能減少啟動
工作，但只適用於已匯出的 specialization。未涵蓋的 shape 仍可能需要
JIT 編譯或使用其他 backend。

請參閱固定 tag 的
[`aiter/aot/flydsl`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/aot/flydsl)
目錄。Manifest 並不能證明 runtime 選到了該 kernel。

### 4.3 已調校與未調校的表格

下列
[`aiter/configs/model_configs`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/configs/model_configs)
中的 K3 專用檔案涵蓋這些公開 format family：

- BF16 GEMM；
- A8W8 B-preshuffled GEMM；
- A4W4 block-scaled GEMM；
- A16W4、A8W4、A4W4 與 I4 fused MoE；
- FP8 FMHA AOT 設定。

該目錄包含 tuned 與 untuned variant。Tuned row 是針對特定 shape 與
architecture 實測後選出的結果，不代表同一個 tile 在所有情況下都是
最佳選擇。

`AITER_FLYDSL_FORCE=1` 會移除部分 fallback 選項，但不會為不受支援的
operator、shape 或 architecture 建立 FlyDSL 實作。

## 5. 面向 K3 的公開 Operator 目錄

下表整理固定版本的公開原始碼中，面向 K3 的 FlyDSL/AITER 使用情境。
「可用」表示有公開的 operator 或調校路徑，不代表每個 serving framework
都會為每個 K3 request 選擇它。

| K3 使用情境 | 公開的 AITER/FlyDSL family | 邊界 |
|---|---|---|
| Dense projection | BF16 GEMM、A8W8 B-preshuffled GEMM 與 A4W4 block-scaled GEMM 調校 | AITER 可能選擇 FlyDSL、assembly 或其他 backend |
| Top-k 與 expert metadata | top-k、MoE sorting、route map 與 group/local lookup helper | Routing policy 屬於 model/runtime |
| 量化並 scatter routed row | [`moe_fused_route_quant_scatter.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py)、scatter-copy、scale-preshuffle helper | Row order、destination、data 與 scale 共同構成一份 contract |
| Expert GEMM 1 | [`mxfp4_gemm1.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm1.py)與 MoE stage-1 wrapper | 產生 gate/up data；支援的 format 與 shape 取決於 dispatch |
| 混合格式的 two-stage MoE | [`mixed_moe_gemm_2stage.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py) | 涵蓋公開的 mixed-format 路徑，但不是每一種 fused-MoE 路徑 |
| SiTUv2 加上 requantization | AITER `situv2_and_mul_quant` | 公開的 AITER operator；未檢查選到的 build 前，不要將它標示為 FlyDSL |
| Expert GEMM 2 | [`mxfp4_gemm2.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm2.py)與 MoE stage-2 wrapper | 使用 activation 及其 scale |
| Expert output combine | FlyDSL gather/reduce、MoE reduce、dispatch/combine 與 MegaMoE helper | Local combine 與 inter-rank combine 是不同情況 |
| MLA 準備工作 | [`qk_norm_rope_quant.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/qk_norm_rope_quant.py)、KV gather/B projection | 有準備路徑不代表 attention core 使用 FlyDSL |
| MLA attention 與 merge | [`fmha_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/fmha_kernels.py)、paged/FP8 MLA 路徑，以及 [`mla_reduce_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/mla_reduce_kernels.py) | AITER 也有 assembly、Triton 與 Opus MLA 路徑 |
| Expert-parallel 搬移 | intra-node dispatch/combine、communication-fused MoE、MegaMoE 與 quick all-reduce | 需要相符的 rank metadata、IPC 設定與 topology |
| KDA | AITER Triton/Gluon KDA family | 本章不宣稱存在公開的 K3 FlyDSL KDA 路徑 |

AITER 中其他通用的 FlyDSL kernel，例如 convolution、HSTU、無關的 GEMM
與其他 model-specific operator，不屬於這份 K3 目錄。反過來說，這份
目錄也不宣稱 K3 全部都在 FlyDSL 中執行。

## 6. 量化與 Preshuffling

K3 的原生 QAT representation 使用 MXFP4 weight 與 MXFP8 activation。
另一條 AMD checkpoint 路徑則在 attention 中結合 MXFP4 與
per-token-per-channel FP8。Format 名稱同時描述 arithmetic 與 storage。

| Format | 儲存的值 | Scale 工作 |
|---|---|---|
| FP8 | 每個 element 一個八位元浮點值 | 找出 amax、推導 scale、轉換 |
| INT8/INT4 | 有號整數加上一個 scale | reduce、round、clamp，有時還要 pack |
| MXFP8 | 小型 block 中共用 scale 的 FP8 value | 對每個 block 執行 reduce，並排列 block scale |
| MXFP4 | 共用 E8M0 block scale 的 packed E2M1 value | reduce、encode、nibble-pack，並 swizzle scale |

精確的 block size、scale orientation 與 byte order 由 operator contract
決定。不要只根據「4-bit」自行推斷。

### 6.1 為什麼需要 preshuffle

數學上的 weight matrix 以 \(W[k,n]\) 索引，但單一 MFMA wave 不會按照
簡單的 row-major 順序使用它。Preshuffling 會依照 copy 與 MMA layout
所需的順序儲存 weight，必要時也會重排 scale。

這樣可以：

- 讓 wave load 連續；
- 避免 runtime 反覆重新排列；
- 減少 LDS bank conflict；
- 將 packed nibble 與 scale 放在會使用它們的 fragment 旁邊。

Preshuffling 不是沒有代價的 transpose。Checkpoint converter 或準備
步驟、選到的 GEMM、scale layout 與 architecture 必須彼此一致。只要其中
一份 contract 不符，kernel 仍可能執行，但產生看似合理卻錯誤的值。

### 6.2 在 GEMM 前驗證 format

先執行：

```bash
cd aiter
python3 op_tests/test_quant_mxfp4.py
python3 op_tests/test_gemm_a8w8.py
```

測試全零 block、最大有限值、rounding midpoint、partial group、奇數
row count 與空 input。測量 GEMM 速度前，先將反量化的值與 FP32 reference
比較。

## 7. 完整的 K3 MoE 路徑

K3 有 896 個 routed expert，每個 token 選擇 16 個 expert，也會使用
shared expert。Routed 路徑如下：

```text
router logits
  → grouped top-k and routing weights
  → route + quantize + scatter
  → expert GEMM 1 (gate/up)
  → SiTUv2 + activation requantization
  → expert GEMM 2 (down)
  → weighted reduce/combine
```

每個箭頭都是可測試的邊界。

### 7.1 Route

Router 會產生選到的 expert ID 與 weight。Runtime policy 也可能重新
normalize weight，並將 global expert 對應到 local rank。AITER 的 routing
與 sorting helper 會將這種 sparse 選擇轉成 grouped GEMM 可使用的
metadata。

請檢查：

- 穩定的 token/expert 順序；
- global 與 local expert ID 的意義；
- caller 可能產生 duplicate selection 時的處理方式；
- empty expert 與 zero-token batch；
- capacity、padding 與 alignment；
- routing-weight dtype 與 normalization。

Routing 的 FLOP 很少，但只要一個 index 錯誤，就可能破壞完整 layer。

### 7.2 Quantize 與 scatter

融合的 route/quant/scatter kernel 會讀取 token row、計算所需的 activation
scale、轉換數值，並將該 row 寫入依 expert 分組的 storage。使用 expert
parallelism 時，它也會準備要送到其他 rank 的 data 與 metadata。

請同時驗證四項 output：

1. destination expert 或 rank；
2. destination row；
3. quantized data；
4. 用來還原該 row 的 scale。

只測試 quantized byte 會遺漏 routing 錯誤。只測試 row map，則會遺漏
scale 綁到錯誤 token 的問題。

### 7.3 Expert GEMM 1

GEMM 1 會套用每個選定 expert 的 gate 與 up projection。即使整體 batch
很大，單一 expert 的有效 M 仍可能很小且不平均。Kernel 必須在 MFMA
效率、padding 與 launch overhead 之間取得平衡。

公開的 MXFP4 與混合格式 two-stage kernel，會為受支援的 format 與 shape
組合提供 FlyDSL 路徑。實際呼叫是否使用這些路徑，仍由 dispatch 決定。

請在 scattered row order 下比較這個邊界的 output。此時不要將 row
合併回 token order，否則可能掩蓋排序錯誤。

### 7.4 SiTUv2 與 activation requantization

K3 對 gate input \(g\) 與 up input \(u\) 使用：

$$
y =
\left[\beta_1\tanh(g/\beta_1)\sigma(g)\right]
\left[\beta_2\tanh(u/\beta_2)\right],
\qquad \beta_1=4,\quad \beta_2=25.
$$

| 符號 | 意義 |
|---|---|
| \(g,u\) | GEMM 1 產生的 gate 與 up half |
| \(\sigma\) | Sigmoid |
| \(\beta_1,\beta_2\) | K3 bound |
| \(y\) | 融合後的 expert activation |

AITER 的 `situv2_and_mul_quant` 會將此算式與 row-scale 計算及 FP8
轉換融合：

```python
from aiter.ops.activation import situv2_and_mul_quant

out = torch.empty((tokens, width), device="cuda", dtype=aiter.dtypes.fp8)
scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)
situv2_and_mul_quant(out, x, scale, width, 4.0, 25.0)
```

將 `out.float() * scale` 與公式的 FP32 實作比較。測試中也要保留
zero row 與 empty batch：

```bash
python3 op_tests/test_situv2_and_mul_quant.py -d bf16 -m 1 -n 768
```

這是一個 AITER operator 邊界。只看公開 API，無法證明其實作為
FlyDSL。

### 7.5 Expert GEMM 2

GEMM 2 使用 quantized activation 及其 row scale，再套用每個 expert
的 down projection。它讀取 row 的順序，必須與 scatter 和 GEMM 1
產生的順序完全相同。

套用 routing weight 前，先比較它的 output。這裡若有不一致，不應透過
最後合併的 tensor 進行除錯。

### 7.6 Weighted combine

Combine stage 會恢復 token order，將每個選定 expert 的 output 乘上
routing weight，再對 top-16 contribution 執行 reduce。它也可以在
model 定義的位置加入 shared-expert result。

使用 expert parallelism 時，combine 還包含通訊問題：

```text
local expert outputs
  → send or reduce across ranks
  → gather contributions for each original token
  → multiply by routing weights
  → sum in token order
```

先檢查一個 token 的單一 expert contribution，再檢查全部 16 個。
最後，才將完整的 local 或 distributed combine 與 high-precision
reference 比較。

## 8. MLA 支援與 KDA 邊界

K3 混用 gated Multi-head Latent Attention（MLA）layer 與 KDA layer。
它們不必使用相同的 backend。

### 8.1 公開的 FlyDSL/AITER MLA building block

固定版本的 AITER 原始碼包括下列公開路徑：

- Q/K normalization、RoPE 與 quantization；
- KV gather 與 B projection；
- FP8 FMHA；
- 適用於受支援 page layout 的 paged MLA 工作；
- split attention output 與 log-sum-exp reduction。

這些都是實際的 FlyDSL/AITER 使用情境，但支援範圍受到 architecture、
dtype、head shape、page size，以及 decode 或 prefill mode 限制。AITER
也提供非 FlyDSL 的 MLA 實作。請檢查 dispatch，不要將所有 MLA 時間都
算在 FlyDSL 上。

### 8.2 KDA 仍使用 Triton/Gluon

KDA 是本章明確的 backend 邊界。AITER 公開的 K3 KDA 工作位於固定 tag 的
[`kimi_delta_attn`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/ops/triton/kimi_delta_attn)、
[`gated_delta_net`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/ops/triton/gated_delta_net)
與
[`chunk_delta_attn`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/ops/triton/_triton_kernels/chunk_delta_attn)
Triton/Gluon family。原始碼 tree 中也有通用的 FlyDSL gated-delta helper
code；該 helper 不能證明存在完整的 K3 FlyDSL KDA backend。本章不做此
宣稱。

已發布的 SGLang K3 command 會啟用 AITER，並強制使用受支援的 FlyDSL
路徑，但仍傳入：

```bash
--attention-backend triton
```

這並不矛盾。同一個 model request 可以讓 FlyDSL 處理 MoE 與支援
operator，同時由 Triton/Gluon 處理 KDA 及選定的 attention backend。

## 9. 通訊與 Expert Parallelism

Tensor parallelism 會拆分 dense 工作；expert parallelism 則將 expert
放在不同 rank。K3 需要在 grouped GEMM 前後保持一致的 routing metadata
與資料搬移。

與此設計相關的公開 FlyDSL/AITER 通訊 family 包括：

- intra-node dispatch/combine；
- communication-fused MoE；
- MegaMoE dispatch、GEMM 與 combine stage；
- 類似 reduce-scatter/all-gather 的支援；
- 適用於受支援 format 的 quick all-reduce 路徑。

原始碼中可用的 family，不代表特定 K3 配方已選擇它。Runtime flag、
world size、topology、data type 與 shape 仍會影響選擇。

為了確保正確性：

- 所有 rank 必須使用一致的 token 與 expert metadata；
- send 與 receive count 必須正確納入 empty expert；
- 不可混用 local 與 global expert ID；
- stream 與 event 必須保護可重複使用的 buffer；
- IPC handle 與 peer access 必須有效；
- graph capture 需要穩定的 address 與 launch topology；
- reduction 順序與 dtype 可能改變數值誤差。

先從一張 GPU 開始，再用手寫 route map 測試兩個 rank。Empty 與
imbalanced expert 的情況通過後，才增加到完整 rank 數量。

## 10. Quark 止於 Checkpoint 邊界

AMD K3 conversion 路徑可以直接量化 checkpoint，不必先以傳統方式載入
完整 model：

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

Quark 寫入 data 與 metadata，不會選擇 runtime kernel。若 checkpoint
的 packing 或 scale layout 錯誤，AITER 不會重新解讀它。

請將下列項目記錄為同一組可重現資訊：

- source checkpoint identifier 與 revision；
- Quark release 與 K3 template；
- quantization scheme 與 layer override；
- output checkpoint identifier；
- serving image 與 AITER/FlyDSL 版本；
- 目標 GPU architecture。

模擬舊版 layout 的 runtime compatibility flag 可能對某個 release
有用，但不應將它描述為 MXFP4 的永久特性。

## 11. 本機 Operator 測試階梯

不要從八張 GPU 開始。請依序完成下列階梯，並讓每一項計時結果都有
對應的正確性結果。

### 11.1 FlyDSL 語言 smoke test

```bash
cd FlyDSL
python3 examples/01-vectorAdd.py
python3 -m pytest tests/kernels/test_vec_add.py
python3 -m pytest tests/kernels/test_preshuffle_gemm.py -m "not large_shape"
python3 -m pytest tests/kernels/test_quant.py
python3 -m pytest tests/kernels/test_moe_gemm.py
python3 -m pytest tests/kernels/test_flash_attn_fwd.py
```

只測試編譯或 lowering，不能證明 GPU 數值正確。判讀通過結果前，
請先查看 test marker 與 target。

### 11.2 AITER operator 測試

安裝符合 ROCm 與 Python 的 wheel，或建置固定版本的原始碼：

```bash
git clone --recursive --branch v0.1.23 https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

接著一次測試一個邊界：

```bash
python3 op_tests/test_quant_mxfp4.py
python3 op_tests/test_gemm_a8w8.py
python3 op_tests/test_situv2_and_mul_quant.py -d bf16 -m 1 -n 768
python3 op_tests/test_moeTopkSoftmax.py
python3 op_tests/test_moe_2stage.py
python3 op_tests/test_mla.py
python3 op_tests/test_rmsnorm2d.py
```

使用該測試支援的參數，加入下列情況：

1. zero token；
2. one token；
3. 奇數與邊緣 dimension；
4. empty 與嚴重 imbalanced expert；
5. 一個確切的 K3 production shape；
6. cold 與 warm launch 測量。

### 11.3 多張 GPU 測試

只有在本機 MoE 通過後，才執行通訊測試。FlyDSL 的 all-reduce 測試位於
`tests/kernels/test_allreduce.py`；AITER 則有獨立的 multi-GPU operator
測試。啟動測試前，請確認其預期的 world size 與 topology 符合機器。

## 12. 八張 GPU 的 MAD 配方

公開的
[ROCm MAD K3 配方](https://github.com/ROCm/MAD/blob/develop/benchmark/kimi_k3/README.md)
是完整 model command 與 image 的依據。它需要：

- 八張 MI350X 或 MI355X GPU（`gfx950`）；
- 已發布 launch 中的 tensor parallel size 8；
- 約 1.56 TB 的 checkpoint 儲存空間。

MAD 發布了三種 framework 的 K3 執行方式：

```bash
# vLLM
madengine run --tags pyt_vllm_kimi-k3 --keep-model-dir --live-output

# SGLang
madengine run --tags pyt_sglang_kimi-k3 --keep-model-dir --live-output

# ATOM
madengine run --tags pyt_atom_kimi-k3 --keep-model-dir --live-output
```

在聚焦 FlyDSL 的 SGLang 路徑中，已發布的 container 使用：

```bash
SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1
AITER_FLYDSL_FORCE=1
AITER_SITUV2_A8W4=1
```

並以可看出重要 backend 邊界的方式啟動 server：

```bash
sglang serve --model-path /model_weights \
  --trust-remote-code --tp-size 8 \
  --attention-backend triton --dtype bfloat16 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 256 \
  --host 127.0.0.1 --port 30000 \
  --disable-radix-cache \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```

請使用 MAD 提供的 image 與完整 command，不要混用不同 release 的
flag。本機 FlyDSL 測試成功是必要證據，但不能取代載入 checkpoint、
提供 request 服務及檢查 model output。

## 13. Profiling 與除錯

請依照以下順序：

1. **Reference。** 將單一邊界與 FP32 PyTorch 實作比較。
2. **Format。** 驗證 packed value、scale shape、scale ownership 與
   preshuffle 版本。
3. **Metadata。** 檢查 token row、expert ID、rank ID 與 empty expert。
4. **Dispatch。** 記錄 architecture、operator、選到的 backend 與
   tuning row。
5. **Compilation。** 將 cold JIT 時間與 warm execution 分開，並保留
   失敗 specialization 的 compiler error。
6. **Kernel profile。** 使用 `rocprofv3` 檢查 launch gap、duration、
   memory traffic、MFMA 使用率、occupancy 與 LDS 行為。
7. **Communication profile。** 尋找 rank imbalance、serialization、
   缺少 overlap 與過多 copy。
8. **End to end。** 測量 serving latency 與 throughput，再檢查 output
   quality。

常見的失敗模式：

| 症狀 | 優先檢查 |
|---|---|
| BF16 正確，但 quantized output 錯誤 | Scale orientation、block size、packing 與 preshuffle |
| 單一 expert 錯誤 | Route map、local expert ID，以及該 expert 的 scale/weight offset |
| 只有最終結果錯誤 | Routing weight 與 combine order |
| 第一次呼叫非常慢 | JIT compilation 與 cache path |
| 強制使用 FlyDSL 時失敗 | 該 shape 與 architecture 是否有受支援的 FlyDSL 路徑 |
| 一個 rank 停住 | Send/receive count、empty expert、stream event 與 collective order |
| 小型測試通過，但 graph capture 失敗 | Stable buffer、warm-up coverage 與 capture-safe communication |

## 14. Production 檢查清單

將 K3 路徑視為 production-ready 前，請記錄：

- [ ] 確切的 checkpoint、Quark、runtime image、AITER 與 FlyDSL revision；
- [ ] GPU architecture、firmware/driver、ROCm、PyTorch 與 world size；
- [ ] 每個關鍵 operator 選到的 backend 與 tuning row；
- [ ] quantization、scale、packing 與 preshuffle contract；
- [ ] route、scatter、兩個 GEMM、activation 與 combine 的本機 reference
      結果；
- [ ] empty、奇數、imbalanced 與支援範圍內的最大 shape；
- [ ] cold-start 與 steady-state 測量；
- [ ] graph-capture 行為與 fallback 行為；
- [ ] multi-rank timeout、error propagation 與 health check；
- [ ] weight、cache、JIT/AOT code 與通訊 buffer 的 memory headroom；
- [ ] end-to-end accuracy 或 task-quality 檢查，而非只有 kernel tolerance；
- [ ] 具代表性的 prompt length 與 concurrency profile trace。

## 重點整理

1. FlyDSL 會明確呈現 value ownership、copy 與 MFMA layout。
2. AITER 會 dispatch 多種 backend family；呼叫 AITER 不能證明使用了
   FlyDSL kernel。
3. 公開的 FlyDSL K3 核心路徑是 quantized MoE：route、quantize/scatter、
   GEMM 1、SiTUv2/requantize、GEMM 2 與 combine。
4. AITER 提供實用的 FlyDSL MLA building block，而在本章所述的公開 K3
   路徑中，KDA 仍以 Triton/Gluon 為邊界。
5. Quantized layout、scale、preshuffling、tuning 與通訊 metadata
   都是正確性的一部分。
6. 本機 operator 測試與八張 GPU 的 MAD 執行回答不同問題；兩者都是
   production 證據所必需。

## 練習

1. 追蹤 FlyDSL vector addition 中的 value ownership。標示 logical
   coordinate、physical offset、thread、copy atom 與 edge predicate。
2. 畫出一個從 global weight 到 MFMA B fragment 的 preshuffled GEMM
   tile，並指出 checkpoint converter 還必須知道哪些資訊。
3. 使用與不使用 `AITER_FLYDSL_FORCE=1`，分別記錄一個 K3 GEMM shape
   的 AITER dispatch。解釋結果時，不要假設強制路徑支援所有 shape。
4. 建立 SiTUv2 的 FP32 reference，並針對 zero、bound 與 rounding
   midpoint，將它與反量化後的 `situv2_and_mul_quant` output 比較。
5. 為 route → quant/scatter → GEMM 1 → SiTUv2/requant → GEMM 2 →
   combine 畫出 buffer diagram，標示 row order、dtype、scale shape
   與 owner。
6. 建立 two-rank expert-parallel 測試，其中包含一個 empty expert 與
   一個 hot expert。執行前，先列出 send 與 receive count。
7. 根據 AITER trace，將每個 K3 operator 分類為 FlyDSL、
   Triton/Gluon、assembly/Opus、HIP/CK 或 unknown。在 dispatch 證據
   識別出 unknown 項目前，請保持其 unknown 狀態。
