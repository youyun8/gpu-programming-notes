# 15 – SGLang 中的 Triton – 以 Kimi K3 服務為例

> **第六部 · Kimi K3 案例** · 先備知識：
> [14 – Triton：從第一個 Kernel 到生產環境](14-triton.md) ·
> 程式：[`examples/15-triton-k3/`](examples/15-triton-k3/test_model_kernels.py) ·
> 下一章：[16 – AITER 中的 FlyDSL：在 AMD GPU 上執行 Kimi K3](16-aiter-flydsl-kimi-k3.md)

本章將從小型 Triton 程式一路走到完整的服務流程。
系統採用 SGLang，並以 Kimi K3 作為端對端追蹤案例。我們會從一個融合 activation 開始，
接著追蹤 state 與 token 如何流經 attention、expert、sampling、通訊與 production dispatch。

這三個專案的角色各不相同。**Kimi K3** 定義開放權重模型。
**AMD Quark** 負責轉換與量化 checkpoint。**SGLang** 負責服務，
包括排程、cache、分散式執行與 backend 選擇。SGLang 結合 Triton、CUDA、
CuTe DSL、AITER、FlashInfer 與供應商函式庫。SGLang 呼叫的 kernel
不一定是 Triton kernel。

這裡的範例都小到可以在 Triton 的 CPU interpreter 中執行。
它們不會複製綁定特定版本的實作細節，但會呈現與 production kernel 相同的資料流。

**你將學會**

- SGLang 中各類 Triton 使用情境的完整地圖；
- SiTU-GLU、MXFP4 風格量化、indexed cache 與 recurrent KDA 的運作方式；
- decode、chunkwise prefill 與 speculative decoding 為何需要不同的 state 演算法；
- MLA、MoE、sampling 與通訊在 K3 request 中的位置；
- production dispatch 為何有時會選擇 CUDA、CuTe DSL、AITER 或供應商函式庫，而不是 Triton。

## 1. 原始碼界線與固定版本

| 層級 | 專案 | 職責 |
|---|---|---|
| 模型定義 | [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) | 架構、權重與技術報告 |
| 模型最佳化 | [AMD Quark](https://github.com/amd/Quark/tree/release/0.12) | 量化、校正與 checkpoint 轉換 |
| Serving runtime | [SGLang](https://github.com/sgl-project/sglang) | Batching、cache 所有權、排程與 backend dispatch |
| 可攜式 kernel | Triton / FLA | Attention、state update、routing 與資料搬移 |
| 硬體專用 kernel | AITER、CuTe DSL、FlashKDA、FlashInfer | 針對支援裝置與 shape 的較快路徑 |

[K3 官方儲存庫](https://github.com/MoonshotAI/Kimi-K3) 包含架構、設定與模型程式碼，
但**不含 Triton 原始碼**。用來服務 K3 的公開 Triton 實作主要位於
[SGLang 的 KDA 路徑](https://github.com/sgl-project/sglang/tree/fc9e1c8d296216ff1e216dfbe7286ef392448d28/python/sglang/srt/layers/attention/linear)
與 [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention/tree/fa06b39153e09b2347387fff990d7d2d07ddd04b/fla/ops/kda)。
[FlashKDA](https://github.com/MoonshotAI/FlashKDA) 是 **CUDA/CUTLASS，不是 Triton**。
SGLang 可以使用 Triton KDA 路徑作為可攜式替代方案，但這不會讓 FlashKDA 本身變成 Triton。

!!! note "本章使用的版本"

    原始碼連結固定在 Quark release/0.12 時期的程式碼、SGLang commit
    `fc9e1c8` 與 FLA commit `fa06b39`。這些專案變動很快，請依環境中安裝的版本核對 signature。

## 2. 追蹤一個 K3 Request 如何通過 SGLang

Prompt 進入 SGLang 後會依序走過以下流程：

```text
schedule and allocate cache slots
  → preprocess projections, SiTU, and norms
  → run KDA or MLA attention
  → route tokens to MoE experts
  → quantize, move, and combine data as required
  → sample the next token
  → commit accepted cache and recurrent state
```

Prefill 會處理許多 prompt token。Decode 會為每個 active request 處理一個新 token。
Speculative decoding 會處理多個候選 token，但只 commit 被接受的 prefix。
這些模式共用模型權重，卻不一定共用相同的 kernel。

K3 共有 93 層：69 個 KDA 層與 24 個 gated Multi-head Latent Attention
（MLA）層。它也是 sparse MoE，具有 896 個 routed expert，每個 token 會選擇
16 個 expert。因此所需工作遠超過一個 attention kernel。

### 2.1 SGLang 的 Triton 使用情境分類

下方架構涵蓋與 K3 相關的 kernel 類別，以及 SGLang 中相關的 Triton 機制。

| 類別 | SGLang 中的工作 | Triton 的優勢 |
|---|---|---|
| 前處理、activation 與 norm | 短 convolution、SiTU-GLU、Q/K L2 normalization、RMSNorm、gated output norm、gate/decay/beta 轉換 | 融合受記憶體頻寬限制的 elementwise 工作與 reduction |
| Cache 與 state 搬移 | paged KV 寫入、recurrent-state gather/scatter、page 搬移、accepted-token map | 直接表達 mask、stride 與 pointer indirection |
| KDA decode | recurrent state update 與 output projection | 固定大小的 state tile 很適合 block tensor |
| KDA chunkwise prefill | 區域 QK/KK 乘積、triangular solve、chunk state passing | 規則的矩陣 tile、scan 與 variable-length mask |
| Speculative state | snapshot、parent-state selection、ReplaySSM ring-buffer replay | 只搬移或 replay 可能被接受的 state |
| MLA attention | RoPE、Q/K norm、paged cache、split-KV attention、output merge | 融合自訂 cache layout 與 online reduction |
| MoE routing 與資料搬移 | top-k、expert alignment、token permutation、grouped expert GEMM、weighted combine | 在規則的矩陣運算周圍處理不規則 metadata |
| 量化 | per-token/group FP8 或 INT8、AWQ dequantization、MXFP4/MXFP8/NVFP4 helper | 結合 reduction、scaling、conversion 與 packing |
| Sampling | top-p/min-p renormalization、rejection sampling、tree reconstruction | 避免在許多短小 tensor 操作之間來回搬移資料 |
| 通訊 | 融合 all-reduce/residual 工作、symmetric-memory helper、sequence-parallel metadata | 將區域轉換與分散式資料搬移融合 |

SGLang 可能為每一列選擇不同的 backend。它的 kernel namespace 也包含
CUDA JIT、CuTe DSL、AITER、Helion、FlashInfer 與函式庫呼叫。
將效能歸因於 Triton 前，務必檢查實際選到的 backend 與 dispatch 條件。

## 3. 階段 1：前處理、SiTU 與 Norm

第一個實用觀念是 fusion。K3 前處理包含 projection、normalization、
gate 轉換與有界 activation。每項操作都很簡單，但寫出每個中間 tensor 的成本很高。

### 3.1 融合 SiTU-GLU

K3 以有界的 SiTU activation 取代一般的 SwiGLU。對 gate 輸入 \(g\) 與 up 輸入 \(u\)，
教學 kernel 會計算

$$
y =
\left[\beta_1 \tanh(g/\beta_1)\sigma(g)\right]
\left[\beta_2 \tanh(u/\beta_2)\right],
\qquad \beta_1=4,\quad \beta_2=25.
$$

| 符號 | 意義 |
|---|---|
| \(g,u\) | projected input 的兩半 |
| \(\sigma\) | Sigmoid |
| \(\beta_1,\beta_2\) | K3 設定使用的界限 |
| \(y\) | 融合後的 activation 輸出 |

[`situ_glu.py`](examples/15-triton-k3/situ_glu.py) 會載入兩半，並只寫入一次結果：

```python
gate = tl.load(x_ptr + offsets, mask=mask, other=0.0)
up = tl.load(x_ptr + n + offsets, mask=mask, other=0.0)
bounded_gate = BETA1 * (2 / (1 + tl.exp(-2 * gate / BETA1)) - 1)
bounded_up = BETA2 * (2 / (1 + tl.exp(-2 * up / BETA2)) - 1)
out = bounded_gate * (1 / (1 + tl.exp(-gate))) * bounded_up
```

這很適合使用 Triton，因為分開的 PyTorch 操作會讀寫大型暫存 tensor。
SGLang 目前也有非 Triton 的 SiTU 路徑。公式是 K3 專用，
實作選擇則取決於 runtime 與硬體。

### 3.2 只在資料流允許時融合 Norm

KDA 會在 recurrent update 前處理 Q、K、gate、decay 與 beta。
Q/K L2 normalization 會沿 head 維度做 reduction，RMSNorm 則沿 hidden 維度做 reduction。
兩者都採用相同的基本 Triton 流程：

1. 以 tile 為單位載入一列；
2. 以 FP32 累加平方值；
3. 乘上平方根倒數；
4. 套用任何 weight、gate 或 output 轉換；
5. 寫入一次。

Fusion 可以減少記憶體流量，但也可能增加 register 使用量。
如果融合後的 tile 會 spill、另一個 backend 已產生 normalized value，
或 dispatch 邊界需要中間 tensor，就應保留獨立的 norm kernel。
第一個練習會將 Q/K normalization 加入已測試的 KDA kernel。

## 4. 格式背景：Quark MXFP4

K3 服務通常從轉換後的低精度權重開始。Quark 很適合用來了解背景，
因為它定義 checkpoint 表示方式；它並不是 SGLang scheduler，也不是 KDA 實作。

Quark 的公開 Triton 程式碼涵蓋 OCP microscaling 與 FP8 轉換。其
[MX 實作](https://github.com/amd/Quark/blob/f7d8cefc7a6c973ff90cb87a6b154cbe3cc9aef2/quark/torch/kernel/mx/triton.py)
對 MXFP4 等格式使用 32-value block：

1. 對 block 做 reduction，取得最大絕對值；
2. 導出共用的 E8M0 二次方 scale；
3. 每個值除以該 scale；
4. 四捨五入為 E2M1 值；
5. 每兩個四位元值打包進一個 byte；
6. 將 scale swizzle 成 consumer 所需的 layout。

[`mxfp4_qdq.py`](examples/15-triton-k3/mxfp4_qdq.py) 實作步驟 1–4，
並回傳反量化後的值。它使用有限的 E2M1 magnitude
\(\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}\)。

!!! warning "教學用 QDQ 不是 checkpoint converter"

    相容的 checkpoint 必須符合 Quark 確切的 scale rounding、NaN 與 zero 規則、
    nibble 順序、padding 及 scale swizzle。需要互通性時，請使用 Quark 匯出的
    `qdq_mxfp4_triton` 或 `dq_mxfp4_triton`。本範例只抽出數值概念，
    讓它能在沒有完整模型時測試。

Quark 也有 Triton E5M3 轉換路徑，明確處理 subnormal 與
round-to-nearest-even。這些 kernel 屬於最佳化與 fake-quantization。
Quark 不負責 KDA、MoE routing 或 production expert GEMM。

## 5. 階段 2：搬移 Cache 與 Recurrent State

離線範例會依 batch 順序保存 state，但 serving runtime 無法這麼做。
Request 會在不同時間抵達與結束，因此 scheduler 會將每個 active request
對應到 persistent cache slot。

[`state_cache.py`](examples/15-triton-k3/state_cache.py) 展示核心模式：

```python
row = tl.program_id(0)
slot = tl.load(slots_ptr + row)
cols = tl.arange(0, BLOCK)
values = tl.load(source_ptr + row * width + cols, mask=cols < width)
tl.store(cache_ptr + slot * width + cols, values, mask=cols < width)
```

同樣的模式也出現在 KV cache、recurrent KDA state、Mamba state
與 speculative-decoding metadata。Production kernel 還會加上 page offset、
layer/head stride、量化儲存，以及來自 request metadata 的邊界。

兩次寫入同一個 slot 會發生 race，因此 wrapper 會拒絕重複 slot。
Production scheduler 會保證唯一性，或定義 atomic／有序更新。

KV cache 與 recurrent state 不能互換。MLA 會儲存依 token 編索引的 key 與 value，
通常放在 page 中。KDA 則會把固定 shape 的矩陣 state 從一個 token 傳到下一個 token。
兩者都需要 request-to-slot indirection，但 allocation、lifetime 與 rollback 規則不同。

## 6. 階段 3A：一次 Recurrent KDA Decode

對一個 token，簡化的 KDA state update 為

$$
D_t = \operatorname{Diag}(\alpha_t)S_{t-1},
$$

$$
r_t = v_t - D_t^\mathsf{T}k_t,\qquad
S_t = D_t + \beta_t k_t r_t^\mathsf{T},\qquad
o_t = S_t^\mathsf{T}q_t.
$$

| 符號 | Shape | 意義 |
|---|---:|---|
| \(q_t,k_t,\alpha_t\) | \(K\) | query、key 與 channel decay |
| \(v_t,r_t,o_t\) | \(V\) | value、prediction residual 與 output |
| \(S_t,D_t\) | \(K\times V\) | recurrent state 及其衰減值 |
| \(\beta_t\) | 此教學形式中的 scalar | 更新強度 |

[`kda_step.py`](examples/15-triton-k3/kda_step.py) 會啟動二維 grid。
第一軸選擇 batch × head，第二軸選擇一個 \(V\)-tile。
每個 program 會載入完整的 \(K\) 維度，以及自己負責的 state column：

```python
decayed = state * alpha[:, None]
residual = v - tl.sum(decayed * k[:, None], axis=0)
updated = decayed + k[:, None] * (beta * residual)[None, :]
out = tl.sum(updated * q[:, None], axis=0)
```

State 保持 FP32，因為 rounding error 會在 recurrence 中累積。
調校過的 kernel 可以讓 Q、K 與 V 使用較低 precision。實際 K3 路徑還會融合
input extraction、Q/K L2 normalization、bounded decay、beta activation 與 output gating。

此教學形式會清楚呈現 recurrence，但不代表每個 production K3 decode 都使用 Triton。
SGLang 可以依 device、mode、dtype 與已安裝的 extension，dispatch 到 Triton KDA 路徑、
FlashKDA、CuTe DSL、FlashInfer、Helion 或其他支援的 backend。

## 7. 階段 3B：Chunkwise Prefill 與 Speculative State

### 7.1 Decode 是 Recurrent 運算

Decode 會接收一個新 token。上述 recurrent kernel 很合適：
讀取一份 state、更新並寫回一份 state。Continuous batching
會為每個 request 提供 cache slot。

### 7.2 Prefill 採用 Chunkwise 運算

Prefill 一次會接收許多 prompt token。逐 token 套用 decode 會讓 GPU 使用率過低。
Chunkwise KDA 改為：

1. 計算 gate prefix sum；
2. 形成 chunk 內的 QK 與 KK 乘積；
3. 建立 WY representation 並解 triangular system；
4. 在 chunk 間傳遞 state；
5. 重建 output。

[FLA `ops/kda`](https://github.com/fla-org/flash-linear-attention/tree/fa06b39153e09b2347387fff990d7d2d07ddd04b/fla/ops/kda)
下的公開實作展示完整演算法，包括 variable-length packed input。
將那些程式改寫成短教學 kernel 反而會隱藏重要 invariant，
因此本章將 production API 留作進階練習。

Chunk size 是調校選擇，不是模型變更。小 chunk 的 state-passing overhead 較高。
大 chunk 會增加暫存空間，也可能降低 occupancy。Packed variable-length prompt
也需要 offset 與 mask，確保一個 request 絕不會讀到另一個 request 的 token。

### 7.3 Speculative Decoding 與 ReplaySSM

Draft token 可能遭拒。驗證完成前，kernel 不得覆寫 committed state。
常見設計會保留中間 snapshot、為 verification tree 的每個 node 選擇 parent state，
並只將 accepted token 合併進主要 state。SGLang 的 ReplaySSM 路徑會加入 ring buffer，
避免重新計算每份 state。Scheduler 會提供 parent 與 accepted-token metadata；
kernel 則 gather、replay 並 commit 對應的 state。

安全規則很簡單：speculative 工作可以更新暫存 state，
但只有 accepted token 可以更新 persistent KDA state 或 MLA KV cache。

## 8. 階段 3C：MLA Attention

K3 有 24 層使用 gated MLA，而不是 KDA。這條路徑採用不同的 state 模型：

1. project 並 normalize Q 與 K；
2. 對 positional component 套用 RoPE；
3. 將 compressed KV data 寫入 paged cache；
4. 讀取各 request 選定的 page；
5. 執行 prefill 或 split-KV decode attention；
6. 合併 partial output，並套用 output gate。

Triton 很適合 cache transform、RoPE、normalization、online softmax、
split reduction 與 output fusion。調校過的 MLA attention core 也可能改用
FlashInfer、FlashMLA、CuTe DSL、AITER 或其他架構專用實作。
KDA state 是 recurrent matrix；MLA state 則是依 token 編索引的 paged cache。
SGLang 的 hybrid attention layer 必須同時管理兩者。

## 9. 階段 4：MoE Routing 與資料搬移

Attention 完成後，K3 會將每個 token 分派給 896 個 routed expert 中的 16 個。
完整流程遠超過一次 grouped matrix multiplication：

```text
router logits
  → top-k expert selection and weights
  → count and align tokens by expert
  → permute or dispatch tokens
  → expert GEMM 1
  → SiTU and optional activation quantization
  → expert GEMM 2
  → weighted combine and restore token order
```

Triton 很適合 top-k helper、expert alignment、permutation、quantization、
grouped GEMM 與 weighted combine。在 expert parallel 模式下，
dispatch 與 combine 也會跨裝置進行。Metadata 並不規則，
但 token 分組後，每個 expert 的矩陣運算都很規則。

Production MoE GEMM 通常使用 CUDA/CUTLASS、CuTe DSL、AITER、CK
或供應商函式庫 kernel。這些路徑可以使用架構專用的矩陣指令、layout、
persistent scheduling 與經過調校的低精度 epilogue。
Triton 仍適合可攜式路徑，以及這些 GEMM 周圍的資料搬移。

## 10. 階段 5：服務流程中的量化

量化會出現在數個邊界：

- checkpoint conversion 會打包 weight 與 scale；
- weight loader 會保留確切的 nibble 與 scale layout；
- activation kernel 會計算 per-token 或 per-group scale；
- GEMM 會使用 MXFP4、MXFP8、FP8、INT8、AWQ 或其他支援格式；
- cache kernel 可能會將 value 轉換為量化儲存格式。

不要把名稱中含有「FP4」的格式全都視為相同。
Scale type、group size、rounding、packing、padding 與 swizzle 都是介面的一部分。
第 4 節的 Quark 範例用來說明 MXFP4 數值運算；
production consumer 則會決定必要的實體 layout。

## 11. 階段 6：Sampling 與通訊

執行完最後一個模型層後，SGLang 在 GPU 上仍有工作要做。
Sampling 可以套用 temperature 與 penalty、重新正規化 top-p 或 min-p distribution、
抽取或拒絕 candidate，以及重建 speculative tree。這些操作會使用短 reduction、
mask、scan 與 gather，因此 Triton 可以取代一連串小型 framework launch。

Tensor、data、expert 與 sequence parallelism 都會增加通訊需求。
Triton 可以準備 symmetric-memory buffer、將 residual 或 scaling 工作與區域 copy 融合，
並處理 sequence-parallel metadata，但它不會取代網路傳輸。
裝置間的資料仍由 NCCL、RCCL 或其他 communication backend 搬移。

同樣的 Triton 模式會在整個 request 中反覆出現：masked load、row reduction、
online reduction、pointer indirection、stable permutation、tile GEMM、scan 與 fusion。

## 12. Backend Dispatch：路徑為何可能不使用 Triton

SGLang 會在 runtime 或模型設定期間選擇 operator 實作。決策可能取決於：

- CUDA 或 ROCm、GPU 架構，以及已安裝的 extension；
- prefill、decode 或 speculative verification mode；
- dtype、quantization scheme、head size、page size 與 batch shape；
- graph-capture 與 distributed-execution 需求；
- 經過調校的 backend 是否支援確切的模型功能。

CUDA/CUTLASS 常用於 NVIDIA 專用的 tensor-core layout 與手動調校的 pipeline。
CuTe DSL 能明確控制 layout 與 MMA，適合進行類似的特化。
AITER 封裝針對 AMD 調校的 CK、HIP、assembly、Triton 與其他實作。
FlashInfer 與供應商函式庫則提供持續維護的專用 operator。
當可攜性、快速迭代、fusion 與自訂資料搬移比極致的裝置專用最佳化更重要時，
Triton 最能發揮優勢。

因此：

- **FlashKDA 是 CUDA/CUTLASS，不是 Triton。**
- AITER 是 dispatch 與 operator 函式庫；呼叫 AITER 不代表選到的 kernel 是 Triton。
- `.py` wrapper 可能啟動 CUDA、CuTe DSL、AITER 或函式庫 kernel。
- 效能結果必須標明選到的 backend、dtype、shape 與硬體。

請從模型層一路追蹤 backend，經過 registry、特定 mode 的實作，
直到最後的 kernel launch。不要根據 Python 檔名，
或 SGLang 其他位置出現 Triton，就推斷此處使用 Triton。

## 重點整理

1. Quark、K3 與 SGLang 位於技術堆疊的不同層。
2. K3 官方儲存庫不含 Triton 原始碼，而 FlashKDA 是 CUDA/CUTLASS 專案。
3. KDA decode 是 recurrent state update；KDA prefill 是 chunkwise matrix algorithm。
4. MLA 使用 paged token cache，KDA 則帶著固定 shape 的 recurrent state。
5. Triton 適用於整個服務流程，但 SGLang dispatch 可能選擇更專用的 backend。

## 13. Production 整合

教學 kernel 必須先明確定義介面契約，才能成為 SGLang backend：

1. 定義接受的 device、dtype、shape、mode 與 quantization format；
2. 讓 wrapper 驗證 stride、alignment、cache slot 與 metadata；
3. 在明確的 dispatch 條件後註冊實作；
4. 為不支援的 input 保留已知正確的 fallback；
5. 分別連接 prefill、decode 與 speculative-state 介面；
6. 保持 transactional state：只 commit accepted token；
7. 在 log 或 metric 中揭露選到的 backend；
8. 將 SGLang、Triton、PyTorch 與 ROCm/CUDA 版本一起固定。

除非經過驗證的格式允許其他選擇，否則 committed recurrent state 應保持 FP32。
請將 Quark packing 與 scale swizzling 視為資料格式的一部分。
NVIDIA 與 AMD 必須分別測試：原始碼可攜不代表 tile shape、compiler output
或 tuning 也相同。請使用 K3 授權中的名稱；「開放權重」比未加限定的「開源」精確。

## 14. 測試

從儲存庫根目錄執行現有範例：

```bash
cd tutorials/examples/15-triton-k3
python3 test_model_kernels.py
```

沒有 GPU 時，script 會啟用 `TRITON_INTERPRET=1`。它會檢查非二次方大小的 SiTU input、
已變更的 KDA state 與 output、全零與部分 MX block，以及 indexed-cache round trip。

Production test 還應加入：

- 奇數 head 與 hidden size、partial page、empty mask 與 zero-token 工作；
- packed variable-length prompt 與混合的 prefill/decode batch；
- 重複或無效的 cache slot，以及 request cancellation；
- 接受零個、部分與全部 token 的 speculative branch；
- quantization boundary value、packing order、padding 與 scale layout；
- 每個啟用 backend 的結果一致性，以及不支援 shape 時的 fallback；
- multi-rank expert dispatch、combine 與 communication failure。

計時前請先檢查數值結果。重新結合的 reduction 需要明確的 tolerance，
而 recurrent test 應比較每個 intermediate state，不能只比較最後一個 token。

## 15. 效能分析

測量前請先暖機 JIT 與 cache。在 NVIDIA 上，可以用以下指令檢查小型 KDA 範例：

```bash
ncu -k regex:kda_step_kernel python3 test_model_kernels.py
```

在 AMD 上請使用對應的 ROCm profiler。除了個別 kernel，
也要分析完整的 SGLang request。請分別評估 prefill throughput、
decode latency 與 speculative verification，因為它們的 shape 與 bottleneck 不同。

至少記錄：

- 選到的 backend 與確切的軟體 revision；
- device、dtype、quantization、batch、sequence 與 cache shape；
- kernel time、launch count、memory traffic、occupancy 與 spill；
- 分散式執行的 communication time 與 overlap；
- 暖機後的端對端 latency 與 throughput。

如果較快的獨立 kernel 需要額外的 conversion、cache 搬移、synchronization
或 dispatch overhead，端對端效能反而可能更差。

## 練習

1. 將 Q/K L2 normalization 融合進 `kda_step_kernel`，sum 保持 FP32。
2. 擴充 state cache，加入獨立的 request、layer 與 head stride。
3. 每個 byte 打包兩個 E2M1 code，再與 Quark 0.12 的 byte order 比較。
4. 在 KDA reference 外加入 sequence loop，驗證每個 intermediate state，而不只最終 output。
5. 在 packed variable-length prompt 上呼叫固定版本的 FLA chunkwise KDA API，並與 recurrent decode 比較。
6. 設計 ReplaySSM test，讓 accepted prefix 長度分別為 0、1 與完整 draft length。
7. 畫出 continuous batching 下，一個 KDA layer 與一個 MLA layer 的 cache layout。
8. 在你的機器上追蹤 SGLang 的 K3 backend selection。列出哪些 operator 使用
   Triton、CUDA/CUTLASS、CuTe DSL、AITER、FlashInfer 或供應商函式庫，
   並解釋每個 fallback。
