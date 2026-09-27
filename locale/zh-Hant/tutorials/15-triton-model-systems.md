# 15 – Quark、Kimi K3 與 SGLang 中的 Triton

> **第四部 · 可攜式模型 Kernel** · 先備知識：[14 – Triton](14-triton.md)
> 與 [13 – Softmax、LayerNorm 和 FlashAttention](13-softmax-attention.md) ·
> 程式：[`examples/15-triton-k3/`](examples/15-triton-k3/test_model_kernels.py) ·
> 下一章：[05 – CDNA3 與 MFMA](05-amd-cdna3-mfma.md)

本章從小型 Triton kernel 進入真實模型系統，追蹤三個各司其職的專案：

- **AMD Quark** 轉換並量化模型。它的 Triton kernel 實作 MXFP4、FP8 等數值格式。
- **Kimi K3** 是開放權重的 mixture-of-experts 模型，主要的新 operator 是 Kimi Delta Attention（KDA）。
- **SGLang** 負責模型服務，並將 Triton 與 CUDA、CuTe DSL、AITER、FlashInfer 及供應商函式庫結合。SGLang 裡的檔案不一定是 Triton。

這裡的範例都小到可在 Triton 的 CPU interpreter 中執行。它們不複製綁定特定版本的實作細節，但會教授與 production kernel 相同的資料流。

**你將學會**

- Quark、K3 serving 與 SGLang 在何處使用 Triton；
- SiTU-GLU、MXFP4 風格量化、indexed cache 與 recurrent KDA 的運作方式；
- prefill、decode、continuous batching 與 speculative decoding 如何改變 kernel；
- 如何判斷範例是忠實的格式實作、教學模型，還是 production backend。

## 1. 釐清專案界線

| 層級 | 專案 | 職責 |
|---|---|---|
| 模型定義 | [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) | 架構、權重與技術報告 |
| 模型最佳化 | [AMD Quark](https://github.com/amd/Quark/tree/release/0.12) | 量化、校正與 checkpoint 轉換 |
| Serving runtime | [SGLang](https://github.com/sgl-project/sglang) | Batching、cache 所有權、排程與 backend dispatch |
| 可攜式 kernel | Triton / FLA | 量化、attention、state update、routing 與資料搬移 |
| 硬體專用 kernel | AITER、CuTe DSL、FlashKDA、FlashInfer | 支援裝置與 shape 的較快路徑 |

K3 官方儲存庫不含 Triton 原始碼。用於服務 K3 的公開 Triton 實作主要位於
[SGLang 的 KDA 路徑](https://github.com/sgl-project/sglang/tree/fc9e1c8d296216ff1e216dfbe7286ef392448d28/python/sglang/srt/layers/attention/linear)
與 [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention/tree/fa06b39153e09b2347387fff990d7d2d07ddd04b/fla/ops/kda)。
[FlashKDA](https://github.com/MoonshotAI/FlashKDA) 是 CUDA/CUTLASS kernel，不是 Triton kernel；Triton 是它的可攜式 fallback。

!!! note "本章採用的版本"

    原始碼連結固定在 Quark 0.12 時期、SGLang commit `fc9e1c8` 與 FLA commit `fa06b39`。這些專案變動很快，請依安裝於環境中的版本核對 signature。

## 2. K3 Operator 地圖

K3 有 93 層：69 個 KDA 層與 24 個 gated Multi-head Latent Attention（MLA）層。它也是 sparse MoE，具有 896 個 routed expert，每個 token 選擇 16 個。因此工作不只是一個 attention kernel。

| 區域 | 常見 kernel | Triton 的優勢 |
|---|---|---|
| 輸入準備 | 短 causal convolution、Q/K normalization、gate 與 beta 轉換 | 多個小操作可合為一次 launch |
| KDA decode | recurrent state update 與 output projection | 固定大小的 state tile 很適合 block tensor |
| KDA prefill | 區域 QK/KK 乘積、triangular solve、chunk state propagation | 規則的 dense tile 與 scan |
| MLA | RoPE、QK norm、paged cache、split-KV attention、output merge | 自訂 cache layout 與 fusion |
| MoE | top-k、token permutation、grouped expert GEMM、weighted combine | 在規則 GEMM 周圍處理不規則 metadata |
| 量化 | MXFP4/MXFP8/FP8 轉換與 scaling | reduction 加上位元／資料轉換 |
| Activation 與 norm | SiTU-GLU、RMSNorm、gated output norm | 將受頻寬限制的操作鏈融合成一個 kernel |
| Serving metadata | indexed state read/write、page table、accepted-token map | mask 與 pointer arithmetic 寫法精簡 |

SGLang 可能為每一列選不同 backend。其 kernel namespace 也包含 CUDA JIT、CuTe DSL、AITER、Helion、FlashInfer 與函式庫呼叫。將效能歸因於 Triton 前，務必檢查實際選到的 backend 與 dispatch 條件。

## 3. 案例一：融合 SiTU-GLU

K3 以有界 activation 取代一般 SwiGLU。對 gate 輸入 \(g\) 與 up 輸入 \(u\)，教學 kernel 計算

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

[`situ_glu.py`](examples/15-triton-k3/situ_glu.py) 載入兩半，並只寫入一次結果：

```python
gate = tl.load(x_ptr + offsets, mask=mask, other=0.0)
up = tl.load(x_ptr + n + offsets, mask=mask, other=0.0)
bounded_gate = BETA1 * (2 / (1 + tl.exp(-2 * gate / BETA1)) - 1)
bounded_up = BETA2 * (2 / (1 + tl.exp(-2 * up / BETA2)) - 1)
out = bounded_gate * (1 / (1 + tl.exp(-gate))) * bounded_up
```

這很適合用 Triton，因為分開的 PyTorch 操作會反覆讀寫大型暫存 tensor。SGLang 目前也有非 Triton 的 SiTU 路徑。公式屬於 K3；實作選擇則取決於 runtime 與硬體。

## 4. 案例二：理解 Quark 的 MX Kernel

Quark 的公開 Triton 程式涵蓋 OCP microscaling 與 FP8 轉換。其
[MX 實作](https://github.com/amd/Quark/blob/f7d8cefc7a6c973ff90cb87a6b154cbe3cc9aef2/quark/torch/kernel/mx/triton.py)
對 MXFP4 等格式使用 32-value block：

1. 對 block 做 reduction，取得最大絕對值；
2. 導出共用的 E8M0 二次方 scale；
3. 每個值除以該 scale；
4. 四捨五入成 E2M1 值；
5. 每兩個四位元值打包進一個 byte；
6. 將 scale swizzle 成 consumer 所需的 layout。

[`mxfp4_qdq.py`](examples/15-triton-k3/mxfp4_qdq.py) 實作步驟 1–4，並回傳反量化後的值。它使用有限 E2M1 magnitude \(\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}\)。

!!! warning "教學用 QDQ 不是 checkpoint converter"

    相容 checkpoint 必須符合 Quark 確切的 scale rounding、NaN 與 zero 規則、nibble 順序、padding 及 scale swizzle。需要互通性時，請使用 Quark 匯出的 `qdq_mxfp4_triton` 或 `dq_mxfp4_triton`。本範例只抽出數值概念，讓它能在沒有完整模型時測試。

Quark 也有 Triton E5M3 轉換路徑，明確處理 subnormal 與 round-to-nearest-even。這些 kernel 屬於最佳化與 fake-quantization。Quark 不負責 KDA、MoE routing 或 production expert GEMM。

## 5. 案例三：在 Serving Cache 中搬移 State

離線範例依 batch 順序保存 state；serving runtime 做不到這點。Request 會在不同時間抵達與結束，因此 scheduler 會將每個 active request 對應到 persistent cache slot。

[`state_cache.py`](examples/15-triton-k3/state_cache.py) 展示核心模式：

```python
row = tl.program_id(0)
slot = tl.load(slots_ptr + row)
cols = tl.arange(0, BLOCK)
values = tl.load(source_ptr + row * width + cols, mask=cols < width)
tl.store(cache_ptr + slot * width + cols, values, mask=cols < width)
```

同樣模式也出現在 KV cache、recurrent KDA state、Mamba state 與 speculative-decoding metadata。Production kernel 還會加上 page offset、layer/head stride、量化儲存，以及來自 request metadata 的邊界。

兩次寫入同一 slot 會發生 race，因此 wrapper 會拒絕重複 slot。Production scheduler 則會保證唯一性，或定義 atomic／有序更新。

## 6. 案例四：一次 Recurrent KDA Decode

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

[`kda_step.py`](examples/15-triton-k3/kda_step.py) 啟動二維 grid。第一軸選擇 batch × head，第二軸選擇一個 \(V\)-tile。每個 program 載入完整 \(K\) 維度與自己負責的 state column：

```python
decayed = state * alpha[:, None]
residual = v - tl.sum(decayed * k[:, None], axis=0)
updated = decayed + k[:, None] * (beta * residual)[None, :]
out = tl.sum(updated * q[:, None], axis=0)
```

State 保持 FP32，因為 rounding error 會反覆累積。調校過的 kernel 可讓 Q、K、V 使用較低 precision。真實 K3 路徑也會融合 input extraction、Q/K L2 normalization、bounded decay、beta activation 與 output gating。

## 7. Decode、Prefill 與 Speculation 是不同演算法

### 7.1 Decode

Decode 接收一個新 token。上述 recurrent kernel 很合適：讀取一份 state、更新並寫回。Continuous batching 為每個 request 提供 cache slot。

### 7.2 Prefill

Prefill 一次接收許多 prompt token。逐 token 套用 decode 會讓 GPU 使用率過低。Chunkwise KDA 改為：

1. 計算 gate prefix sum；
2. 形成 chunk 內 QK 與 KK 乘積；
3. 建立 WY representation 並解 triangular system；
4. 在 chunk 間傳播 state；
5. 重建 output。

[FLA `ops/kda`](https://github.com/fla-org/flash-linear-attention/tree/fa06b39153e09b2347387fff990d7d2d07ddd04b/fla/ops/kda)
下的公開實作展示完整演算法，包括 variable-length packed input。把那些程式改寫成短教學 kernel 反而會隱藏重要 invariant，因此本章將 production API 留作進階練習。

### 7.3 Speculative decoding

Draft token 可能遭拒。驗證完成前，kernel 不得覆寫 committed state。常見設計會保留中間 snapshot、為 verification tree 的每個 node 選擇 parent state，並只將 accepted token 合併進主要 state。SGLang 的 ReplaySSM 路徑再加入 ring buffer，避免重新計算每份 state。

## 8. 更廣泛的 SGLang Triton 地圖

K3 只使用 SGLang 的一部分。同樣的 Triton 技能也適用於 runtime 的其他 operator：

| 群組 | 代表用途 |
|---|---|
| Attention | paged/split-KV decode、prefill、GQA/MQA、sliding window、sink、softcap、MLA |
| KV cache | indexed write、gather/scatter、page movement、quantized-cache conversion |
| MoE | top-k、expert alignment、token permutation、grouped GEMM、weighted combine |
| 量化 | per-token/group FP8、INT8、AWQ dequantization、MXFP8 與 NVFP4 helper |
| Norm 與 activation | RMSNorm、residual + norm、gated norm、SiLU/GELU-and-mul |
| Mamba/SSM | causal convolution、chunk BMM、scan、state passing 與 cache update |
| Sampling | top-p/min-p renormalization、rejection sampling 與 tree reconstruction |
| 通訊 | 融合 all-reduce/residual helper 與 sequence-parallel metadata |

反覆出現的模式比檔案數量重要：masked load、row reduction、online reduction、pointer indirection、stable permutation、tile GEMM 與 kernel fusion。

## 9. 執行與驗證

```bash
cd tutorials/examples/15-triton-k3
python3 test_model_kernels.py
```

沒有 GPU 時，script 會啟用 `TRITON_INTERPRET=1`。它會檢查非二次方尺寸、被修改的 KDA state、全零 MX block、部分 MX block，以及 indexed-cache round trip。

在 GPU 上，同時 profile kernel 與周邊 runtime：

```bash
ncu -k regex:kda_step_kernel python3 test_model_kernels.py
```

計時前先檢查數值結果。重新結合的 reduction 與 recurrent state update 需要明確容許誤差。比較 latency 前務必先暖機 JIT cache。

## 10. Production 檢查清單

1. 將 SGLang、Triton、PyTorch 與 ROCm/CUDA 版本一起固定。
2. 記錄 dispatch 選到哪個 backend；「SGLang 效能」不代表只使用 Triton。
3. 除非經驗證的格式另有規定，否則 committed recurrent state 保持 FP32。
4. 測試奇數 shape、partial page、empty mask 與 variable-length batch。
5. 將 Quark 的 packing 與 swizzling 視為資料格式的一部分。
6. 分別測試 CUDA 與 AMD。原始碼可攜不代表 tile shape 或 tuning 相同。
7. 使用 K3 授權的名稱。「開放權重」比未加限定的「開源」精確。

## 重點整理

1. Quark、K3 與 SGLang 位於 stack 的不同層。
2. Triton 最擅長 reduction、資料轉換與 fusion 和規則 tile 相交之處。
3. KDA decode 是 recurrent state update；KDA prefill 是 chunkwise matrix algorithm。
4. Serving 會加入 cache indirection、variable length 與 transactional state。
5. 小型教學 kernel 必須說明它省略了 production format 或 backend 的哪些部分。

## 練習

1. 將 Q/K L2 normalization 融合進 `kda_step_kernel`，sum 保持 FP32。
2. 擴充 state cache，加入獨立的 request、layer 與 head stride。
3. 每個 byte 打包兩個 E2M1 code，再與 Quark 0.12 的 byte order 比較。
4. 在 KDA reference 外加入 sequence loop，驗證每個中間 state，而不只最終 output。
5. 在你的機器上追蹤 SGLang 的 K3 backend selection，列出哪些 operator 使用 Triton、AITER、CUDA 或函式庫。
