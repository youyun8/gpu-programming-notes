# 15 – SGLang 中的 Triton – 以 Kimi K3 推論服務為例

> **第六部 · Kimi K3 案例** · 先備知識：
> [14 – Triton：從第一個 Kernel 到生產環境](14-triton.md) ·
> 程式：[`examples/15-triton-k3/`](examples/15-triton-k3/test_model_kernels.py) ·
> 下一章：[16 – AITER 中的 FlyDSL：在 AMD GPU 上執行 Kimi K3](16-aiter-flydsl-kimi-k3.md)

本章從小型 Triton 程式出發，一路走到完整的推論服務流程。系統是 SGLang，
端對端追蹤的案例是 Kimi K3。我們先看一個融合後的激活函數，再跟著狀態與
token 依序經過注意力、專家（expert）、取樣、通訊，最後看生產環境如何分派後端。

這三個專案各有分工：**Kimi K3** 定義開放權重模型；**AMD Quark** 負責轉換與量化
checkpoint；**SGLang** 負責推論服務，包括排程、快取、分散式執行與後端選擇。
SGLang 會同時用到 Triton、CUDA、CuTe DSL、AITER、FlashInfer 與硬體廠商的函式庫，
所以「被 SGLang 呼叫的 kernel」不一定就是 Triton kernel。

本章範例都很小，可以直接在 Triton 的 CPU 直譯器上執行。它們不照抄特定版本的
實作細節，但資料流與生產環境的 kernel 相同。

**你將學會**

- SGLang 中 Triton 各類用途的全貌；
- SiTU-GLU、MXFP4 式量化、索引式快取與遞迴式 KDA 如何運作；
- 為什麼 decode、分段式 prefill 與推測解碼需要不同的狀態演算法；
- MLA、MoE、取樣與通訊在一次 K3 請求中各占哪個位置；
- 為什麼生產環境有時選擇 CUDA、CuTe DSL、AITER 或廠商函式庫，而不是 Triton。

## 1. 各專案的分工與固定版本

| 層級 | 專案 | 職責 |
|---|---|---|
| 模型定義 | [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) | 架構、權重與技術報告 |
| 模型最佳化 | [AMD Quark](https://github.com/amd/Quark/tree/release/0.12) | 量化、校正與 checkpoint 轉換 |
| 推論執行環境 | [SGLang](https://github.com/sgl-project/sglang) | 批次處理、快取管理、排程與後端分派 |
| 可攜式 kernel | Triton / FLA | 注意力、狀態更新、路由與資料搬移 |
| 硬體專用 kernel | AITER、CuTe DSL、FlashKDA、FlashInfer | 針對特定裝置與形狀的更快路徑 |

[K3 官方儲存庫](https://github.com/MoonshotAI/Kimi-K3) 提供架構、設定檔與模型程式碼，
但**裡面沒有任何 Triton 原始碼**。實際用來服務 K3 的公開 Triton 實作，主要在
[SGLang 的 KDA 路徑](https://github.com/sgl-project/sglang/tree/fc9e1c8d296216ff1e216dfbe7286ef392448d28/python/sglang/srt/layers/attention/linear)
與 [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention/tree/fa06b39153e09b2347387fff990d7d2d07ddd04b/fla/ops/kda)。
[FlashKDA](https://github.com/MoonshotAI/FlashKDA) 則是**以 CUDA/CUTLASS 寫成，不是 Triton**。
SGLang 可以改用 Triton 版 KDA 作為可攜的替代方案，但這不會讓 FlashKDA 本身變成 Triton。

!!! note "本章使用的版本"

    原始碼連結固定在 Quark release/0.12 時期的程式碼、SGLang commit `fc9e1c8`
    與 FLA commit `fa06b39`。這些專案更新很快，請以你環境中安裝的版本核對函式簽名。

## 2. 追蹤一次 K3 請求在 SGLang 中的旅程

一段提示詞（prompt）進入 SGLang 後，會依序經過：

```text
schedule and allocate cache slots
  → preprocess projections, SiTU, and norms
  → run KDA or MLA attention
  → route tokens to MoE experts
  → quantize, move, and combine data as required
  → sample the next token
  → commit accepted cache and recurrent state
```

Prefill 一次處理許多提示詞 token；decode 則替每個進行中的請求各處理一個新 token；
推測解碼（speculative decoding）一次處理好幾個候選 token，但只提交被接受的前綴。
這三種模式共用同一份權重，卻不一定共用同一個 kernel。

K3 共 93 層：69 層 KDA，24 層帶閘控的多頭潛在注意力（Multi-head Latent Attention，MLA）。
它同時也是稀疏 MoE：共有 896 個路由專家，每個 token 選用其中 16 個。
因此需要處理的工作遠不只一個注意力 kernel。

### 2.1 SGLang 中 Triton 的用途分類

下表整理與 K3 相關的 kernel 類別，以及 SGLang 裡對應的 Triton 機制。

| 類別 | SGLang 中的工作 | Triton 的優勢 |
|---|---|---|
| 前處理、激活與正規化 | 短卷積、SiTU-GLU、Q/K 的 L2 正規化、RMSNorm、帶閘控的輸出正規化、gate/decay/beta 轉換 | 把受頻寬限制的逐元素運算與歸約融合在一起 |
| 快取與狀態搬移 | 分頁式 KV 寫入、遞迴狀態的 gather/scatter、頁面搬移、已接受 token 的對照表 | 直接寫出遮罩、跨距與指標間接存取 |
| KDA decode | 遞迴狀態更新與輸出投影 | 固定大小的狀態分塊正好對應區塊張量 |
| KDA 分段式 prefill | 段內 QK/KK 乘積、三角求解、段間狀態傳遞 | 規則的矩陣分塊、掃描與變長遮罩 |
| 推測解碼的狀態 | 快照、挑選父狀態、ReplaySSM 環形緩衝區重播 | 只搬移或重播可能被接受的狀態 |
| MLA 注意力 | RoPE、Q/K 正規化、分頁快取、split-KV 注意力、輸出合併 | 融合自訂的快取配置與線上歸約 |
| MoE 路由與搬移 | top-k、依專家對齊、token 重排、分組專家 GEMM、加權合併 | 在規則的矩陣運算周圍處理不規則的中繼資料 |
| 量化 | 逐 token／逐組的 FP8 或 INT8、AWQ 反量化、MXFP4/MXFP8/NVFP4 輔助函式 | 把歸約、縮放、型別轉換與打包合在一起 |
| 取樣 | top-p/min-p 重新正規化、拒絕取樣、樹狀結構重建 | 避免在一連串小張量運算之間反覆讀寫 |
| 通訊 | 融合 all-reduce 與殘差運算、對稱記憶體輔助函式、序列平行的中繼資料 | 把本地轉換與分散式資料搬移融合 |

SGLang 可能替表中每一列選擇不同的後端。它的 kernel 命名空間裡還有 CUDA JIT、
CuTe DSL、AITER、Helion、FlashInfer 與各種函式庫呼叫。在把效能歸功於 Triton 之前，
一定要先確認實際選中的後端，以及選中它的分派條件。

## 3. 階段 1：前處理、SiTU 與正規化

第一個值得學的觀念是融合。K3 的前處理包含投影、正規化、閘控轉換與有界激活函數。
每一步都很簡單，但如果每一步都把中間張量寫回記憶體，成本會很高。

### 3.1 融合 SiTU-GLU

K3 用有界的 SiTU 激活函數取代常見的 SwiGLU。設 gate 輸入為 \(g\)、up 輸入為 \(u\)，
教學 kernel 計算

$$
y =
\left[\beta_1 \tanh(g/\beta_1)\sigma(g)\right]
\left[\beta_2 \tanh(u/\beta_2)\right],
\qquad \beta_1=4,\quad \beta_2=25.
$$

| 符號 | 意義 |
|---|---|
| \(g,u\) | 投影後輸入的前後兩半 |
| \(\sigma\) | Sigmoid 函數 |
| \(\beta_1,\beta_2\) | K3 設定檔中的上下界 |
| \(y\) | 融合後的激活輸出 |

[`situ_glu.py`](examples/15-triton-k3/situ_glu.py) 一次載入兩半，結果也只寫入一次：

```python
gate = tl.load(x_ptr + offsets, mask=mask, other=0.0)
up = tl.load(x_ptr + n + offsets, mask=mask, other=0.0)
bounded_gate = BETA1 * (2 / (1 + tl.exp(-2 * gate / BETA1)) - 1)
bounded_up = BETA2 * (2 / (1 + tl.exp(-2 * up / BETA2)) - 1)
out = bounded_gate * (1 / (1 + tl.exp(-gate))) * bounded_up
```

用 Triton 寫這個 kernel 很划算：若拆成多個 PyTorch 運算，就得反覆讀寫大型暫存張量。
不過 SGLang 目前也有不用 Triton 的 SiTU 路徑。公式是 K3 決定的，實作方式則由
執行環境與硬體決定。

### 3.2 只在資料流允許時才融合正規化

KDA 在遞迴更新之前，會先處理 Q、K、gate、decay 與 beta。Q/K 的 L2 正規化沿著
head 維度歸約，RMSNorm 則沿著隱藏維度歸約。兩者在 Triton 中的骨架相同：

1. 分塊載入一列；
2. 以 FP32 累加平方和；
3. 乘上平方根的倒數；
4. 套用權重、閘控或輸出轉換；
5. 只寫回一次。

融合可以省下記憶體流量，但也可能增加暫存器用量。遇到下列情況，就該保留獨立的
正規化 kernel：融合後的分塊會溢出暫存器；其他後端已經算好正規化結果；或是分派
邊界需要這個中間張量。第一題練習就是把 Q/K 正規化加進已測試過的 KDA kernel。

## 4. 格式背景：Quark 的 MXFP4

K3 推論服務通常從轉換好的低精度權重開始。了解 Quark 有助於理解背景，因為它定義了
checkpoint 的表示格式；但它既不是 SGLang 的排程器，也不是 KDA 的實作。

### 4.1 MXFP4 量化步驟

Quark 公開的 Triton 程式碼涵蓋 OCP 微縮放（microscaling）與 FP8 轉換。它的
[MX 實作](https://github.com/amd/Quark/blob/f7d8cefc7a6c973ff90cb87a6b154cbe3cc9aef2/quark/torch/kernel/mx/triton.py)
對 MXFP4 等格式以 32 個值為一個區塊：

1. 對區塊歸約，求出最大絕對值；
2. 由此推出共用的 E8M0 縮放因子（2 的整數次方）；
3. 每個值除以這個縮放因子；
4. 捨入到最接近的 E2M1 值；
5. 每兩個 4 位元值打包成一個位元組；
6. 把縮放因子重排（swizzle）成下游運算所需的配置。

[`mxfp4_qdq.py`](examples/15-triton-k3/mxfp4_qdq.py) 實作了步驟 1–4，並回傳反量化後的值。
它使用的 E2M1 有限值為 \(\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}\)。

### 4.2 教學範例與正式格式的差距

!!! warning "教學用的量化／反量化不是 checkpoint 轉換器"

    相容的 checkpoint 必須完全符合 Quark 的縮放因子捨入方式、NaN 與零的處理規則、
    半位元組（nibble）順序、補齊方式與縮放因子重排。需要互通時，請直接使用 Quark
    匯出的 `qdq_mxfp4_triton` 或 `dq_mxfp4_triton`。本範例只抽出數值概念，
    讓你不必載入完整模型也能測試。

Quark 還有一條 Triton 的 E5M3 轉換路徑，明確處理次正規數與「捨入到最近偶數」。
這些 kernel 屬於模型最佳化與模擬量化（fake quantization）的範疇；KDA、MoE 路由與
生產環境的專家 GEMM 都不歸 Quark 管。

## 5. 階段 2：搬移快取與遞迴狀態

### 5.1 以槽位索引快取

離線範例可以照批次順序存放狀態，推論服務卻不行。請求在不同時間抵達、也在不同時間
結束，所以排程器會替每個進行中的請求配置一個常駐的快取槽位（slot）。

[`state_cache.py`](examples/15-triton-k3/state_cache.py) 示範了核心模式：

```python
row = tl.program_id(0)
slot = tl.load(slots_ptr + row)
cols = tl.arange(0, BLOCK)
values = tl.load(source_ptr + row * width + cols, mask=cols < width)
tl.store(cache_ptr + slot * width + cols, values, mask=cols < width)
```

同樣的模式也出現在 KV 快取、KDA 遞迴狀態、Mamba 狀態與推測解碼的中繼資料。
生產環境的 kernel 還會加上頁面偏移、layer 與 head 的跨距、量化儲存，以及從請求
中繼資料取得的邊界。

兩個寫入落在同一個槽位就會產生競爭條件，所以包裝函式會拒絕重複的槽位。正式的
排程器若不能保證槽位唯一，就必須定義原子或有序的更新方式。

### 5.2 KV 快取與遞迴狀態不同

KV 快取與遞迴狀態不能混為一談。MLA 以 token 為索引儲存 key 與 value，通常切成頁面；
KDA 則把一個固定形狀的矩陣狀態從前一個 token 傳給下一個。兩者都需要「請求 → 槽位」
的間接對應，但配置方式、生命週期與回溯規則都不一樣。

## 6. 階段 3A：KDA decode 的一次遞迴更新

對單一 token，簡化後的 KDA 狀態更新為

$$
D_t = \operatorname{Diag}(\alpha_t)S_{t-1},
$$

$$
r_t = v_t - D_t^\mathsf{T}k_t,\qquad
S_t = D_t + \beta_t k_t r_t^\mathsf{T},\qquad
o_t = S_t^\mathsf{T}q_t.
$$

| 符號 | 形狀 | 意義 |
|---|---:|---|
| \(q_t,k_t,\alpha_t\) | \(K\) | query、key 與逐通道衰減 |
| \(v_t,r_t,o_t\) | \(V\) | value、預測殘差與輸出 |
| \(S_t,D_t\) | \(K\times V\) | 遞迴狀態，以及衰減後的狀態 |
| \(\beta_t\) | 本教學形式中為純量 | 更新強度 |

[`kda_step.py`](examples/15-triton-k3/kda_step.py) 啟動二維網格：第一軸對應
batch × head，第二軸對應一個 \(V\) 方向的分塊。每個程式實例載入完整的 \(K\) 維度，
以及自己負責的那幾欄狀態：

```python
decayed = state * alpha[:, None]
residual = v - tl.sum(decayed * k[:, None], axis=0)
updated = decayed + k[:, None] * (beta * residual)[None, :]
out = tl.sum(updated * q[:, None], axis=0)
```

狀態維持 FP32，因為捨入誤差會隨著遞迴不斷累積；調校過的 kernel 則可以讓 Q、K、V
使用較低精度。實際的 K3 路徑還會把輸入擷取、Q/K 的 L2 正規化、有界衰減、beta 激活
與輸出閘控一起融合進來。

這個教學形式是為了讓遞迴關係一目了然，並不表示所有生產環境的 K3 decode 都用 Triton。
SGLang 會依裝置、模式、資料型別與已安裝的擴充套件，分派到 Triton 版 KDA、FlashKDA、
CuTe DSL、FlashInfer、Helion 或其他支援的後端。

## 7. 階段 3B：分段式 prefill 與推測解碼的狀態

### 7.1 Decode 是遞迴運算

Decode 每次只收到一個新 token，上面的遞迴 kernel 正好適用：讀一份狀態、更新、
寫回一份狀態。連續批次處理（continuous batching）會替每個請求提供快取槽位。

### 7.2 Prefill 是分段運算

Prefill 一次收到大量提示詞 token。如果照 decode 的方式逐一處理，GPU 大部分時間都會閒著。
分段式（chunkwise）KDA 改為：

1. 計算 gate 的前綴和；
2. 算出段內的 QK 與 KK 乘積；
3. 建立 WY 表示法並解三角方程組；
4. 在段與段之間傳遞狀態；
5. 重建輸出。

[FLA `ops/kda`](https://github.com/fla-org/flash-linear-attention/tree/fa06b39153e09b2347387fff990d7d2d07ddd04b/fla/ops/kda)
底下的公開實作展示了完整演算法，也支援多個變長輸入打包在一起。硬把它縮成一個短小的
教學 kernel，反而會把重要的不變量藏起來，所以本章把呼叫正式 API 留作進階練習。

分段大小是調校參數，不會改變模型本身。分段太小，段間傳遞狀態的開銷就高；分段太大，
暫存空間變多，也可能降低佔用率。打包在一起的變長提示詞還需要偏移量與遮罩，確保
某個請求絕不會讀到別的請求的 token。

### 7.3 推測解碼與 ReplaySSM

草稿 token 可能被拒絕，所以在驗證完成前，kernel 不能覆寫已提交的狀態。常見做法是
保留中間快照、替驗證樹上的每個節點挑出父狀態，最後只把被接受的 token 併入主狀態。
SGLang 的 ReplaySSM 路徑多加了一個環形緩衝區，避免每份狀態都要重算。排程器提供
父節點與已接受 token 的中繼資料，kernel 據此收集、重播並提交對應的狀態。

安全規則很簡單：推測階段可以更新暫時狀態，但只有被接受的 token 才能更新常駐的
KDA 狀態或 MLA 的 KV 快取。

## 8. 階段 3C：MLA 注意力

K3 有 24 層使用帶閘控的 MLA 而非 KDA，這條路徑的狀態模型完全不同：

1. 對 Q 與 K 做投影與正規化；
2. 對位置相關的分量套用 RoPE；
3. 把壓縮後的 KV 資料寫入分頁快取；
4. 讀取每個請求對應的頁面；
5. 執行 prefill 注意力，或 split-KV 的 decode 注意力；
6. 合併各部分的輸出，再套用輸出閘控。

快取轉換、RoPE、正規化、線上 softmax、分段歸約與輸出融合都很適合用 Triton 寫。
至於調校到極致的 MLA 注意力核心，則可能改用 FlashInfer、FlashMLA、CuTe DSL、AITER
或其他針對特定架構的實作。KDA 的狀態是一個遞迴矩陣，MLA 的狀態則是以 token 為索引的
分頁快取；SGLang 的混合注意力層必須同時管理這兩種狀態。

## 9. 階段 4：MoE 路由與資料搬移

注意力之後，K3 會把每個 token 分派到 896 個路由專家中的 16 個。完整流程遠不只
一次分組矩陣乘法：

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

top-k 輔助函式、依專家對齊、重排、量化、分組 GEMM 與加權合併，都很適合用 Triton 實作。
在專家平行（expert parallel）模式下，分派與合併還會跨越多張 GPU。中繼資料雖然不規則，
但 token 依專家分組之後，每個專家的矩陣運算就相當規則。

生產環境的 MoE GEMM 常改用 CUDA/CUTLASS、CuTe DSL、AITER、CK 或廠商函式庫的 kernel，
因為這些路徑能利用特定架構的矩陣指令、資料配置、常駐式排程，以及調校過的低精度
epilogue。Triton 仍適合用於可攜的路徑，以及這些 GEMM 前後的資料搬移。

## 10. 階段 5：量化貫穿整條服務流程

量化會出現在好幾個邊界上：

- checkpoint 轉換時，打包權重與縮放因子；
- 載入權重時，原樣保留半位元組與縮放因子的配置；
- 激活值 kernel 計算逐 token 或逐組的縮放因子；
- GEMM 讀取 MXFP4、MXFP8、FP8、INT8、AWQ 或其他支援的格式；
- 快取 kernel 可能把數值轉成量化後的儲存格式。

不要把名字裡帶「FP4」的格式都當成同一種。縮放因子型別、分組大小、捨入方式、打包、
補齊與重排都屬於介面的一部分。第 4 節的 Quark 範例用來說明 MXFP4 的數值原理，
實際的實體配置則由下游的使用者決定。

## 11. 階段 6：取樣與通訊

### 11.1 取樣

模型最後一層算完後，SGLang 在 GPU 上還有工作要做。取樣可能要套用 temperature 與
懲罰項、對 top-p 或 min-p 分布重新正規化、抽出或拒絕候選 token，以及重建推測解碼的樹。
這些運算由短小的歸約、遮罩、掃描與 gather 組成，用 Triton 就能取代一長串小型框架呼叫。

### 11.2 通訊

張量平行、資料平行、專家平行與序列平行都會帶來通訊。Triton 可以準備對稱記憶體緩衝區、
把殘差或縮放運算與本地複製融合，也能處理序列平行的中繼資料，但它無法取代網路傳輸；
GPU 之間的資料仍由 NCCL、RCCL 或其他通訊後端搬運。

回頭看整個請求，同樣的 Triton 模式一再出現：帶遮罩的載入、逐列歸約與線上歸約、
指標間接存取、穩定重排、分塊 GEMM、掃描與融合。

## 12. 後端分派：為什麼某條路徑不是 Triton

SGLang 會在執行時或模型初始化時選擇運算子的實作。

### 12.1 分派依據

決策可能取決於：

- CUDA 或 ROCm、GPU 架構與已安裝的擴充套件；
- prefill、decode 或推測驗證模式；
- 資料型別、量化方案、head 大小、頁面大小與批次形狀；
- 是否需要圖擷取（graph capture）與分散式執行；
- 調校過的後端是否完整支援該模型功能。

### 12.2 各後端的強項

CUDA/CUTLASS 常用於 NVIDIA 專屬的 Tensor Core 資料配置與手工調校的管線。CuTe DSL
能明確控制資料配置與 MMA，適合做類似的特化。AITER 打包了針對 AMD 調校的 CK、HIP、
組合語言、Triton 等多種實作。FlashInfer 與廠商函式庫則提供有人持續維護的專用運算子。
當可攜性、快速迭代、融合與自訂資料搬移，比榨出最後一點硬體效能更重要時，Triton
最能發揮所長。

### 12.3 常見誤解

因此請記得：

- **FlashKDA 是 CUDA/CUTLASS，不是 Triton。**
- AITER 是分派兼運算子函式庫；呼叫了 AITER，不代表選中的 kernel 是 Triton。
- 一個 `.py` 包裝函式可能啟動的是 CUDA、CuTe DSL、AITER 或函式庫 kernel。
- 回報效能時，必須註明選中的後端、資料型別、形狀與硬體。

請從模型層開始，沿著註冊表、各模式的實作，一路追到最後的 kernel 啟動點。不要只憑
Python 檔名，或 SGLang 其他地方用了 Triton，就推斷這裡也是 Triton。

## 重點整理

1. Quark、K3 與 SGLang 位於技術堆疊的不同層。
2. K3 官方儲存庫沒有 Triton 原始碼，FlashKDA 則是 CUDA/CUTLASS 專案。
3. KDA decode 是遞迴的狀態更新；KDA prefill 是分段的矩陣演算法。
4. MLA 使用分頁的 token 快取，KDA 則攜帶固定形狀的遞迴狀態。
5. Triton 在整條服務流程中都派得上用場，但 SGLang 可能分派到更專用的後端。

## 13. 整合進生產環境

教學用 kernel 要成為 SGLang 的後端，必須先把介面約定講清楚：

1. 明定支援的裝置、資料型別、形狀、模式與量化格式；
2. 讓包裝函式檢查跨距、對齊、快取槽位與中繼資料；
3. 以明確的分派條件註冊這個實作；
4. 對不支援的輸入，保留一個確定正確的備援路徑；
5. prefill、decode 與推測解碼的狀態介面分別接上；
6. 讓狀態更新具交易性：只提交被接受的 token；
7. 在日誌或指標中顯示選中的後端；
8. SGLang、Triton、PyTorch 與 ROCm/CUDA 的版本一起固定。

除非某種格式已經過驗證，否則已提交的遞迴狀態應保持 FP32。Quark 的打包方式與縮放
因子重排屬於資料格式本身，不能隨意更動。NVIDIA 與 AMD 必須分開測試：原始碼可攜，
不代表分塊形狀、編譯結果或調校參數也相同。提到 K3 時請沿用其授權名稱；「開放權重」
比籠統的「開源」更精確。

## 14. 測試

在儲存庫根目錄執行現有範例：

```bash
cd tutorials/examples/15-triton-k3
python3 test_model_kernels.py
```

沒有 GPU 時，腳本會自動設定 `TRITON_INTERPRET=1`。它會檢查長度不是 2 的次方的
SiTU 輸入、KDA 更新後的狀態與輸出、全零與不完整的 MX 區塊，以及索引式快取的來回寫讀。

生產環境的測試還應涵蓋：

- 奇數的 head 數與隱藏維度、不滿一頁的資料、全空的遮罩與零個 token 的工作；
- 打包的變長提示詞，以及 prefill 與 decode 混合的批次；
- 重複或無效的快取槽位，以及請求中途取消；
- 推測分支被接受零個、部分與全部 token 的情況；
- 量化的邊界值、打包順序、補齊與縮放因子配置；
- 每個啟用的後端結果一致，以及遇到不支援形狀時的備援；
- 多 rank 的專家分派、合併與通訊失敗。

計時之前先確認數值正確。重新結合順序的歸約需要明確的容許誤差；遞迴測試則要比對
每一個中間狀態，不能只看最後一個 token。

## 15. 效能分析

量測前先讓 JIT 與快取暖機。在 NVIDIA 上，可以這樣檢查小型 KDA 範例：

```bash
ncu -k regex:kda_step_kernel python3 test_model_kernels.py
```

在 AMD 上請改用對應的 ROCm 效能分析工具。除了單一 kernel，也要分析完整的 SGLang
請求。prefill 吞吐量、decode 延遲與推測驗證要分開量，因為三者的形狀與瓶頸都不同。

至少要記錄：

- 選中的後端與確切的軟體版本；
- 裝置、資料型別、量化方式、批次、序列長度與快取形狀；
- kernel 時間、啟動次數、記憶體流量、佔用率與暫存器溢出；
- 分散式執行時的通訊時間與重疊程度；
- 暖機後的端對端延遲與吞吐量。

單獨量測較快的 kernel，若需要額外的型別轉換、快取搬移、同步或分派開銷，端對端反而可能更慢。

## 練習

1. 把 Q/K 的 L2 正規化融合進 `kda_step_kernel`，平方和保持 FP32。
2. 擴充狀態快取，讓請求、layer 與 head 各有獨立的跨距。
3. 每個位元組打包兩個 E2M1 編碼，再與 Quark 0.12 的位元組順序比對。
4. 在 KDA 參考實作外加一層序列迴圈，驗證每個中間狀態，而不只是最終輸出。
5. 用打包的變長提示詞呼叫固定版本的 FLA 分段式 KDA API，並與遞迴 decode 的結果比對。
6. 設計一個 ReplaySSM 測試，讓被接受的前綴長度分別為 0、1 與完整草稿長度。
7. 畫出連續批次處理下，一個 KDA 層與一個 MLA 層的快取配置。
8. 在你的機器上追蹤 SGLang 為 K3 選擇後端的過程。列出哪些運算子用了 Triton、
   CUDA/CUTLASS、CuTe DSL、AITER、FlashInfer 或廠商函式庫，並說明每個備援的理由。
