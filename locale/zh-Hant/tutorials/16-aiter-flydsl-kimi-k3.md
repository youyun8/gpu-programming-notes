# 16 – AITER 中的 FlyDSL – 在 AMD GPU 上執行 Kimi K3

> **第六部 · Kimi K3 案例** · 先備知識：
> [矩陣乘法 1 – 基礎](04-tiled-matmul.md)與
> [矩陣乘法 8 – Tensor Core](gemm/07-tensor-cores.md)、
> [Triton – 從第一個 Kernel 到生產環境](14-triton.md)，以及
> [CDNA3 與 MFMA](05-amd-cdna3-mfma.md) ·
> 上一章：[15 – SGLang 中的 Triton：以 Kimi K3 推論服務為例](15-triton-model-systems.md) ·
> 下一章：[08 – 部署本站](08-deploying-this-site.md)

本章沿著一條實際上線的路徑，從 Kimi K3 的 checkpoint 一路追到 AMD GPU 上執行的
kernel。重點是經由 AITER 呼叫的 FlyDSL kernel，但每一層軟體的分界都會講清楚：

```text
Quark converts checkpoint data
    → SGLang, vLLM, or ATOM schedules a request
        → AITER exposes and dispatches operators
            → FlyDSL, Triton/Gluon, CK, HIP, assembly, or Opus runs a kernel
```

Quark 不是推論執行環境；AITER 不是單一種 kernel 語言；FlyDSL 也沒有實作 AITER 的
每一個運算子。尤其要注意：本章介紹的公開 K3 配方中，Kimi Delta Attention（KDA）
仍然走 Triton/Gluon 路徑。

**你將學會**

- Kimi K3、Quark、推論執行環境、AITER 與 FlyDSL 各自扮演的角色；
- FlyDSL 的張量、資料配置（layout）、複製與 MMA 模型；
- AITER 如何結合分派、JIT 與 AOT 打包，以及調校表；
- 量化格式與預先重排（preshuffle）的儲存方式，如何成為 kernel 的介面約定；
- K3 MoE 從路由到加權合併的完整路徑；
- 哪些面向 K3 的公開 AITER 運算子使用 FlyDSL，哪些沒有；
- MLA、專家平行與通訊如何搭配 MoE 路徑；
- 如何先測好單一運算子，再跑八張 GPU 的 MAD 配方。

## 1. 固定公開軟體的版本

本章的範例與原始碼連結使用：

- [AITER v0.1.23](https://github.com/ROCm/aiter/tree/v0.1.23)；
- [FlyDSL v0.3.4.1](https://github.com/ROCm/FlyDSL/tree/v0.3.4.1)；
- [Quark release/0.12](https://github.com/amd/Quark/tree/release/0.12)。

AITER v0.1.23 綁定 FlyDSL 0.3.4.1。FlyDSL 改版很快，請以對應 tag 的範例為準，
不要直接照抄線上最新文件裡的 API。

### 1.1 檢查機器

建置前先確認環境：

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

ROCm 版 PyTorch 刻意沿用 `torch.cuda` 命名空間，請不要把它改成 `torch.rocm`。

| 目標架構 | 本章可以合理期待的支援程度 |
|---|---|
| `gfx942`（MI300X/MI325X） | 許多 AITER 運算子，以及實用的本機 kernel 測試 |
| `gfx950`（MI350X/MI355X） | 已發布的八張 GPU K3 配方所針對的架構 |
| RDNA 系列 | 本章多數路徑仍屬實驗性質或不支援 |

單一運算子能跑，不代表整個 2.8 兆參數的模型放得進去，也不代表已發布的配方支援那張 GPU。

## 2. 每一層只做一件事

| 層級 | 負責的工作 | 它「不」保證什麼 |
|---|---|---|
| Kimi K3 模型 | 定義架構、權重、896 個路由專家、top-16 路由、KDA 與帶閘控的 MLA | 使用哪一種 AMD kernel 後端 |
| Quark | 量化並轉換 checkpoint 張量 | 請求排程或執行期分派 |
| SGLang、vLLM 或 ATOM | 管理批次、快取、圖擷取、平行 rank 與後端選擇 | 選中的每個 AITER 運算子都是 FlyDSL |
| AITER | 提供可從 PyTorch 呼叫的 AMD 運算子，並挑選實作 | 所有運算子都用同一種語言實作 |
| FlyDSL | 用 Python 描述資料配置、複製、MMA 運算與啟動方式 | 一套完整的模型服務系統 |

這樣分層也是除錯的利器：轉換後 checkpoint 裡的縮放因子錯了，改 MFMA 分塊也修不好；
本機 GEMM 算對了，也不能證明專家平行的中繼資料有正確交換。

## 3. FlyDSL：從數值到 MFMA

FlyDSL 是一種 Python DSL，明確區分裝置端程式與主機端的啟動程式：

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

具體 API 會持續變動，但背後的思考方式很穩定。

### 3.1 張量與資料配置

`fx.Tensor` 是一段儲存空間的檢視（view）。它的資料配置把邏輯座標對應到實體位移。
資料配置可以描述全域記憶體中的分塊、LDS 分塊或暫存器片段（fragment），也可以拆分、
組合維度。

每個階段都要問自己：

> 這個執行緒（或這個 wave）擁有哪些邏輯上的值？它們存放在哪裡？

有了這個問題，就不必再手寫索引算式。資料配置是正確性的一部分，而不只是效能提示。

### 3.2 複製原子與分塊複製

複製原子（copy atom）描述一種合法的搬移方式，例如一次搬 128 位元的向量複製。
分塊複製（tiled copy）把這個原子重複套用到更大的邏輯分塊上，並把各部分分給不同執行緒。
來源與目的地的切分方式，對「哪個值歸誰」必須有一致的定義。

在邊緣分塊上，複製仍需要判斷式（predicate）。只有在對齊、有效 lane 與儲存配置都符合
約定時，寬向量操作才安全。

### 3.3 MMA 原子與分塊 MMA

MMA 原子代表一條硬體矩陣指令，在 CDNA 上通常就是一次 MFMA。分塊 MMA 把這個原子展開到
更大的輸出分塊上，並定義：

- 哪些 lane 持有 A 與 B 的片段；
- 每個 lane 擁有哪些累加器的值；
- wave 如何沿 M、N、K 重複工作；
- 暫存器片段如何銜接全域記憶體或 LDS 的複製。

kernel 通常會把複製與一次次的 MMA 步驟排成管線，最後套用 epilogue 並寫回結果。
這正是[分塊矩陣乘法](04-tiled-matmul.md)中建立的矩陣路徑，只是改用資料配置代數來表達。

### 3.4 用四個範例循序學習

```bash
git clone --branch v0.3.4.1 https://github.com/ROCm/FlyDSL.git
cd FlyDSL
pip install flydsl==0.3.4.1 pytest pandas
python3 examples/01-vectorAdd.py
```

| 步驟 | 對應 tag 的範例 | 新概念 |
|---|---|---|
| 1 | [`examples/01-vectorAdd.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/01-vectorAdd.py) | 張量、資料配置、判斷式與 128 位元複製 |
| 2 | [`examples/02-tiledCopy.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/02-tiledCopy.py) | 全域記憶體／LDS 分塊與執行緒切分 |
| 3 | [`examples/03-tiledMma.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/03-tiledMma.py) | 暫存器片段與分塊 MFMA |
| 4 | [`examples/04-preshuffle_gemm.py`](https://github.com/ROCm/FlyDSL/blob/v0.3.4.1/examples/04-preshuffle_gemm.py) | 實際上線的權重配置與 GEMM |

不要一開始就啃完整的 MoE kernel。先用這四個範例追蹤一個分塊的旅程：
全域記憶體 → LDS 或暫存器 → MMA 片段 → 輸出。

## 4. AITER 的分派、編譯與調校

AITER 提供給執行環境一個穩定的運算子介面，背後選中的實作可能是 FlyDSL、Triton/Gluon、
Composable Kernel、HIP、組合語言、Opus 或其他程式產生器。

分派可能取決於：

- GPU 架構；
- 輸入、權重、累加器與輸出的型別；
- 量化方式與縮放因子格式；
- M、N、K、token 數、頁面大小與注意力模式；
- 跨距、對齊、是否已預先重排等資料配置資訊；
- 環境變數的覆寫設定；
- 是否有吻合的調校設定。

因此：

> 「執行環境用了 AITER」不等於「執行環境用了 FlyDSL」。

### 4.1 JIT

FlyDSL 的包裝函式可以在第一次呼叫時針對參數特化並編譯 kernel，之後由 AITER 快取結果。
冷啟動的第一次呼叫包含 Python 分派、程式碼產生、編譯與載入；暖機後的呼叫則應直接重用
已編譯的特化版本。

計時前一定要先暖機；比較不同執行結果時，也要記下快取狀態。

### 4.2 AOT

AITER 也有 AOT 清單（manifest），其中包含 MoE 系列的公開 FlyDSL 清單。清單描述可在模型
執行前就建置、打包好的 kernel。AOT 能減少啟動工作，但只涵蓋已匯出的特化版本；沒涵蓋到的
形狀仍可能需要 JIT 編譯，或改走其他後端。

請參考對應 tag 的
[`aiter/aot/flydsl`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/aot/flydsl)
目錄。清單裡有某個 kernel，不代表執行時真的選中了它。

### 4.3 已調校與未調校的設定表

[`aiter/configs/model_configs`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/configs/model_configs)
中專為 K3 準備的檔案，涵蓋下列公開格式：

- BF16 GEMM；
- A8W8、B 矩陣預先重排的 GEMM；
- A4W4、區塊縮放的 GEMM；
- A16W4、A8W4、A4W4 與 I4 的融合 MoE；
- FP8 FMHA 的 AOT 設定。

這個目錄同時有已調校（tuned）與未調校（untuned）的版本。已調校的一列，是在某個形狀與
架構上實測後挑出的選擇，並不表示同一個分塊在任何情況下都最好。

`AITER_FLYDSL_FORCE=1` 會拿掉部分備援選項，但不會憑空替不支援的運算子、形狀或架構
生出一個 FlyDSL 實作。

## 5. 面向 K3 的公開運算子目錄

下表整理固定版本的公開原始碼中，與 K3 相關的 FlyDSL/AITER 用途。「可用」表示存在公開的
運算子或調校路徑，不代表每個推論框架在每次 K3 請求中都會選它。

| K3 用途 | 公開的 AITER/FlyDSL 實作 | 界線 |
|---|---|---|
| 稠密投影 | BF16 GEMM、A8W8 B 預先重排 GEMM 與 A4W4 區塊縮放 GEMM 的調校 | AITER 可能選 FlyDSL、組合語言或其他後端 |
| top-k 與專家中繼資料 | top-k、MoE 排序、路由表，以及分組／本地查找輔助函式 | 路由策略屬於模型與執行環境 |
| 量化並分散路由後的列 | [`moe_fused_route_quant_scatter.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py)、scatter-copy、縮放因子預先重排的輔助函式 | 列的順序、目的地、資料與縮放因子必須一起符合約定 |
| 專家 GEMM 1 | [`mxfp4_gemm1.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm1.py)與 MoE 第一階段包裝函式 | 產生 gate/up 資料；支援的格式與形狀由分派決定 |
| 混合格式的兩階段 MoE | [`mixed_moe_gemm_2stage.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py) | 涵蓋公開的混合格式路徑，但不是所有融合 MoE 路徑 |
| SiTUv2 加上重新量化 | AITER `situv2_and_mul_quant` | 公開的 AITER 運算子；沒確認實際建置版本前，別把它標成 FlyDSL |
| 專家 GEMM 2 | [`mxfp4_gemm2.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/mxfp4_gemm2.py)與 MoE 第二階段包裝函式 | 讀取激活值及其縮放因子 |
| 專家輸出合併 | FlyDSL gather/reduce、MoE reduce、dispatch/combine 與 MegaMoE 輔助函式 | 本地合併與跨 rank 合併是兩回事 |
| MLA 前置處理 | [`qk_norm_rope_quant.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/kernels/qk_norm_rope_quant.py)、KV gather／B 投影 | 前置處理用 FlyDSL，不代表注意力核心也是 |
| MLA 注意力與合併 | [`fmha_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/fmha_kernels.py)、分頁／FP8 MLA 路徑，以及 [`mla_reduce_kernels.py`](https://github.com/ROCm/aiter/blob/v0.1.23/aiter/ops/flydsl/mla_reduce_kernels.py) | AITER 另有組合語言、Triton 與 Opus 的 MLA 路徑 |
| 專家平行的資料搬移 | 節點內 dispatch/combine、融合通訊的 MoE、MegaMoE 與 quick all-reduce | 需要一致的 rank 中繼資料、IPC 設定與拓撲 |
| KDA | AITER 的 Triton/Gluon KDA 實作 | 本章不主張存在公開的 K3 FlyDSL KDA 路徑 |

AITER 裡其他通用的 FlyDSL kernel，例如卷積、HSTU、與此無關的 GEMM 及其他模型專用運算子，
都不在這份 K3 目錄內。反過來說，這份目錄也不代表 K3 全部都跑在 FlyDSL 上。

## 6. 量化與預先重排

K3 原生的量化感知訓練（QAT）格式，使用 MXFP4 權重與 MXFP8 激活值。AMD 的另一條 checkpoint
路徑則在注意力層結合 MXFP4 與逐 token、逐通道的 FP8。格式名稱同時描述了算術方式與儲存方式。

| 格式 | 儲存的內容 | 縮放因子的處理 |
|---|---|---|
| FP8 | 每個元素一個 8 位元浮點數 | 找出最大絕對值、推出縮放因子、轉換 |
| INT8/INT4 | 有號整數加上一個縮放因子 | 歸約、捨入、截斷，有時還要打包 |
| MXFP8 | 一小塊 FP8 值共用一個縮放因子 | 逐區塊歸約，並排好區塊縮放因子 |
| MXFP4 | 打包的 E2M1 值，共用一個 E8M0 區塊縮放因子 | 歸約、編碼、半位元組打包，並重排縮放因子 |

確切的區塊大小、縮放因子方向與位元組順序，都由運算子的約定決定；不要只憑「4 位元」
三個字自行推斷。

### 6.1 為什麼需要預先重排

數學上的權重矩陣以 \(W[k,n]\) 索引，但一個 MFMA wave 並不是照單純的列優先順序讀取它。
預先重排就是依照複製與 MMA 資料配置需要的順序來存放權重，必要時連縮放因子一起重排。

這樣可以：

- 讓 wave 的載入變成連續存取；
- 省去執行期反覆的重新排列；
- 減少 LDS 的 bank 衝突；
- 把打包的半位元組與縮放因子放在用得到它們的片段旁邊。

預先重排不是免費的轉置。checkpoint 轉換器（或準備步驟）、選中的 GEMM、縮放因子配置
與硬體架構四者必須一致。只要其中一項約定對不上，kernel 照樣能跑，卻會算出看似合理
實則錯誤的結果。

### 6.2 先驗證格式，再碰 GEMM

先執行：

```bash
cd aiter
python3 op_tests/test_quant_mxfp4.py
python3 op_tests/test_gemm_a8w8.py
```

測試項目要包含全零區塊、最大有限值、捨入中點、不完整的分組、奇數列數與空輸入。
在量 GEMM 速度之前，先把反量化後的值與 FP32 參考結果比對。

## 7. 完整的 K3 MoE 路徑

K3 有 896 個路由專家，每個 token 選 16 個，另外還有共享專家。路由專家這條路徑如下：

```text
router logits
  → grouped top-k and routing weights
  → route + quantize + scatter
  → expert GEMM 1 (gate/up)
  → SiTUv2 + activation requantization
  → expert GEMM 2 (down)
  → weighted reduce/combine
```

每個箭頭都是一個可以單獨測試的邊界。

### 7.1 路由

路由器輸出選中的專家編號與權重。執行環境的策略可能還會重新正規化權重，並把全域專家編號
對應到本地 rank。AITER 的路由與排序輔助函式，會把這種稀疏的選擇轉成分組 GEMM 能直接
使用的中繼資料。

要檢查：

- token 與專家的順序是否穩定；
- 全域與本地專家編號各代表什麼；
- 若呼叫端可能重複選同一個專家，如何處理；
- 沒有分到 token 的專家，以及零個 token 的批次；
- 容量、補齊與對齊；
- 路由權重的資料型別與正規化方式。

路由幾乎不花計算量，但只要一個索引錯了，整層的結果就全毀了。

### 7.2 量化與分散

融合後的 route/quant/scatter kernel 會讀入一列 token、算出所需的激活縮放因子、轉換數值，
再把這一列寫進依專家分組的儲存區。在專家平行模式下，它還要替其他 rank 準備資料與中繼資料。

以下四項輸出要一起驗證：

1. 目的地專家或 rank；
2. 目的地列號；
3. 量化後的資料；
4. 用來還原這一列的縮放因子。

只測量化後的位元組，會漏掉路由錯誤；只測列對照表，會漏掉「縮放因子配錯 token」的問題。

### 7.3 專家 GEMM 1

GEMM 1 對每個被選中的專家套用 gate 與 up 投影。即使整體批次很大，單一專家實際分到的
M 仍可能很小而且參差不齊，kernel 必須在 MFMA 效率、補齊浪費與啟動開銷之間取捨。

公開的 MXFP4 與混合格式兩階段 kernel，替支援的格式與形狀組合提供了 FlyDSL 路徑；
某次呼叫到底有沒有用上，仍由分派決定。

在這個邊界上，請以分散後的列順序比對輸出。先別把列合併回 token 順序，否則可能掩蓋
排序錯誤。

### 7.4 SiTUv2 與激活值重新量化

設 gate 輸入為 \(g\)、up 輸入為 \(u\)，K3 使用：

$$
y =
\left[\beta_1\tanh(g/\beta_1)\sigma(g)\right]
\left[\beta_2\tanh(u/\beta_2)\right],
\qquad \beta_1=4,\quad \beta_2=25.
$$

| 符號 | 意義 |
|---|---|
| \(g,u\) | GEMM 1 輸出的 gate 與 up 兩半 |
| \(\sigma\) | Sigmoid 函數 |
| \(\beta_1,\beta_2\) | K3 的上下界 |
| \(y\) | 融合後的專家激活輸出 |

AITER 的 `situv2_and_mul_quant` 把這個算式、逐列縮放因子的計算與 FP8 轉換融合在一起：

```python
from aiter.ops.activation import situv2_and_mul_quant

out = torch.empty((tokens, width), device="cuda", dtype=aiter.dtypes.fp8)
scale = torch.empty((tokens, 1), device="cuda", dtype=torch.float32)
situv2_and_mul_quant(out, x, scale, width, 4.0, 25.0)
```

把 `out.float() * scale` 與公式的 FP32 實作比對。測試中要保留全零的列與空批次：

```bash
python3 op_tests/test_situv2_and_mul_quant.py -d bf16 -m 1 -n 768
```

這是 AITER 運算子層級的邊界；光看公開 API，無法斷定它的實作是 FlyDSL。

### 7.5 專家 GEMM 2

GEMM 2 讀入量化後的激活值與逐列縮放因子，再套用每個專家的 down 投影。它讀取列的順序，
必須與 scatter 和 GEMM 1 產生的順序完全一致。

請在乘上路由權重之前就比對它的輸出。這裡若有差異，不要等到最後合併完的張量才開始除錯。

### 7.6 加權合併

合併階段把結果還原成 token 順序，將每個選中專家的輸出乘上對應的路由權重，再把 16 份貢獻
加總。它也可以在模型規定的位置加上共享專家的結果。

在專家平行模式下，合併同時也是一個通訊問題：

```text
local expert outputs
  → send or reduce across ranks
  → gather contributions for each original token
  → multiply by routing weights
  → sum in token order
```

先檢查一個 token 的一份專家貢獻，再檢查全部 16 份，最後才把完整的本地或分散式合併結果，
與高精度參考結果比對。

## 8. MLA 支援與 KDA 的界線

K3 混用帶閘控的多頭潛在注意力（MLA）層與 KDA 層，兩者不必使用同一個後端。

### 8.1 公開的 FlyDSL/AITER MLA 元件

固定版本的 AITER 原始碼包含下列公開路徑：

- Q/K 正規化、RoPE 與量化；
- KV gather 與 B 投影；
- FP8 FMHA；
- 支援之頁面配置下的分頁 MLA；
- 分段注意力輸出，以及 log-sum-exp 歸約。

這些都是實際存在的 FlyDSL/AITER 用途，但支援範圍受架構、資料型別、head 形狀、頁面大小，
以及 decode 或 prefill 模式限制。AITER 也附帶非 FlyDSL 的 MLA 實作，所以請檢查分派結果，
別把 MLA 的時間全算在 FlyDSL 頭上。

### 8.2 KDA 仍然走 Triton/Gluon

KDA 是本章最明確的後端界線。AITER 公開的 K3 KDA 實作，位於對應 tag 的
[`kimi_delta_attn`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/ops/triton/kimi_delta_attn)、
[`gated_delta_net`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/ops/triton/gated_delta_net)
與
[`chunk_delta_attn`](https://github.com/ROCm/aiter/tree/v0.1.23/aiter/ops/triton/_triton_kernels/chunk_delta_attn)
等 Triton/Gluon 實作中。原始碼裡雖然也有通用的 FlyDSL gated-delta 輔助程式，但那不足以證明
存在完整的 K3 FlyDSL KDA 後端，本章也不這麼主張。

已發布的 SGLang K3 指令雖然啟用了 AITER，並強制在支援的地方使用 FlyDSL，卻仍然傳入：

```bash
--attention-backend triton
```

這並不矛盾：同一個請求可以讓 FlyDSL 處理 MoE 與周邊運算子，同時由 Triton/Gluon 負責 KDA
與選定的注意力後端。

## 9. 通訊與專家平行

張量平行把稠密運算切開；專家平行則把不同專家放到不同 rank 上。K3 在分組 GEMM 前後，
都需要一致的路由中繼資料與資料搬移。

### 9.1 相關的公開通訊元件

與此設計相關的公開 FlyDSL/AITER 通訊實作包括：

- 節點內的 dispatch/combine；
- 融合通訊的 MoE；
- MegaMoE 的 dispatch、GEMM 與 combine 階段；
- 類似 reduce-scatter/all-gather 的支援；
- 支援格式下的 quick all-reduce 路徑。

原始碼裡有，不代表某個 K3 配方實際選用了它；執行期旗標、world size、拓撲、資料型別與
形狀仍會左右選擇。

### 9.2 正確性要點

為了確保正確：

- 所有 rank 對 token 與專家的中繼資料必須一致；
- 傳送與接收的筆數要正確涵蓋沒分到 token 的專家；
- 本地與全域的專家編號不可混用；
- 可重複使用的緩衝區必須由 stream 與 event 保護；
- IPC handle 與 peer 存取必須有效；
- 圖擷取需要固定的位址與固定的啟動拓撲；
- 歸約順序與資料型別都可能改變數值誤差。

先在一張 GPU 上跑通，再用手寫的路由表測兩個 rank。等「空專家」與「負載極不平均」的
情況都通過後，才擴大到完整的 rank 數。

## 10. Quark 的職責止於 checkpoint

AMD 的 K3 轉換路徑可以直接量化 checkpoint，不必先用傳統方式把整個模型載入記憶體：

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

Quark 只寫出資料與中繼資料，不負責挑選執行期的 kernel。如果 checkpoint 的打包方式或縮放
因子配置錯了，AITER 也不會替你重新解讀。

下列項目應視為一組、一起記錄，才能重現結果：

- 來源 checkpoint 的名稱與版本；
- Quark 版本與 K3 範本；
- 量化方案與各層的覆寫設定；
- 輸出 checkpoint 的名稱；
- 推論服務映像檔，以及 AITER/FlyDSL 版本；
- 目標 GPU 架構。

某個版本可能提供模擬舊資料配置的相容性旗標，這在過渡期很有用，但不該把它說成 MXFP4
本身的永久特性。

## 11. 本機運算子的測試階梯

別一開始就上八張 GPU。請一階一階往上爬，而且每一筆計時結果旁邊，都要有對應的正確性結果。

### 11.1 FlyDSL 語言的冒煙測試

```bash
cd FlyDSL
python3 examples/01-vectorAdd.py
python3 -m pytest tests/kernels/test_vec_add.py
python3 -m pytest tests/kernels/test_preshuffle_gemm.py -m "not large_shape"
python3 -m pytest tests/kernels/test_quant.py
python3 -m pytest tests/kernels/test_moe_gemm.py
python3 -m pytest tests/kernels/test_flash_attn_fwd.py
```

只測編譯或降階（lowering）的測試，無法證明 GPU 上的數值正確。判讀「通過」之前，
先看清楚測試的標記與目標平台。

### 11.2 AITER 運算子測試

安裝與 ROCm、Python 版本相符的 wheel，或從固定版本的原始碼建置：

```bash
git clone --recursive --branch v0.1.23 https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

接著一次只測一個邊界：

```bash
python3 op_tests/test_quant_mxfp4.py
python3 op_tests/test_gemm_a8w8.py
python3 op_tests/test_situv2_and_mul_quant.py -d bf16 -m 1 -n 768
python3 op_tests/test_moeTopkSoftmax.py
python3 op_tests/test_moe_2stage.py
python3 op_tests/test_mla.py
python3 op_tests/test_rmsnorm2d.py
```

利用各測試支援的參數，補上下列情況：

1. 零個 token；
2. 一個 token；
3. 奇數與邊界上的維度；
4. 空專家，以及負載嚴重不均的專家；
5. 一組與 K3 實際上線完全相同的形狀；
6. 冷啟動與暖機後的啟動時間。

### 11.3 多 GPU 測試

本機 MoE 全部通過後，才開始跑通訊測試。FlyDSL 的 all-reduce 測試在
`tests/kernels/test_allreduce.py`；AITER 另有獨立的多 GPU 運算子測試。啟動前，
請確認測試預期的 world size 與拓撲符合你的機器。

## 12. 八張 GPU 的 MAD 配方

### 12.1 需求與支援的框架

公開的
[ROCm MAD K3 配方](https://github.com/ROCm/MAD/blob/develop/benchmark/kimi_k3/README.md)
是完整模型指令與映像檔的權威來源。它需要：

- 八張 MI350X 或 MI355X GPU（`gfx950`）；
- 已發布的啟動設定使用 tensor parallel size 8；
- 約 1.56 TB 的空間存放 checkpoint。

MAD 針對三種框架發布了 K3 的執行方式：

```bash
# vLLM
madengine run --tags pyt_vllm_kimi-k3 --keep-model-dir --live-output

# SGLang
madengine run --tags pyt_sglang_kimi-k3 --keep-model-dir --live-output

# ATOM
madengine run --tags pyt_atom_kimi-k3 --keep-model-dir --live-output
```

### 12.2 SGLang 中的 FlyDSL 路徑

在以 FlyDSL 為主的 SGLang 路徑中，已發布的容器設定了：

```bash
SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1
AITER_FLYDSL_FORCE=1
AITER_SITUV2_A8W4=1
```

並以下列指令啟動伺服器，其中可以清楚看到關鍵的後端界線：

```bash
sglang serve --model-path /model_weights \
  --trust-remote-code --tp-size 8 \
  --attention-backend triton --dtype bfloat16 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 256 \
  --host 127.0.0.1 --port 30000 \
  --disable-radix-cache \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```

### 12.3 什麼才算證據

請直接使用 MAD 提供的映像檔與完整指令，不要把不同版本的旗標拼湊在一起。本機 FlyDSL
測試通過是必要的證據，但無法取代「實際載入 checkpoint、提供服務並檢查模型輸出」這一步。

## 13. 效能分析與除錯

請依照以下順序進行：

1. **參考結果。** 把單一邊界的輸出與 FP32 PyTorch 實作比對。
2. **格式。** 確認打包的值、縮放因子形狀、縮放因子的歸屬，以及預先重排的版本。
3. **中繼資料。** 檢查 token 列、專家編號、rank 編號與空專家。
4. **分派。** 記錄架構、運算子、選中的後端與調校表中的那一列。
5. **編譯。** 把冷啟動的 JIT 時間與暖機後的執行時間分開，並保留失敗特化版本的編譯錯誤訊息。
6. **kernel 分析。** 用 `rocprofv3` 檢查啟動間隙、執行時間、記憶體流量、MFMA 使用率、
   佔用率與 LDS 行為。
7. **通訊分析。** 找出 rank 之間的負載不均、被迫序列化、缺少重疊，以及多餘的複製。
8. **端對端。** 量測服務的延遲與吞吐量，再檢查輸出品質。

常見的失敗模式：

| 症狀 | 優先檢查 |
|---|---|
| BF16 正確，量化後的輸出卻錯 | 縮放因子方向、區塊大小、打包方式與預先重排 |
| 只有某一個專家錯 | 路由表、本地專家編號，以及該專家的縮放因子／權重位移 |
| 只有最終結果錯 | 路由權重與合併順序 |
| 第一次呼叫特別慢 | JIT 編譯與快取路徑 |
| 強制 FlyDSL 時失敗 | 該形狀與架構是否真的有支援的 FlyDSL 路徑 |
| 某個 rank 卡住 | 傳送／接收筆數、空專家、stream event 與集體通訊的順序 |
| 小測試都過，圖擷取卻失敗 | 緩衝區位址是否固定、暖機是否涵蓋所有路徑、通訊是否能安全擷取 |

## 14. 上線前檢查清單

在宣稱某條 K3 路徑可以上線之前，請記錄：

- [ ] 確切的 checkpoint、Quark、執行環境映像檔、AITER 與 FlyDSL 版本；
- [ ] GPU 架構、韌體／驅動程式、ROCm、PyTorch 版本與 world size；
- [ ] 每個關鍵運算子選中的後端與調校表中的那一列；
- [ ] 量化、縮放因子、打包與預先重排的約定；
- [ ] 路由、分散、兩個 GEMM、激活與合併各自的本機參考比對結果；
- [ ] 空輸入、奇數、負載不均，以及支援範圍內最大的形狀；
- [ ] 冷啟動與穩定狀態的量測；
- [ ] 圖擷取與備援路徑的行為；
- [ ] 多 rank 的逾時、錯誤傳遞與健康檢查；
- [ ] 權重、快取、JIT/AOT 程式碼與通訊緩衝區所需的記憶體餘裕；
- [ ] 端對端的準確度或任務品質檢查，而不只是 kernel 層級的容許誤差；
- [ ] 在具代表性的提示詞長度與並行數下，收集效能分析紀錄。

## 重點整理

1. FlyDSL 把「值歸誰」、複製方式與 MFMA 資料配置都明確寫出來。
2. AITER 會分派到多種後端；呼叫了 AITER，不能證明用的是 FlyDSL kernel。
3. 公開的 FlyDSL K3 主軸是量化 MoE：路由、量化／分散、GEMM 1、SiTUv2／重新量化、
   GEMM 2 與合併。
4. AITER 提供實用的 FlyDSL MLA 元件；但在本章描述的公開 K3 路徑中，KDA 仍由
   Triton/Gluon 負責。
5. 量化配置、縮放因子、預先重排、調校與通訊中繼資料，全都屬於正確性的一部分。
6. 本機運算子測試與八張 GPU 的 MAD 執行回答的是不同問題；要證明能上線，兩者缺一不可。

## 練習

1. 追蹤 FlyDSL 向量加法中「值歸誰」的關係，標出邏輯座標、實體位移、執行緒、複製原子
   與邊界判斷式。
2. 畫出一個預先重排的 GEMM 分塊，從全域記憶體中的權重一路到 MFMA 的 B 片段，並指出
   checkpoint 轉換器還必須知道哪些資訊。
3. 分別在設定與不設定 `AITER_FLYDSL_FORCE=1` 的情況下，記錄某個 K3 GEMM 形狀的 AITER
   分派結果。解釋時不要假設強制路徑支援所有形狀。
4. 替 SiTUv2 建立 FP32 參考實作，並在零值、上下界與捨入中點上，與反量化後的
   `situv2_and_mul_quant` 輸出比對。
5. 為「路由 → 量化／分散 → GEMM 1 → SiTUv2／重新量化 → GEMM 2 → 合併」畫出緩衝區示意圖，
   標出列順序、資料型別、縮放因子形狀與擁有者。
6. 建立一個兩個 rank 的專家平行測試，其中包含一個空專家與一個熱門專家。執行前先寫出
   預期的傳送與接收筆數。
7. 根據 AITER 的執行紀錄，把每個 K3 運算子歸類為 FlyDSL、Triton/Gluon、組合語言／Opus、
   HIP/CK 或未知。在分派證據確認之前，未知的項目就維持未知。
