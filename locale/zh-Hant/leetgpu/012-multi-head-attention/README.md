---
title: 多頭注意力
platform: LeetGPU
upstream: hard/12_multi_head_attention
url: https://leetgpu.com/challenges/multi-head-attention
difficulty: hard
tags: [attention, flash-attention, multi-head, online-softmax]
status: solved
---

# 多頭注意力

**平台：** LeetGPU · **難度：** hard · [題目說明](https://leetgpu.com/challenges/multi-head-attention)

## 題意

計算不含投影的多頭自注意力。$Q, K, V$ 是
$N \times d_{\text{model}}$ float32 矩陣，沿欄方向分成 $h$ 個寬度
$d_k = d_{\text{model}}/h$ 的頭（$1 \le N \le 10^4$，
$2 \le d_{\text{model}} \le 1024$，$1 \le h \le d_{\text{model}}$；
基準 $N = d_{\text{model}} = 1024$；容許誤差 `1e-5`）。
各頭重新串接成 $N \times d_{\text{model}}$ 輸出。難點是 $d_k$
可能從 1（$h = d_{\text{model}}$）到 1024（$h = 1$），
無法為每列採用固定的暫存器或共享記憶體預算。

## 圖解

![多頭注意力：每個 head 是 Q、K、V 中的一段欄位，以步幅定址](figure.svg)

每種顏色代表一個 head，也就是 Q、K、V 中寬度為 dₖ 的一段欄位。各 head 各自計算注意力，再寫回輸出的相同欄位，所以切分與串接 head 都不必搬移資料。

## 數學表述

$$
\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(H_0, \dots, H_{h-1}), \qquad
H_i = \operatorname{softmax}_{\text{row}}\!\left(\frac{Q_i K_i^{\mathsf T}}{\sqrt{d_k}}\right) V_i
$$

$$
Q_i = Q[:,\ i d_k : (i+1) d_k], \quad K_i = K[:,\ i d_k : (i+1) d_k], \quad V_i = V[:,\ i d_k : (i+1) d_k]
$$

| 符號 | 意義 |
|---|---|
| $N$ | 序列長度（$Q$、$K$、$V$ 與輸出的列數） |
| $d_{\text{model}}$ | 模型寬度（欄數） |
| $h$ | 頭數 |
| $d_k$ | 頭寬度 $d_{\text{model}}/h$ |
| $Q_i,\ K_i,\ V_i$ | 輸入的第 $i$ 個欄區塊，各為 $N \times d_k$ |
| $H_i$ | 第 $i$ 頭的輸出（$N \times d_k$），寫入欄 $[i d_k, (i+1)d_k)$ |
| $\operatorname{softmax}_{\text{row}}$ | 對每列獨立套用的 Softmax |

每列的線上 softmax 更新沿用 [Softmax 注意力](../006-softmax-attention/)
所推導的目前最大值 $m$、分母 $\ell$、累加器 $\mathbf a$，
以及修正因子 $\alpha = e^{m - m'}$。

### 將頭視為跨距

第 $i$ 頭的元素 $(r, c)$ 位於偏移量
$r \cdot d_{\text{model}} + i\,d_k + c$。因此一個頭只是一個基底指標
（$+\,i\,d_k$）和列跨距（$d_{\text{model}}$）。拆分與串接各頭
**不需搬移資料**。核心接收小型 `AttnGeom` 結構（列數、頭維度、列跨距、
每頭偏移量、縮放值），並由因果注意力、GQA 及其他注意力變體重用。

## 解題思路

### 網格

使用 $\lceil N/4\rceil \times h$ 個區塊，每區塊 128 個執行緒。
`blockIdx.y` 選擇頭，每個 warp 處理一個查詢列。

### 切分頭維度

$d_k$ 最大可達 1024，但 $32 \times 1024$ 的浮點分塊需要 128 KB
共享記憶體。因此核心透過共享記憶體，以**每片 128 欄**串流處理 K 與 V：

1. **分數。** 對每個 32 鍵分塊，走訪
   $c_0 = 0, 128, \dots$。暫存 $K$ 的 $32 \times 128$ 切片
   （間距 129 以避免 bank 衝突），讓 lane $\ell$ 將查詢切片與鍵
   $\ell$ 的部分點積累加。最後一片完成後，lane $\ell$ 擁有完整分數
   $s_\ell$。
2. **線上 Softmax。** 每個分塊各做一次 `warpMax` 與 `warpSum`，
   再將修正值 $\alpha$ 套用到累加器。
3. **$PV$。** 再次走訪切片：暫存 $V$ 的 $32 \times 128$ 切片，
   以 `__shfl_sync` 廣播各 $p_j$，更新該切片的累加器暫存器。

### 將累加器保留在暫存器

每個 lane 最多負責 $8 \text{ slices} \times 4 = 32$ 個輸出欄。
切片迴圈以編譯期上限（`kMaxSlices = 8`）套用 `#pragma unroll`，
並在 $c_0 \ge d_k$ 時提早中止。因此 `acc[]` 的每個索引都是編譯期常數，
陣列會留在暫存器，不會溢出到區域記憶體。

當 $d_k = 1024$ 時，共享記憶體為
$4 d_k + 32\cdot129 + 32\cdot128$ 個浮點數，約 49 KB，超過預設 48 KB，
因此啟動器以 `cudaFuncSetAttribute(…MaxDynamicSharedMemorySize…)` 提高上限。

## 成本分析

$$
W = 4N^2 d_{\text{model}}, \qquad
Q_{\text{DRAM}} \approx 4\left(2Nd_{\text{model}} + \left\lceil\frac{N}{4}\right\rceil \cdot 2N d_{\text{model}}\right)
$$

| 符號 | 意義 |
|---|---|
| $W$ | FLOP：所有頭中，分數需要 $2N^2d_k$、$PV$ 需要 $2N^2d_k$，再乘以 $h$ |
| $Q_{\text{DRAM}}$ | 位元組數：讀取 $Q$ 並寫入輸出一次；每個 4 列區塊串流讀取該頭的全部鍵和值 |

當 $N = d_{\text{model}} = 1024$，$W \approx 4.3$ GFLOP。
$K$ 與 $V$（8 MB）可放入 L2，因此 $\lceil N/4\rceil$ 次重讀多由 L2 提供。
限制因素是透過共享記憶體執行的 FP32 FMA 吞吐量。自然的下一步是張量核心版本
（bf16/tf32 `mma.sync` 搭配 64 查詢分塊）。

## 常見陷阱

- **頭索引。** 列跨距是 $d_{\text{model}}$，不是 $d_k$。
  使用 $d_k$ 會讓第一頭以外的所有頭讀錯元素。
- **$d_k < 32$。** 許多 lane 沒有輸出欄；`c < width` 防護會讓它們閒置，
  但它們仍須參與 shuffle 與屏障。
- **超過 48 KB 的動態共享記憶體**需要明確設定屬性，否則啟動會無聲失敗。
- **縮放。** 應為 $1/\sqrt{d_k}$，不是 $1/\sqrt{d_{\text{model}}}$。

## 驗證

所有 LeetGPU 案例都在 [cuemu](../../tools/cuemu/README.md) 上通過，
包括 $h = d_{\text{model}}$（$d_k = 1$）與 $h = 1$（$d_k = 1024$）。
另以壓力測試將 $d_k = 1024$ 與 PyTorch 比較。

## 延伸閱讀

- [Softmax 注意力](../006-softmax-attention/)、[多頭交叉注意力](../026-multi-head-cross-attention/)、
  [GQA](../080-grouped-query-attention/)、[MLA](../114-multi-head-latent-attention/)。
