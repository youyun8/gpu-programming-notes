---
title: 累積總和
platform: Tensara
upstream: cumsum
url: https://tensara.org/problems/cumsum
difficulty: medium
tags: [scan, prefix-sum, fp64-accumulation]
status: solved
---

# 累積總和

**平台：** Tensara · **難度：** medium · [題目敘述](https://tensara.org/problems/cumsum)

## 問題

計算長度為 $N$（64 K … 1 M）的 float32 向量之內含式前綴和，行為與 `torch.cumsum(x, dim=0)` 相同。檢查誤差為
`rtol = 3e-2`、`atol = 1e-2`。這是經典的掃描問題；解法以泛型方式撰寫一次，並由 [cumprod](../cumprod/) 重複使用。

## 公式

$$
y_i = \sum_{j=0}^{i} x_j
$$

| 符號 | 意義 |
|---|---|
| $x_j$ | 輸入元素，$0 \le j < N$ |
| $y_i$ | 輸出：前 $i+1$ 個輸入的總和 |

使用每段 $L$ 個元素的分段時，區塊 $c$ 涵蓋 $[cL, (c+1)L)$，且

$$
T_c = \sum_{j=cL}^{(c+1)L-1} x_j, \qquad E_c = \sum_{c'=0}^{c-1} T_{c'}, \qquad
y_i = E_c + \sum_{j=cL}^{i} x_j \quad (cL \le i < (c+1)L)
$$

| 符號 | 意義 |
|---|---|
| $L$ | 分段長度，$256 \times 8 = 2048$ |
| $T_c$ | 分段 $c$ 的總和 |
| $E_c$ | 傳入分段 $c$ 的互斥進位值 |

在一個執行緒區塊內，warp 層級掃描使用 Hillis–Steele 遞迴關係：

$$
v^{(s+1)}_\ell = \begin{cases} v^{(s)}_{\ell - 2^s} + v^{(s)}_\ell, & \ell \ge 2^s \\ v^{(s)}_\ell, & \text{otherwise} \end{cases}, \qquad s = 0, \dots, 4
$$

| 符號 | 意義 |
|---|---|
| $\ell$ | 通道索引，$0 \dots 31$ |
| $v^{(s)}_\ell$ | 步驟 $s$ 後，通道 $\ell$ 的部分總和（使用 $2^s$ 作為 `__shfl_up_sync` 的位移量） |

## 方法

1. **`chunkTotals`**：每個執行緒加總其 8 個連續元素，再對 256 個執行緒的值執行區塊掃描，得到 $T_c$。
2. **`scanTotals`**：使用單一區塊，以每次 256 個值的方式掃描總值（最多 512 個），在各趟之間保留累計總值，並寫入互斥進位值 $E_c$。
3. **`scanChunks`**：每個執行緒重新載入其 8 個項目，區塊掃描每個執行緒的總和（先以 warp shuffle 處理，再由一個 warp 掃描 8 個 warp 總值），接著每個執行緒以
   $E_c + (\text{exclusive prefix of its thread})$ 為起點，依序走訪其 8 個項目。
4. 所有累加器皆使用 `double`，因此在 $10^6$ 次加法中不會累積捨入誤差；每個元素的結果只在最後捨入為 float 一次。

## 成本分析

$$
Q = 12N\ \text{bytes}, \qquad W = 2N\ \text{adds (plus } O(N/8)\text{ in the block scans)}, \qquad T_{\min} = \frac{12N}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取兩次、寫入一次 |
| $W$ | 加法次數 |
| $\beta$ | DRAM 頻寬 |

當 $N = 2^{20}$ 時，資料流量約需 6 µs；三次啟動和中間的單區塊核心函式所需時間也大致相同。單趟解耦回看掃描只讀取一次資料，可將流量降至 $8N$ 位元組。

## 常見陷阱

- 使用**內含式**而非互斥式：$y_0 = x_0$。
- **fp32 誤差累積**：若使用 float 進位值，相對誤差會隨 $N$ 增長；寬鬆的誤差容許範圍能掩蓋此問題，而 fp64 可消除此問題。
- **讀取共享總值後執行 `__syncthreads()`**：掃描輔助函式會在迴圈中呼叫，因此不可在其他 warp 仍讀取 `warp_totals` 時覆寫它。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 相關內容

- [累積乘積](../cumprod/)、[一維移動總和](../running-sum-1d/)、
  LeetGPU [前綴和](../../leetgpu/016-prefix-sum/)、
  LeetGPU [分段前綴和](../../leetgpu/070-segmented-prefix-sum/)。
