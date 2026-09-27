---
title: 沿維度加總
platform: Tensara
upstream: sum-dim
url: https://tensara.org/problems/sum-dim
difficulty: easy
tags: [reduction, strided-reduction, sum]
status: solved
---

# 沿維度加總

**平台：** Tensara · **難度：** 簡單 · [題目敘述](https://tensara.org/problems/sum-dim)

## 問題

對任意秩的 float32 張量執行 `torch.sum(x, dim, keepdim=True)`，
歸約維度為 `dim`，並保留歸約維度（`keepdim=True`）。
測試會沿不同軸（0、1、2 或 3）歸約從 $(16, 128, 256)$ 到
$(64, 128, 128, 128)$ 的形狀。檢查條件為 `rtol = 2e-4`、`atol = 9e-4`。

## 公式

將張量視為三個軸（`dim` 前的所有內容、歸約軸、其後的所有內容）：

$$
O = \prod_{k<d} S_k, \qquad R = S_d, \qquad I = \prod_{k>d} S_k, \qquad
x[o, j, i] = \text{in}\bigl[(oR + j)I + i\bigr]
$$

| 符號 | 意義 |
|---|---|
| $S_k$ | 第 $k$ 軸的大小；$d$ 是 `dim` |
| $O$ | 外部大小（$d$ 前所有軸大小的乘積） |
| $R$ | 歸約軸的長度 |
| $I$ | 內部大小（$d$ 後所有軸大小的乘積）；也是歸約軸的記憶體步幅 |
| $x[o, j, i]$ | 外部索引 $o$、歸約索引 $j$、內部索引 $i$ 的元素 |

$$
\text{out}[oI + i] = \sum_{j=0}^{R-1} x[o, j, i]
$$

| 符號 | 意義 |
|---|---|
| out | $O\cdot I$ 個結果，索引為 $oI + i$ |

## 方法

歸約使用小型累加器結構 `Acc`（identity、make、combine、shuffle、finish），
並插入兩個通用核心。所有 `*-dim` 問題與
[Argmax](../argmax/)/[Argmin](../argmin/) 都共用這些核心：

- **$I = 1$**（歸約連續的最後一軸）：**每個輸出使用一個 warp**。
  各 lane 跨步讀取一列的 $R$ 個連續 float（每次 warp 載入都是一條合併的
  128 位元組資料線）、以 `combine` 摺疊，再執行五次
  `__shfl_down_sync`；lane 0 套用 `finish` 並儲存。
- **$I > 1$**：**每個輸出 $(o, i)$ 使用一個執行緒**，走訪 $j$ 時使用
  步幅 $I$。相鄰 $i$ 的執行緒讀取相鄰位址，因此即使各執行緒本身跨步，
  每一步在整個 warp 中仍是合併存取。

依測試架構而定，`shape` 可能是主機或裝置指標，因此以
`cudaMemcpyDefault` 複製，並在主機上計算 $O, R, I$。輸出會保留大小為 1
的歸約軸（`keepdim=True`），但記憶體不受影響：仍以相同順序包含
$O\cdot I$ 個元素。

本題的累加器是在 fp32 中使用 `{identity: 0, combine: +}`。因為
$R \le 1024$，每個 lane 或執行緒的 fp32 總和精度足以通過容許誤差。
長度為 $R$ 的總和，其捨入誤差界限為

$$
\Bigl\lvert \widehat{\textstyle\sum} - \textstyle\sum \Bigr\rvert \le (R - 1)\,u \sum_j \lvert x_j\rvert, \qquad u = 2^{-24}
$$

| 符號 | 意義 |
|---|---|
| $\widehat{\sum}$ | 計算所得的總和 |
| $u$ | fp32 單位捨入誤差 |

而 $I = 1$ 核心中的 warp 樹狀歸約可進一步縮短加總鏈。

## 成本分析

$$
Q = 4\,ORI + 4\,OI\ \text{bytes}, \qquad T_{\min} = \frac{4\,ORI}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：讀取一次張量，每個輸出寫入一個值 |
| $\beta$ | DRAM 頻寬 |

最大測試 $(64, 128, 128, 128)$ 為 537 MB，在 2 TB/s 下約為 0.27 ms。
弱點包括：$I = 1$ 且 $R$ 很小時（例如 $R = 64$，對
$(128, 64, 64, 64)$ 沿 dim 3），每列一個 warp 會閒置一半 lane；而 $I > 1$ 但
$O\cdot I$ 很小時（例如 $(32, 512, 512)$ 沿 dim 0 產生 262 K 個執行緒，
各自迴圈 32 次），每個輸出的平行度很低。沿 $R$ 分配執行緒可改善兩者。

## 注意事項

- **`atol = 9e-4`**：1024 個標準常態值的總和大小約為 32，因此數個 ulp
  的誤差約為 $\sim 10^{-5}$，遠低於限制。
- **keepdim** 只會改變形狀中繼資料，不會改變資料配置。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 相關內容

- [沿維度取平均](../mean-dim/)、[沿維度取乘積](../product-dim/)、
  [累積和](../cumsum/)、LeetGPU [歸約](../../leetgpu/004-reduction/)。
