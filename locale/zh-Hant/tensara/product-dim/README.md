---
title: 沿維度取乘積
platform: Tensara
upstream: product-dim
url: https://tensara.org/problems/product-dim
difficulty: easy
tags: [reduction, strided-reduction, product]
status: solved
---

# 沿維度取乘積

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/product-dim)

## 問題

對任意秩的 float32 張量沿 `dim` 維度執行 `torch.prod(x, dim, keepdim=True)`，並保留縮減後的維度（`keepdim=True`）。測試會沿不同軸（0、1、2 或 3）縮減形狀從 $(16, 128, 256)$ 到 $(64, 128, 128, 128)$ 的張量。檢查條件為 `rtol = 2e-2`、`atol = 7e-3`。

## 公式

將張量視為三個軸（`dim` 之前的所有維度、要縮減的軸，以及之後的所有維度）：

$$
O = \prod_{k<d} S_k, \qquad R = S_d, \qquad I = \prod_{k>d} S_k, \qquad
x[o, j, i] = \text{in}\bigl[(oR + j)I + i\bigr]
$$

| 符號 | 意義 |
|---|---|
| $S_k$ | 軸 $k$ 的大小；$d$ 即 `dim` |
| $O$ | 外部大小（$d$ 之前各軸大小的乘積） |
| $R$ | 縮減軸的長度 |
| $I$ | 內部大小（$d$ 之後各軸大小的乘積）；也是縮減軸的記憶體步幅 |
| $x[o, j, i]$ | 外部索引 $o$、縮減索引 $j$、內部索引 $i$ 的元素 |

$$
\text{out}[oI + i] = \prod_{j=0}^{R-1} x[o, j, i]
$$

| 符號 | 意義 |
|---|---|
| out | $O\cdot I$ 個結果，索引為 $oI + i$ |

## 方法

縮減運算使用一個小型累加器結構 `Acc`（identity、make、combine、shuffle、finish），接到兩個泛用核心中。所有 `*-dim` 問題以及 [Argmax](../argmax/)/[Argmin](../argmin/) 都共用這套核心：

- **$I = 1$**（縮減連續的最後一軸）：**每個輸出使用一個 warp**。各 lane 跨步讀取一列 $R$ 個連續 float（每次 warp 載入都是一條合併存取的 128 位元組資料線），用 `combine` 合併，再執行五次 `__shfl_down_sync`；lane 0 套用 `finish` 後寫入。
- **$I > 1$**：**每個輸出** $(o, i)$ **使用一個執行緒**，以步幅 $I$ 逐一處理 $j$。相鄰 $i$ 的執行緒會讀取相鄰位址，因此即使各執行緒採跨步讀取，每一步在整個 warp 中仍是合併存取。

依測試框架而定，`shape` 可能是主機或裝置指標，因此使用 `cudaMemcpyDefault` 複製，並在主機上計算 $O, R, I$。輸出會保留大小為 1 的縮減軸（`keepdim=True`），但這不影響記憶體配置：仍以相同順序存放 $O\cdot I$ 個元素。

本題使用 fp32 累加器 `{identity: 1, combine: ×}`。$R$ 個隨機值的乘積很快就會下溢為 0 或上溢為 $\pm\infty$：若輸入的典型大小為 $a$，乘積的行為近似

$$
\ln\Bigl\lvert \prod_j x_j \Bigr\rvert = \sum_j \ln\lvert x_j\rvert \approx R\,\mathbb{E}\bigl[\ln\lvert x\rvert\bigr]
$$

| 符號 | 意義 |
|---|---|
| $\mathbb{E}[\ln\lvert x\rvert]$ | 輸入對數大小的平均值；標準常態分布為負值（約 $-0.64$） |

因此當 $R \ge 200$ 時，PyTorch 與此處的大多數結果都恰好為 0；較寬鬆的容許誤差可涵蓋其餘結果。

## 成本分析

$$
Q = 4\,ORI + 4\,OI\ \text{bytes}, \qquad T_{\min} = \frac{4\,ORI}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：張量讀取一次，每個輸出寫入一個值 |
| $\beta$ | DRAM 頻寬 |

最大的測試 $(64, 128, 128, 128)$ 為 537 MB；在 2 TB/s 下約需 0.27 ms。弱點在於：當 $I = 1$ 且 $R$ 很小時（例如沿 dim 3 處理 $(128, 64, 64, 64)$，此時 $R = 64$），每列一個 warp 會有一半 lane 閒置；而當 $I > 1$ 但 $O\cdot I$ 很小時（例如沿 dim 0 處理 $(32, 512, 512)$，會有 262 K 個執行緒，各自迴圈 32 次），每個輸出的平行度不足。將 $R$ 分給多個執行緒即可改善兩種情況。

## 注意事項

- **單位元為 1**。
- **乘法順序**會改變最後幾個位元（以及發生下溢的位置）；容許誤差已據此設定。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在 [cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 相關內容

- [沿維度取總和](../sum-dim/)、[累積乘積](../cumprod/)。
