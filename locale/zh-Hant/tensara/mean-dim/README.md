---
title: 沿維度取平均值
platform: Tensara
upstream: mean-dim
url: https://tensara.org/problems/mean-dim
difficulty: easy
tags: [reduction, strided-reduction, mean]
status: solved
---

# 沿維度取平均值

**平台：** Tensara · **難度：** 簡單 · [題目說明](https://tensara.org/problems/mean-dim)

## 題意

對任意秩的 float32 張量沿維度 `dim` 執行
`torch.mean(x, dim, keepdim=True)`，並保留歸約後的維度
（`keepdim=True`）。測試會對 $(16, 128, 256)$ 到
$(64, 128, 128, 128)$ 的形狀，沿不同軸（0、1、2 或 3）歸約。
檢查條件為 `rtol = 2e-4`、`atol = 7e-6`。

## 圖解

![沿單一維度取平均（keepdim）：每欄的總和除以 R](figure.svg)

圖中畫出一個外層切片 x[o, :, :]：j 沿著被歸約的軸往下，i 沿著連續的內層軸往右。每一欄（固定 o 與 i）歸約成一個輸出值。

## 數學表述

將張量視為三個軸（`dim` 之前的所有部分、歸約軸、之後的所有部分）：

$$
O = \prod_{k<d} S_k, \qquad R = S_d, \qquad I = \prod_{k>d} S_k, \qquad
x[o, j, i] = \text{in}\bigl[(oR + j)I + i\bigr]
$$

| 符號 | 意義 |
|---|---|
| $S_k$ | 第 $k$ 軸的大小；$d$ 是 `dim` |
| $O$ | 外部大小（$d$ 之前各軸大小的乘積） |
| $R$ | 歸約軸的長度 |
| $I$ | 內部大小（$d$ 之後各軸大小的乘積）；也是歸約軸的記憶體步距 |
| $x[o, j, i]$ | 外部索引 $o$、歸約索引 $j$、內部索引 $i$ 的元素 |

$$
\text{out}[oI + i] = \frac{1}{R}\sum_{j=0}^{R-1} x[o, j, i]
$$

| 符號 | 意義 |
|---|---|
| out | $O\cdot I$ 個結果，索引為 $oI + i$ |
| $1/R$ | 最後套用一次（`finish`） |

## 解題思路

歸約以小型累加器結構 `Acc`（identity、make、combine、shuffle、finish）
插入兩個通用核心中；所有 `*-dim` 問題及
[Argmax](../argmax/)/[Argmin](../argmin/) 都共用這些核心：

- **$I = 1$**（歸約連續的最後一軸）：**每個輸出使用一個 warp**。
  各 lane 跨步走訪含 $R$ 個連續 float 的列（每次 warp 載入都是一條合併的
  128 位元組快取線），使用 `combine` 摺疊，再執行五步
  `__shfl_down_sync`；lane 0 套用 `finish` 並儲存。
- **$I > 1$**：**每個輸出** $(o, i)$ **使用一個執行緒**，以步距 $I$
  迭代 $j$。相鄰 $i$ 的執行緒會讀取相鄰位址，因此即使各執行緒各自跨步，
  每一步在整個 warp 中仍是合併存取。

視測試框架而定，`shape` 可能是主機或裝置指標，因此使用
`cudaMemcpyDefault` 複製，並在主機上計算 $O, R, I$。輸出將歸約軸保留為
大小 1（`keepdim=True`），這不會改變記憶體內容：它仍以相同順序包含
$O\cdot I$ 個元素。

此問題使用總和累加器，並由 `finish` 除以 $R$。

## 成本分析

$$
Q = 4\,ORI + 4\,OI\ \text{bytes}, \qquad T_{\min} = \frac{4\,ORI}{\beta}
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：張量讀取一次，每個輸出寫入一個值 |
| $\beta$ | DRAM 頻寬 |

最大的測試 $(64, 128, 128, 128)$ 為 537 MB：在 2 TB/s 下約 0.27 ms。
弱點在於：當 $I = 1$ 且 $R$ 很小時（例如沿 dim 3 歸約
$(128, 64, 64, 64)$，$R = 64$），每列一個 warp 會有一半 lane 閒置；
而當 $I > 1$ 但 $O\cdot I$ 很小時（例如沿 dim 0 歸約 $(32, 512, 512)$，
只有 262 K 個執行緒且各迭代 32 次），每個輸出的平行度不足。
將 $R$ 拆給多個執行緒可同時改善兩者。

## 常見陷阱

- **嚴格的 `atol = 7e-6`**：$R$ 個標準常態值的平均大小為
  $O(1/\sqrt{R})$，因此相對誤差很重要；應在最後只除一次，
  不要累加 $x/R$（這會多產生 $R$ 次捨入）。
- **除以 $R$**（歸約長度），不是總大小。

## 驗證

所有測試案例（官方尺寸的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考結果比對。

## 延伸閱讀

- [沿維度加總](../sum-dim/)、[層正規化](../layer-norm/)、
  [批次正規化](../batch-norm/)。
