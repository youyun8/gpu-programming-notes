---
title: 影像直方圖
platform: Tensara
upstream: histogram
url: https://tensara.org/problems/histogram
difficulty: easy
tags: [histogram, atomics, shared-memory, privatization]
status: solved
---

# 影像直方圖

**平台：** Tensara · **難度：** easy · [題目說明](https://tensara.org/problems/histogram)

## 問題

計算 $h\times w$ 灰階影像中有多少像素落在 $n_b$ 個分箱中的每一個
（$n_b$ = 64、128 或 256；影像最大為 $4096\times4096$）。像素值以
float 儲存，但其值為整數；參考實作會將它們限制在 $[0, n_b - 1]$
範圍內，並使用 `torch.bincount`。計數以 float 回傳，並以完全相等進行
比較。

## 公式

$$
\text{bin}(v) = \bigl\lfloor \min\bigl(\max(v, 0),\ n_b - 1\bigr) \bigr\rfloor, \qquad
H[k] = \sum_{i=0}^{h-1}\sum_{j=0}^{w-1} \mathbf{1}\bigl[\text{bin}(I_{ij}) = k\bigr]
$$

| 符號 | 意義 |
|---|---|
| $I_{ij}$ | 像素值 |
| $n_b$ | 分箱數量（`num_bins`） |
| $\text{bin}(v)$ | 值 $v$ 限制範圍後的分箱索引 |
| $\mathbf{1}[\cdot]$ | 指示函數：條件成立時為 1，否則為 0 |
| $H[k]$ | 分箱 $k$ 的計數，$0 \le k < n_b$ |

私有化會依區塊拆分計數，再加總各份私有副本：

$$
H[k] = \sum_{b=0}^{G-1} H_b[k], \qquad H_b[k] = \sum_{p \in \mathcal{P}_b} \mathbf{1}\bigl[\text{bin}(I_p) = k\bigr]
$$

| 符號 | 意義 |
|---|---|
| $G$ | 區塊數量 |
| $\mathcal{P}_b$ | 區塊 $b$ 所走訪的像素 |
| $H_b$ | 區塊 $b$ 位於共享記憶體中的私有直方圖 |

## 方法

1. **`zeroBins`** 清除輸出（後續會累加至其中）。
2. **`histogramKernel`**：每個區塊先將共享的 `unsigned` 直方圖歸零，
   再以網格跨步迴圈走訪像素，並在共享記憶體中執行
   `atomicAdd(&s_hist[bin], 1)`。共享原子操作會在 SM 內解決，且只有
   同一區塊內會互相競爭。
3. 經過同步屏障後，每個區塊會對全域直方圖中**非零**的分箱各執行一次
   float `atomicAdd`。
4. 若 $n_b > 8192$，備援作法會直接執行全域原子操作（測試不會使用）。

float 可精確表示至 $2^{24} = 16.7$ M 的計數，正好等於 $4096^2$，因此
最大測試剛好位於極限，仍能維持精確。

## 成本分析

$$
Q = 4hw + 4n_b\ \text{bytes}, \qquad \#\text{global atomics} \le G\,n_b, \qquad \#\text{shared atomics} = hw
$$

| 符號 | 意義 |
|---|---|
| $Q$ | DRAM 位元組數：只讀取影像一次 |
| $G n_b$ | 區塊合併至全域的次數（每個區塊對每個分箱至多一次） |
| $hw$ | 每個像素執行一次共享記憶體原子操作 |

在 $4096^2$ 時：共 67 MB，以 2 TB/s 計算約需 34 µs。使用 64 個分箱
與自然影像時，warp 中許多 lane 會命中同一分箱，造成共享原子操作序列化；
若每個 warp 使用一份私有直方圖（共享記憶體中的分箱 × warp 數），可進一步
減少競爭。

## 注意事項

- **將輸出歸零**：全域原子操作會累加至原有內容。
- **轉型前先限制範圍**：不在 $[0, n_b - 1]$ 內的值必須像
  `torch.clamp` 一樣落入兩端分箱。
- **精確性**：檢查器要求完全相等，因此計數必須是整數（不能取平均，
  也不能在 $2^{24}$ 以內產生浮點漂移）。

## 驗證

所有測試案例（官方尺寸的縮小版本）都已在
[cuemu](../../tools/cuemu/README.md) 上通過，結果與 PyTorch 參考實作一致。

## 相關內容

- [灰階轉換](../grayscale/)、[閾值](../threshold/)、
  LeetGPU [直方圖統計](../../leetgpu/013-histogramming/)。
