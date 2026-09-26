---
title: 二維卷積
platform: Tensara
upstream: conv-2d
url: https://tensara.org/problems/conv-2d
difficulty: medium
tags: [convolution, shared-memory, tiling]
status: solved
---

# 二維卷積

**平台：** Tensara · **難度：** medium · [題目敘述](https://tensara.org/problems/conv-2d)

## 問題

對 $H\times W$ 的 float32 影像，以奇數大小的
$K_h\times K_w$ 核心和零填補執行「same」二維互相關。測試範圍從使用 $13\times13$ 核心的 $16384^2$ 影像，到使用
$127\times127$ 核心的 $4096^2$ 影像，因此不論核心大小，共享記憶體用量都必須維持在固定範圍內。檢查誤差為 `rtol = 2e-4`、`atol = 1e-3`。

## 公式

$$
C[i, j] = \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} \tilde{A}\bigl[i + u - p_h,\ j + v - p_w\bigr]\; B[u, v], \qquad
p_h = \frac{K_h - 1}{2},\quad p_w = \frac{K_w - 1}{2}
$$

| 符號 | 意義 |
|---|---|
| $A$ | 輸入影像 $H\times W$，列優先排列；$\tilde{A}$ 在 $A$ 的邊界外為零 |
| $B$ | 核心 $K_h\times K_w$，兩個維度皆為奇數 |
| $p_h, p_w$ | 垂直和水平填補量（核心半徑） |
| $C$ | 輸出影像 $H\times W$ |
| $i, j$ | 輸出的列與欄 |
| $u, v$ | 核心的列與欄 |

對 $T_y\times T_x$ 像素的輸出圖塊及由 $b$ 個核心列組成的帶狀區域，該帶狀區域會存取的輸入視窗為：

$$
(T_y + b - 1) \times (T_x + K_w - 1)
$$

| 符號 | 意義 |
|---|---|
| $T_y, T_x$ | 輸出圖塊高度與寬度（32 × 32） |
| $b$ | 每趟處理的核心列數（8） |

## 方法

1. 使用 **$32\times32$ 輸出圖塊**及 $32\times8$ 執行緒區塊，每個執行緒在暫存器中保留 4 個輸出列。
2. **每次處理 8 個核心列。**每個帶狀區域中，執行緒區塊會將該區域的核心列（$8\times K_w$）及其輸入視窗
   （$39\times(31 + K_w)$，以零填補）放入共享記憶體。當
   $K_w = 127$ 時，視窗包含 $39\times158$ 個浮點數，因此所有測試案例的共享記憶體用量都低於 29 KB。
3. **內層迴圈**：對每個取樣點 $(u, v)$，一次權重的廣播載入及四次共享記憶體載入會供應四次 FMA。`threadIdx.x` 代表欄，因此 32 條通道會讀取 32 個連續字組（無衝突）。
4. 核心函式將**尾聲函式物件**作為範本參數，讓
   [Conv2D + ReLU + HardSwish](../conv2d-relu-hardswish/) 可重複使用此實作。

## 成本分析

$$
W = 2HWK_hK_w, \qquad
Q \approx 4HW\left(1 + \left\lceil \frac{K_h}{8} \right\rceil \frac{39\,(31+K_w)}{32\cdot 32}\right)\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 浮點運算次數 |
| $Q$ | 通過 L2 的位元組數：每個圖塊、每個帶狀區域各有一個暫存視窗（比例為視窗面積除以圖塊面積），再加上輸出寫入 |

對使用 $127^2$ 個取樣點的 $4096^2$ 影像而言，
$W = 541$ GFLOP：需要數秒的 FP32 運算，因此算術成本遠高於其他成本。對使用 $13^2$ 個取樣點的 $16384^2$ 影像而言，
$W = 91$ GFLOP，而必要流量為 2 GB，仍受限於運算效能。和一維版本相同，每次 FMA 都需要一次共享記憶體載入；沿 $x$ 的暫存器圖塊化，以及可分離或 FFT 方法（當 $K \ge 63$）是主要的最佳化方向。

## 常見陷阱

- **大型核心的共享記憶體**：為 $127\times127$ 核心暫存完整視窗需要 $158^2\cdot 4 = 100$ KB／區塊；分帶處理可讓用量維持固定。
- 視窗超出影像邊緣時應**填零**，不可將索引限制在邊界上。
- 每個帶狀區域需要**兩次 `__syncthreads`**：一次在覆寫圖塊前，一次在載入後。

## 驗證

所有測試案例（官方大小的縮小版本）皆已在
[cuemu](../../tools/cuemu/README.md) 上通過，並與 PyTorch 參考實作比對。

## 相關內容

- [一維卷積](../conv-1d/)、[Conv2D + ReLU + HardSwish](../conv2d-relu-hardswish/)、
  [方框模糊](../box-blur/)、LeetGPU [二維卷積](../../leetgpu/010-2d-convolution/)。
