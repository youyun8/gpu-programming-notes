---
title: 二維 FFT
platform: LeetGPU
upstream: medium/78_2d_fft
url: https://leetgpu.com/challenges/2d-fft
difficulty: medium
tags: [fft, transpose, shared-memory, complex]
status: solved
---

# 二維 FFT

**平台：** LeetGPU · **難度：** 中等 · [題目說明](https://leetgpu.com/challenges/2d-fft)

## 題意

計算 $M\times N$ 複數 float32 訊號的二維 DFT。資料以交錯的 (re, im) 配對
依列優先順序儲存（$M, N \le 4096$；基準測試 $M = N = 2048$；容許誤差
`1e-2`），結果應與 `torch.fft.fft2` 相符。二維 DFT 可分離為先沿列、再沿欄
執行的一維 DFT。工程上的問題在於如何讓兩個階段都能連續讀取記憶體。

## 圖解

![以列–行法計算二維 FFT：列 FFT、轉置、再做列 FFT](figure.svg)

每個階段都完整掃過矩陣一次。轉置把行方向的 FFT 變成列方向，因此每一輪 FFT 都讀取連續的記憶體。

## 數學表述

$$
X_{uv} = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} x_{mn}\,\omega_M^{um}\,\omega_N^{vn}
= \sum_{m=0}^{M-1} \omega_M^{um}\underbrace{\sum_{n=0}^{N-1} x_{mn}\,\omega_N^{vn}}_{Y_{mv}\ (\text{row DFTs})}, \qquad \omega_L = e^{-2\pi i/L}
$$

| 符號 | 意義 |
|---|---|
| $M,\ N$ | 列數與欄數 |
| $x_{mn}$ | 複數輸入樣本；實部位於 `2(mN+n)`，虛部位於 `2(mN+n)+1` |
| $X_{uv}$ | 頻譜係數 |
| $\omega_L$ | 第 $L$ 次本原單位根 |
| $Y_{mv}$ | 中間結果：第 $m$ 列的一維 DFT |

**列－欄演算法：** 對每一列執行 DFT（$M$ 個長度為 $N$ 的轉換），再對結果
的每一欄執行 DFT（$N$ 個長度為 $M$ 的轉換）。使用 FFT 時，成本為
$O(MN\log(MN))$。

### 共享記憶體中的基數 2 時域抽取

當 $L$ 是 2 的冪次時，先將輸入排列成**位元反轉**順序，再執行
$\log_2 L$ 個蝶形階段。在半長度為 $h$ 的階段：

$$
\begin{aligned}
u &= s[i_0], \quad v = s[i_1]\cdot\omega_{L}^{\,p\cdot L/(2h)}, \qquad i_0 = 2hg + p,\ \ i_1 = i_0 + h \\
s[i_0] &\leftarrow u + v, \qquad s[i_1] \leftarrow u - v
\end{aligned}
$$

| 符號 | 意義 |
|---|---|
| $s$ | 保存在共享記憶體中的一列 |
| $h$ | 目前子轉換大小的一半：$1, 2, 4, \dots, L/2$ |
| $g,\ p$ | 蝶形群組及其群組內位置（$0 \le p < h$） |
| $\omega_L^{p L/(2h)}$ | 旋轉因子 $= e^{-2\pi i p/(2h)}$ |

## 解題思路

1. 對 $M$ 列執行 **`fftRows`**（每列一個區塊，512 個執行緒）：
   - *2 的冪次*：以位元反轉順序載入共享記憶體
     （`__brev(i) >> (32 - log2 L)`），接著執行 $\log_2 L$ 個階段、
     每階段 $L/2$ 個蝶形，並在階段間設置同步屏障；
   - *其他大小*：每個輸出直接計算 $O(L^2)$ DFT（測試中只會出現較小的
     非 2 冪次大小）。
2. 以分塊的 32 × 32 共享記憶體轉置（利用填補避免 bank 衝突），執行
   **轉置**（$M\times N \to N\times M$），使各欄變成連續的列。
3. 對轉置矩陣的 $N$ 列執行 **`fftRows`**（長度 $M$）。
4. **轉置回去**並寫入 `spectrum`。

旋轉因子透過 `sincospif(-2(k mod L)/L)` 計算。整數化簡將引數維持在
$[0, 2)$，而 `sincospif` 可避免乘上經過捨入的 $\pi$。在 $L = 4096$
時，兩者都很重要。

一列 4096 個複數值會使用 32 KB 共享記憶體。啟動器會明確選用這個動態大小。

## 成本分析

$$
W \approx 5MN\log_2(MN), \qquad Q \approx \underbrace{2\cdot 8MN}_{\text{row FFTs}}\cdot 2 + \underbrace{2\cdot 8MN}_{\text{transposes}}\cdot 2 = 64MN\ \text{bytes}
$$

| 符號 | 意義 |
|---|---|
| $W$ | 實數 FLOP 數（基數 2 估計值） |
| $Q$ | DRAM 位元組數：每次 FFT 與每次轉置都會讀寫完整的 $M\times N$ 複數陣列（每個元素 8 位元組） |

當大小為 $2048^2$ 時，$W \approx 0.46$ GFLOP，$Q \approx 270$ MB，也就是
約 135 µs 的流量。每列 FFT 全程在共享記憶體中執行，因此每個階段只需
2 次全域存取。將欄 FFT 與轉置融合（直接透過共享記憶體處理欄面板），
可省下一半流量。

## 常見陷阱

- **跨距式欄 FFT。** 直接讀取欄時，每次存取相隔 $N$ 個元素，無法合併。
  轉置可解決此問題。
- 大 $L$ 下的**旋轉因子精度**（見上文）。
- **非 2 冪次大小**會退回 $O(L^2)$。若要處理任意大型尺寸，應使用
  Bluestein（參見 [FFT](../039-fast-fourier-transform/)）。

## 驗證

所有 LeetGPU 測試案例皆在 [cuemu](../../tools/cuemu/README.md) 上以 `1e-2`
通過，包括 $1 \times N$、$M \times 1$ 與非 2 冪次形狀。

## 延伸閱讀

- [快速傅立葉轉換（一維、任意長度）](../039-fast-fourier-transform/)、
  [矩陣轉置](../003-matrix-transpose/)。
